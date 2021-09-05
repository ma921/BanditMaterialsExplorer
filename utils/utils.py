import argparse
import sys
import os
import time
import tempfile
import random
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from collections import Counter
from tqdm import tqdm
from contextlib import ExitStack

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.quasirandom import SobolEngine

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report

import gpytorch
import gpytorch.settings as gpts
import botorch
from botorch.fit import fit_gpytorch_model
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.test_functions import Hartmann
from botorch.utils.transforms import unnormalize
from gpytorch.constraints import Interval
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, RFFKernel, ScaleKernel
from gpytorch.kernels.keops import MaternKernel as KMaternKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
import ax.models.torch.alebo

from utils.competition import experiment

def gen_projection(d: int, D: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Generate the projection matrix B as a (d x D) tensor"""
    B0 = torch.randn(d, D, dtype=dtype, device=device)
    B = B0 / torch.sqrt((B0 ** 2).sum(dim=0))
    return B

def loss_func(pred, true, loss_type, p=0.01):
    if loss_type == 'MSE':
        loss = (pred - true)**2
    elif loss_type == 'MAE':
        loss = np.abs(pred - true)
    elif loss_type == 'LogCosh':
        loss = np.log(np.cosh(pred - true))
    elif loss_type == 'MSLE':
        if pred <= -1:
            loss = (pred - true)**2
        else:
            loss = (np.log(true+1)-np.log(pred+1))**2
    elif loss_type == 'RMSLE':
        if pred <= -1:
            loss = (pred - true)**2
        else:
            loss = np.abs(np.log(true+1)-np.log(pred+1))
    elif loss_type == 'poisson':
        if pred <= 0:
            loss = (pred - true)**2
        else:
            loss = (pred - true * np.log(pred))
    elif loss_type == 'tweedie':
        if pred <= 0:
            loss = (pred - true)**2
        else:
            bias = -true * (true**(1-p)) /(1-p) + (true**(2-p)) / (2-p)
            loss = -true * (pred**(1-p)) /(1-p) + (pred**(2-p)) / (2-p) - bias
    return loss

class candidate_exploration():
    def __init__(self, candidate_all, loss_type, p=0.01):
        self.candidate_all = candidate_all
        self.loss_type = loss_type
        self.p = p

    def search_nearest_point(self, target, exp):
        min_dist = 10000
        idx_min = 0
        for idx, ref_point in enumerate(self.candidate_all):
            dist = np.linalg.norm(target - ref_point)
            if dist < min_dist:
                min_dist = dist
                idx_min = idx

        # aquiring the bandgap
        crystal_name = exp.db_df['crystal_name'][idx_min]
        exp.show_answer(crystal_name)
        bandgap = exp.bandgaps[len(exp.bandgaps)-1]
        loss = loss_func(bandgap, exp.target_bandgap, self.loss_type, self.p)
        return loss

def return_result(exp):
    best_result = 100
    norm_lc = []
    for result in exp.learning_curve:
        if result < best_result:
            best_result = result
        norm_lc.append(best_result)
    return norm_lc

def eval_objective(X, cand_exp, exp):
    candidate_features = X.cpu().squeeze().numpy()
    loss = cand_exp.search_nearest_point(candidate_features, exp)
    return 1/loss

def initialising(target_value=2.534):
    exp = experiment()
    exp.set_target(target_value)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double
    return exp, device, dtype

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sampler', type=str, default='ciq')
    parser.add_argument('--loss_type', type=str, default='poisson')
    parser.add_argument('--model_type', type=str, default='ALEBOGP')
    parser.add_argument('--map_csv', type=str, default='./database/descriptors.csv')
    parser.add_argument('--save_csv', type=str, default='BoTorch_best_model.csv')
    parser.add_argument('--p', type=float, default=0.01)
    parser.add_argument('--D', type=int, default=13)
    args = parser.parse_args()
    return args

def read_database(args, prior_path, exp):
    df_prior = pd.read_csv(prior_path, index_col=0)
    df_prior.reset_index(drop=True, inplace=True)
    df_candidates = pd.read_csv(args.map_csv, index_col=0)
    dim = len(df_candidates.columns)
    
    # read features for prior data
    features_prior = np.zeros([len(df_prior), dim])
    for idx in range(len(df_prior)):
        label = df_prior['crystal_name'][idx]
        if label in df_candidates.index:
            index = list(df_candidates.index).index(label)
            df_temp = df_candidates.iloc[index,:]
            features_prior[idx, :] = np.array(df_temp)
    
    # read features for all candidates
    features_candidate = np.zeros([len(exp.db_df), dim])
    for idx in range(len(exp.db_df)):
        label = exp.db_df['crystal_name'][idx]
        if label in df_candidates.index:
            index = list(df_candidates.index).index(label)
            df_temp = df_candidates.iloc[index,:]
            features_candidate[idx, :] = np.array(df_temp)
    return dim, df_prior, features_prior, features_candidate

def normalise(features_prior, features_candidate):
    max_features = max(features_prior.max(), features_candidate.max())
    min_features = min(features_prior.min(), features_candidate.min())
    features_prior = (features_prior - min_features) / (max_features - min_features)
    features_candidate = (features_candidate - min_features) / (max_features - min_features)
    return features_prior, features_candidate

def load_parameters(args, features_prior, features_candidate, df_prior, dtype, device, exp):
    X_init = torch.from_numpy(features_prior).to(dtype=dtype, device=device)
    preds = np.array(df_prior['indirect_bandgap (eV)'])
    Y_init = np.array([
        loss_func(pred, exp.target_bandgap, args.loss_type, p=args.p) for pred in preds
    ])
    Y_init = torch.from_numpy(Y_init[:, np.newaxis]).to(dtype=dtype, device=device)
    cand_exp = candidate_exploration(features_candidate, args.loss_type, p=args.p)
    return X_init, Y_init, cand_exp