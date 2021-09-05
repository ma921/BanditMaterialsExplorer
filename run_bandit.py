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
from utils.utils import *
from utils.bandit import *

if __name__ == "__main__": 
    # initializing
    target_value = 2.534 # [eV]
    exp, device, dtype = initialising(target_value)
    args = arguments()

    # loading data
    prior_path = './database/prior.csv'
    dim, df_prior, features_prior, features_candidate = read_database(args, prior_path, exp)
    features_prior, features_candidate = normalise(features_prior, features_candidate)    

    # setting experiment parameters
    max_exp = 30 # number of trials
    batch_size = 1 # batch size per trial
    n_exp = 50 # number of experiment to evaluate average learning tendency
    X_init, Y_init, cand_exp = load_parameters(
        args, features_prior, features_candidate, df_prior, dtype, device, exp,
    )

    norm_lc_all = []
    for episode in range(50):
        sys.stdout.flush()
        print('>>>>> '+str(episode)+' / 50')
        exp.set_target(target_value)
        X_chol, Y_chol = run_optimization(
            features_candidate,
            X_init,
            Y_init,
            max_exp,
            batch_size,
            dim,
            cand_exp,
            device,
            dtype,
            args.D,
            args.sampler,
            exp,
            model_type=args.model_type,
        )
        norm_lc = return_result(exp)
        norm_lc_all.append(norm_lc)
        pd.DataFrame(norm_lc_all).to_csv(args.save_csv)

    print('ALL TRAINING PROCESSES HAVE BEEN COMPLETED.')
