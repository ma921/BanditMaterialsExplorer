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

from utils.utils import *

def generate_batch(
    X,
    Y,
    batch_size,
    candidates,
    dim,
    device,
    dtype,
    D,
    sampler="cholesky",  # "cholesky", "ciq", "rff"
    model_type='SingleTaskGP',
    use_keops=False,
):
    assert sampler in ("cholesky", "ciq", "rff", "lanczos")
    assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))

    # NOTE: We probably want to pass in the default priors in SingleTaskGP here later
    if model_type == 'FixedNoiseMultiFidelityGP':
        kernel_kwargs = {"nu": 2.5, "ard_num_dims": 3}
    else:
        kernel_kwargs = {"nu": 2.5, "ard_num_dims": X.shape[-1]}
    if sampler == "rff":
        base_kernel = RFFKernel(**kernel_kwargs, num_samples=1024)
    else:
        base_kernel = (
            KMaternKernel(**kernel_kwargs) if use_keops else MaternKernel(**kernel_kwargs)
        )
    covar_module = ScaleKernel(base_kernel)

    # Fit a GP model
    train_Y = (Y - Y.mean()) / Y.std()
    likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))

    if model_type == 'SingleTaskGP':
        model = botorch.models.SingleTaskGP(X, train_Y, likelihood=likelihood, covar_module=covar_module)
    elif model_type == 'FixedNoiseGP':
        model = botorch.models.FixedNoiseGP(X, train_Y, torch.full_like(train_Y, 0.2), likelihood=likelihood, covar_module=covar_module)
    elif model_type == 'HeteroskedasticSingleTaskGP':
        model = botorch.models.HeteroskedasticSingleTaskGP(X, train_Y, torch.full_like(train_Y, 0.2))
    elif model_type == 'SingleTaskMultiFidelityGP':
        model = botorch.models.SingleTaskMultiFidelityGP(X, train_Y, likelihood=likelihood, data_fidelity=3)
    elif model_type == 'FixedNoiseMultiFidelityGP':
        model = botorch.models.gp_regression_fidelity.FixedNoiseMultiFidelityGP(X, train_Y, torch.full_like(train_Y, 0.2), data_fidelity=3)
    elif model_type == 'MixedSingleTaskGP':
        model = botorch.models.gp_regression_mixed.MixedSingleTaskGP(X, train_Y, cat_dims=[-1], likelihood=likelihood)
    elif model_type == 'ALEBOGP':
        B = gen_projection(
            d=dim, D=D, dtype=dtype, device=device,
        )
        model = ax.models.torch.alebo.ALEBOGP(B=B, train_X=X, train_Y=train_Y, train_Yvar=torch.full_like(train_Y, 0.2))

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)

    # Draw samples on a Sobol sequence
    X_cand = torch.from_numpy(candidates).to(dtype=dtype, device=device)

    # Thompson sample
    with ExitStack() as es:
        if sampler == "cholesky":
            es.enter_context(gpts.max_cholesky_size(float("inf")))
        elif sampler == "ciq":
            es.enter_context(gpts.fast_computations(covar_root_decomposition=True))
            es.enter_context(gpts.max_cholesky_size(0))
            es.enter_context(gpts.ciq_samples(True))
            es.enter_context(gpts.minres_tolerance(2e-3))  # Controls accuracy and runtime
            es.enter_context(gpts.num_contour_quadrature(15))
        elif sampler == "lanczos":
            es.enter_context(gpts.fast_computations(covar_root_decomposition=True))
            es.enter_context(gpts.max_cholesky_size(0))
            es.enter_context(gpts.ciq_samples(False))
        elif sampler == "rff":
            es.enter_context(gpts.fast_computations(covar_root_decomposition=True))

    thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
    X_next = thompson_sampling(X_cand, num_samples=batch_size)

    return X_next

def run_optimization(
    candidates,
    X_init,
    Y_init,
    max_exp,
    batch_size,
    dim,
    cand_exp,
    device,
    dtype,
    D,
    sampler,
    exp,
    model_type='SingleTaskGP',
    use_keops=False,
    seed=None,
):
    X = X_init
    Y = 1/Y_init
    max_evals = len(X) + max_exp
    print(f"{0}) Best value: {(1/Y).min().item():.2e}")

    while len(X) < max_evals:
        # Create a batch
        start = time.time()
        X_next = generate_batch(
            X=X,
            Y=Y,
            batch_size=min(batch_size, max_evals - len(X)),
            candidates=candidates,
            dim=dim,
            device=device,
            dtype=dtype,
            D=D,
            sampler=sampler,
            model_type=model_type,
            use_keops=use_keops,
        )
        end = time.time()
        print(f"Generated batch in {end - start:.1f} seconds")
        Y_next = torch.tensor(
            eval_objective(X_next, cand_exp, exp),
            dtype=dtype,
            device=device
        ).unsqueeze(-1).unsqueeze(-1)

        # Append data
        X = torch.cat((X, X_next), dim=0)
        Y = torch.cat((Y, Y_next), dim=0)

        print(f"{len(X)-len(X_init)}) Best value: {(1/Y).min().item():.2e}")
    return X, Y