# This script contains functions for initialising alpha and beta

from math import log
import numpy as np
from slise.utils import random_sample_int
from slise.data import pca_simple, pca_invert_model
from slise.optimisation import next_beta, loss_residuals, regularised_regression
from lbfgs import fmin_lbfgs


def fast_lstsq(x: np.ndarray, y: np.ndarray, max_iterations: int = 300):
    if x.shape[1] > max_iterations * 20:
        return regularised_regression(x, y, 0, 0, max_iterations)
    else:
        return np.linalg.lstsq(x, y, rcond=None)[0]


def initialise_lasso(
    X: np.ndarray,
    Y: np.ndarray,
    epsilon: float = 0,
    max_iterations: int = 300,
    **kwargs
):
    """
        Initialise alpha and beta to be equivalent to LASSO
    """
    return fast_lstsq(X, Y, max_iterations), 0


def initialise_ols(
    X: np.ndarray,
    Y: np.ndarray,
    epsilon: float,
    beta_max: float = 20,
    max_approx: float = 1.15,
    max_iterations: int = 300,
    beta_max_init: float = 2.5,
    min_beta_step: float = 1e-8,
    **kwargs
):
    """
        Initialise alpha to OLS and beta to "next beta"
    """
    alpha = fast_lstsq(X, Y, max_iterations)
    epsilon = epsilon ** 2
    beta_max = min(beta_max, beta_max_init) / epsilon
    residuals = (Y - X @ alpha) ** 2
    beta = next_beta(residuals, epsilon, 0, beta_max, log(max_approx), min_beta_step)
    return alpha, beta


def initialise_zeros(
    X: np.ndarray,
    Y: np.ndarray,
    epsilon: float,
    beta_max: float = 20,
    max_approx: float = 1.15,
    beta_max_init: float = 2.5,
    min_beta_step: float = 1e-8,
    **kwargs
):
    """
        Initialise alpha to 0 and beta to "next beta"
    """
    epsilon = epsilon ** 2
    beta_max = min(beta_max, beta_max_init) / epsilon
    beta = next_beta(Y ** 2, epsilon, 0, beta_max, log(max_approx), min_beta_step)
    return np.zeros(X.shape[1]), beta


def __create_candidate(
    X: np.ndarray, Y: np.ndarray, pca_treshold: int = 10, max_iterations: int = 300
):
    if X.shape[1] <= pca_treshold:
        sel = random_sample_int(*X.shape)
        return fast_lstsq(X[sel, :], Y[sel], max_iterations)
    else:
        sel = random_sample_int(X.shape[0], pca_treshold)
        pca, v = pca_simple(X[sel, :], pca_treshold)
        mod = fast_lstsq(pca, Y[sel], max_iterations)
        return pca_invert_model(mod, v)


def initialise_candidates(
    X: np.ndarray,
    Y: np.ndarray,
    epsilon: float,
    beta_max: float = 20,
    max_approx: float = 1.15,
    pca_treshold: int = 10,
    num_init: int = 500,
    max_iterations: int = 300,
    beta_max_init: float = 2.5,
    min_beta_step: float = 1e-8,
    **kwargs
) -> np.ndarray:
    """
        Generate a number (num_init) of candidates and select the best one to be
        alpha, and beta to be the corresponding "next_beta"
    """
    # Prepare parameters
    epsilon = epsilon ** 2
    beta_max = min(beta_max, beta_max_init) / epsilon
    max_approx = log(max_approx)
    # Initial model (zeros)
    alpha = np.zeros(X.shape[1])
    residuals = Y ** 2
    beta = next_beta(residuals, epsilon, 0, beta_max, max_approx, min_beta_step)
    loss = loss_residuals(alpha, residuals, epsilon, 0, 0, beta)
    # Find the candidate with the best loss for the next_beta
    for i in range(num_init):
        try:
            model = __create_candidate(X, Y, pca_treshold, max_iterations)
            residuals2 = (Y - X @ model) ** 2
            loss2 = loss_residuals(model, residuals2, epsilon, 0, 0, beta)
            if loss2 < loss:
                alpha = model
                beta = next_beta(
                    residuals2, epsilon, 0, beta_max, max_approx, min_beta_step
                )
                loss = loss_residuals(model, residuals2, epsilon, 0, 0, beta)
        except np.linalg.LinAlgError:
            pass
    return alpha, beta


def __create_candidate2(X: np.ndarray, Y: np.ndarray, max_iterations: int = 300):
    sel = random_sample_int(X.shape[0], 3)
    X = X[sel, :]
    Y = Y[sel]
    return regularised_regression(X, Y, 1e-8, 0, max_iterations)


def initialise_candidates2(
    X: np.ndarray,
    Y: np.ndarray,
    epsilon: float,
    beta_max: float = 20,
    max_approx: float = 1.15,
    num_init: int = 500,
    max_iterations: int = 300,
    beta_max_init: float = 2.5,
    min_beta_step: float = 1e-8,
    **kwargs
) -> np.ndarray:
    """
        Generate a number (num_init) of candidates and select the best one to be
        alpha, and beta to be the corresponding "next_beta"
    """
    # Prepare parameters
    epsilon = epsilon ** 2
    beta_max = min(beta_max, beta_max_init) / epsilon
    max_approx = log(max_approx)
    # Initial model (zeros)
    alpha = np.zeros(X.shape[1])
    residuals = Y ** 2
    beta = next_beta(residuals, epsilon, 0, beta_max, max_approx, min_beta_step)
    loss = loss_residuals(alpha, residuals, epsilon, 0, 0, beta)
    # Find the candidate with the best loss for the next_beta
    for i in range(num_init):
        try:
            model = __create_candidate2(X, Y, max_iterations)
            residuals2 = (Y - X @ model) ** 2
            loss2 = loss_residuals(model, residuals2, epsilon, 0, 0, beta)
            if loss2 < loss:
                alpha = model
                beta = next_beta(
                    residuals2, epsilon, 0, beta_max, max_approx, min_beta_step
                )
                loss = loss_residuals(model, residuals2, epsilon, 0, 0, beta)
        except np.linalg.LinAlgError:
            pass
    return alpha, beta
