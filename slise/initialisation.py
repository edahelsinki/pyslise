# This script contains functions for initialising alpha and beta

from math import log
import numpy as np
from slise.utils import random_sample_int
from slise.data import add_intercept_column, remove_intercept_column,\
    pca_simple, pca_invert_model, pca_rotate, local_into, local_from
from slise.optimisation import next_beta, loss_residuals

def initialise_lasso(X: np.ndarray, Y: np.ndarray, **kwargs):
    """
        Initialise alpha and beta to be equivalent to LASSO
    """
    return np.linalg.lstsq(X, Y, rcond=None)[0], 0

def initialise_ols(X: np.ndarray, Y: np.ndarray, epsilon: float = 0.1, beta_max: float = 5, max_approx: float = 1.15, **kwargs):
    """
        Initialise alpha to OLS and beta to "next beta"
    """
    epsilon = epsilon**2
    beta_max = beta_max / epsilon
    max_approx = log(max_approx)
    alpha = np.linalg.lstsq(X, Y, rcond=None)[0]
    residuals = (Y - X @ alpha)**2
    beta = next_beta(residuals, epsilon, 0, beta_max, max_approx, 1e-8)
    return alpha, beta

def initialise_zeros(X: np.ndarray, Y: np.ndarray, epsilon: float = 0.1, beta_max: float = 5, max_approx: float = 1.15, **kwargs):
    """
        Initialise alpha to 0 and beta to "next beta"
    """
    epsilon = epsilon**2
    beta_max = beta_max / epsilon
    max_approx = log(max_approx)
    alpha = np.zeros(X.shape[1])
    residuals = Y**2
    beta = next_beta(residuals, epsilon, 0, beta_max, max_approx, 1e-8)
    return alpha, beta

def initialise_candidates(X: np.ndarray, Y: np.ndarray, x: np.ndarray = None, epsilon: float = 0.1, intercept: bool = False,
        beta_max: float = 5, max_approx: float = 1.15, pca_treshold: int = 10, inits: int = 500, **kwargs) -> np.ndarray:
    """
        Generate a number (inits) of candidates and select the best one to be alpha,
        and beta to be the corresponding "next beta"
    """
    epsilon = epsilon**2
    beta_max = beta_max / epsilon
    max_approx = log(max_approx)
    alpha = np.zeros(X.shape[1])
    residuals = Y**2
    beta = next_beta(residuals, epsilon, 0, beta_max, max_approx, 1e-8)
    loss = loss_residuals(alpha, residuals, epsilon, 0, beta)
    # Fast functions for generating candidates
    if X.shape[1] <= pca_treshold:
        def init():
            sel = random_sample_int(*X.shape)
            return np.linalg.lstsq(X[sel,:], Y[sel], rcond=None)[0]
    elif x is not None:
        def init():
            sel = random_sample_int(X.shape[0], pca_treshold)
            pca, v = pca_simple(local_from(X, x), pca_treshold)
            xp = pca_rotate(x, v)
            pca = local_into(pca, xp)
            mod = np.linalg.lstsq(pca, Y[sel], rcond=None)[0]
            return pca_invert_model(mod, v)
    elif intercept:
        def init():
            sel = random_sample_int(X.shape[0], pca_treshold)
            pca, v = pca_simple(remove_intercept_column(X), pca_treshold-1)
            mod = np.linalg.lstsq(add_intercept_column(pca), Y[sel], rcond=None)[0]
            return pca_invert_model(mod, v)
    else:
        def init():
            sel = random_sample_int(X.shape[0], pca_treshold)
            pca, v = pca_simple(X[sel,:], pca_treshold)
            mod = np.linalg.lstsq(pca, Y[sel], rcond=None)[0]
            return pca_invert_model(mod, v)
    # Select the best candidate
    for i in range(inits):
        try:
            model = init()
            residuals = (Y - X @ model)**2
            loss2 = loss_residuals(model, residuals, epsilon, 0, beta)
            if loss2 < loss:
                beta = next_beta(residuals, epsilon, 0, beta_max, max_approx, 1e-8)
                loss = loss_residuals(model, residuals, epsilon, 0, beta)
                alpha = model
        except np.linalg.LinAlgError:
            pass
    return alpha, beta
