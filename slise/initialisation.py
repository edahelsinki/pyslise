# This script contains functions for initialising alpha and beta

from math import log
from typing import Optional, Tuple, Union
from warnings import catch_warnings

import numpy as np

from slise.data import pca_invert_model, pca_simple
from slise.optimisation import loss_residuals, next_beta, regularised_regression


def fast_lstsq(
    x: np.ndarray,
    y: np.ndarray,
    weight: Optional[np.ndarray] = None,
    max_iterations: int = 300,
) -> np.ndarray:
    """A fast version of least squares that falls back to optimisation if the input size gest too large.

    Args:
        x (np.ndarray): Data matrix.
        y (np.ndarray): Response vector.
        weight (Optional[np.ndarray], optional): Weight vector for the data items. Defaults to None.
        max_iterations (int, optional): The number of iterations to use in case of optimisation. Defaults to 300.

    Returns:
        np.ndarray: vector of coefficients
    """
    if weight is None or x.shape[1] <= max_iterations * 20:
        return np.linalg.lstsq(x, y, rcond=None)[0]
    else:
        return regularised_regression(x, y, 0, 0, weight, max_iterations)


def initialise_lasso(
    X: np.ndarray,
    Y: np.ndarray,
    epsilon: float = 0,
    weight: Optional[np.ndarray] = None,
    max_iterations: int = 300,
    **kwargs
) -> Tuple[np.ndarray, float]:
    """Initialise alpha and beta to be equivalent to LASSO.

    Args:
        X (np.ndarray): Data matrix.
        Y (np.ndarray): Response vector.
        epsilon (float, optional): The error tolerance. Defaults to 0.
        weight (Optional[np.ndarray], optional): Weight vector for the data items. Defaults to None.
        max_iterations (int, optional): The number of iterations to use in case of optimisation. Defaults to 300.

    Returns:
        Tuple[np.ndarray, float]: `(alpha, beta)`.
    """
    return fast_lstsq(X, Y, weight, max_iterations), 0.0


def initialise_ols(
    X: np.ndarray,
    Y: np.ndarray,
    epsilon: float,
    weight: Optional[np.ndarray] = None,
    beta_max: float = 20,
    max_approx: float = 1.15,
    max_iterations: int = 300,
    beta_max_init: float = 2.5,
    min_beta_step: float = 1e-8,
    **kwargs
) -> Tuple[np.ndarray, float]:
    """Initialise alpha to OLS and beta to `next_beta`.

    Args:
        X (np.ndarray): Data matrix.
        Y (np.ndarray): Response vector.
        epsilon (float, optional): The error tolerance. Defaults to 0.
        weight (Optional[np.ndarray], optional): Weight vector for the data items. Defaults to None.
        beta_max (float, optional): The stopping sigmoid steepness. Defaults to 20.
        max_approx (float, optional): Approximation ratio when selecting the next beta. Defaults to 1.15.
        max_iterations (int, optional): The number of iterations to use in case of optimisation. Defaults to 300.
        beta_max_init (float, optional): Maximum beta. Defaults to 2.5.
        min_beta_step (float, optional): Minimum beta. Defaults to 1e-8.

    Returns:
        Tuple[np.ndarray, float]: `(alpha, beta)`.
    """
    alpha = fast_lstsq(X, Y, weight, max_iterations)
    epsilon = epsilon ** 2
    beta_max = min(beta_max, beta_max_init) / epsilon
    residuals = (Y - X @ alpha) ** 2
    beta = next_beta(
        residuals, epsilon, 0, weight, beta_max, log(max_approx), min_beta_step
    )
    return alpha, beta


def initialise_zeros(
    X: np.ndarray,
    Y: np.ndarray,
    epsilon: float,
    weight: Optional[np.ndarray] = None,
    beta_max: float = 20,
    max_approx: float = 1.15,
    beta_max_init: float = 2.5,
    min_beta_step: float = 1e-8,
    **kwargs
) -> Tuple[np.ndarray, float]:
    """Initialise alpha to 0 and beta to `next_beta`.

    Args:
        X (np.ndarray): Data matrix.
        Y (np.ndarray): Response vector.
        epsilon (float, optional): The error tolerance. Defaults to 0.
        weight (Optional[np.ndarray], optional): Weight vector for the data items. Defaults to None.
        beta_max (float, optional): The stopping sigmoid steepness. Defaults to 20.
        max_approx (float, optional): Approximation ratio when selecting the next beta. Defaults to 1.15.
        beta_max_init (float, optional): Maximum beta. Defaults to 2.5.
        min_beta_step (float, optional): Minimum beta. Defaults to 1e-8.

    Returns:
        Tuple[np.ndarray, float]: `(alpha, beta)`.
    """
    epsilon = epsilon ** 2
    beta_max = min(beta_max, beta_max_init) / epsilon
    beta = next_beta(
        Y ** 2, epsilon, 0, weight, beta_max, log(max_approx), min_beta_step
    )
    return np.zeros(X.shape[1]), beta


def initialise_fixed(
    init: Union[np.ndarray, Tuple[np.ndarray, float]],
    X: np.ndarray,
    Y: np.ndarray,
    epsilon: float,
    weight: Optional[np.ndarray] = None,
    beta_max: float = 20,
    max_approx: float = 1.15,
    beta_max_init: float = 2.5,
    min_beta_step: float = 1e-8,
):
    """Initialise alpha and beta to the given values (or `next_beta` if beta is not given).

    Args:
        init (Union[np.ndarray, Tuple[np.ndarray, float]]): The fixed alpha, and optional beta.
        X (np.ndarray): Data matrix.
        Y (np.ndarray): Response vector.
        epsilon (float, optional): The error tolerance. Defaults to 0.
        weight (Optional[np.ndarray], optional): Weight vector for the data items. Defaults to None.
        beta_max (float, optional): The stopping sigmoid steepness. Defaults to 20.
        max_approx (float, optional): Approximation ratio when selecting the next beta. Defaults to 1.15.
        beta_max_init (float, optional): Maximum beta. Defaults to 2.5.
        min_beta_step (float, optional): Minimum beta. Defaults to 1e-8.

    Returns:
        [type]: [description]
    """
    if isinstance(init, tuple):
        alpha, beta = init
    else:
        epsilon = epsilon ** 2
        beta_max = min(beta_max, beta_max_init) / epsilon
        alpha = init
        beta = next_beta(
            (X @ alpha - Y) ** 2, epsilon, 0, weight, beta_max, log(max_approx),
        )
    return alpha, beta


def __create_candidate(
    X: np.ndarray,
    Y: np.ndarray,
    weight: Optional[np.ndarray] = None,
    pca_treshold: int = 10,
    max_iterations: int = 300,
) -> np.ndarray:
    if X.shape[1] <= pca_treshold:
        sel = np.random.choice(X.shape[0], X.shape[1], False, weight)
        return fast_lstsq(X[sel, :], Y[sel], None, max_iterations)
    else:
        sel = np.random.choice(X.shape[0], pca_treshold, False, weight)
        pca, v = pca_simple(X[sel, :], pca_treshold)
        mod = fast_lstsq(pca, Y[sel], None, max_iterations)
        return pca_invert_model(mod, v)


def initialise_candidates(
    X: np.ndarray,
    Y: np.ndarray,
    epsilon: float,
    weight: Optional[np.ndarray] = None,
    beta_max: float = 20,
    max_approx: float = 1.15,
    pca_treshold: int = 10,
    num_init: int = 500,
    max_iterations: int = 300,
    beta_max_init: float = 2.5,
    min_beta_step: float = 1e-8,
    **kwargs
) -> Tuple[np.ndarray, float]:
    """Generate a number (num_init) of candidates, using PCA to shrink the random subsets.
        Then select the best one to be alpha and beta to be the corresponding `next_beta`

    Args:
        X (np.ndarray): Data matrix.
        Y (np.ndarray): Response vector.
        epsilon (float, optional): The error tolerance. Defaults to 0.
        weight (Optional[np.ndarray], optional): Weight vector for the data items. Defaults to None.
        beta_max (float, optional): The stopping sigmoid steepness. Defaults to 20.
        max_approx (float, optional): Approximation ratio when selecting the next beta. Defaults to 1.15.
        pca_treshold (int, optional): Treshold number of dimension to use PCA. Defaults to 10.
        num_init (int, optional): Number of candidates to generate. Defaults to 500.
        max_iterations (int, optional): The number of iterations to use in case of optimisation. Defaults to 300.
        beta_max_init (float, optional): Maximum beta. Defaults to 2.5.
        min_beta_step (float, optional): Minimum beta. Defaults to 1e-8.

    Returns:
        Tuple[np.ndarray, float]: `(alpha, beta)`.
    """
    # Prepare parameters
    epsilon = epsilon ** 2
    beta_max = min(beta_max, beta_max_init) / epsilon
    max_approx = log(max_approx)
    if weight is not None:
        weight = weight / np.sum(weight)
    # Initial model (zeros)
    alpha = np.zeros(X.shape[1])
    residuals = Y ** 2
    beta = next_beta(residuals, epsilon, 0, weight, beta_max, max_approx, min_beta_step)
    loss = loss_residuals(alpha, residuals, epsilon, beta, 0, 0, weight)
    # Find the candidate with the best loss for the next_beta
    for i in range(num_init):
        try:
            model = __create_candidate(X, Y, weight, pca_treshold, max_iterations)
            residuals2 = (Y - X @ model) ** 2
            loss2 = loss_residuals(model, residuals2, epsilon, beta, 0, 0, weight)
            if loss2 < loss:
                alpha = model
                beta = next_beta(
                    residuals2, epsilon, 0, weight, beta_max, max_approx, min_beta_step
                )
                loss = loss_residuals(model, residuals2, epsilon, beta, 0, 0, weight)
        except np.linalg.LinAlgError:
            pass
    return alpha, beta


def __create_candidate2(
    X: np.ndarray,
    Y: np.ndarray,
    weight: Optional[np.ndarray] = None,
    max_iterations: int = 300,
) -> np.ndarray:
    sel = np.random.choice(X.shape[0], 3, False, weight)
    X = X[sel, :]
    Y = Y[sel]
    with catch_warnings(record=False):
        reg = regularised_regression(X, Y, 1e-8, 0, max_iterations)
    return reg


def initialise_candidates2(
    X: np.ndarray,
    Y: np.ndarray,
    epsilon: float,
    weight: Optional[np.ndarray] = None,
    beta_max: float = 20,
    max_approx: float = 1.15,
    num_init: int = 500,
    max_iterations: int = 300,
    beta_max_init: float = 2.5,
    min_beta_step: float = 1e-8,
    **kwargs
) -> Tuple[np.ndarray, float]:
    """Generate a number (num_init) of candidates, using LASSO to shrink the random subsets.
        Then select the best one to be alpha and beta to be the corresponding `next_beta`

    Args:
        X (np.ndarray): Data matrix.
        Y (np.ndarray): Response vector.
        epsilon (float, optional): The error tolerance. Defaults to 0.
        weight (Optional[np.ndarray], optional): Weight vector for the data items. Defaults to None.
        beta_max (float, optional): The stopping sigmoid steepness. Defaults to 20.
        max_approx (float, optional): Approximation ratio when selecting the next beta. Defaults to 1.15.
        num_init (int, optional): Number of candidates to generate. Defaults to 500.
        max_iterations (int, optional): The number of iterations to use in case of optimisation. Defaults to 300.
        beta_max_init (float, optional): Maximum beta. Defaults to 2.5.
        min_beta_step (float, optional): Minimum beta. Defaults to 1e-8.

    Returns:
        Tuple[np.ndarray, float]: `(alpha, beta)`.
    """
    # Prepare parameters
    epsilon = epsilon ** 2
    beta_max = min(beta_max, beta_max_init) / epsilon
    max_approx = log(max_approx)
    if weight is not None:
        weight = weight / np.sum(weight)
    # Initial model (zeros)
    alpha = np.zeros(X.shape[1])
    residuals = Y ** 2
    beta = next_beta(residuals, epsilon, 0, weight, beta_max, max_approx, min_beta_step)
    loss = loss_residuals(alpha, residuals, epsilon, beta, 0, 0, weight)
    # Find the candidate with the best loss for the next_beta
    for i in range(num_init):
        try:
            model = __create_candidate2(X, Y, weight, max_iterations)
            residuals2 = (Y - X @ model) ** 2
            loss2 = loss_residuals(model, residuals2, epsilon, beta, 0, 0, weight)
            if loss2 < loss:
                alpha = model
                beta = next_beta(
                    residuals2, epsilon, 0, weight, beta_max, max_approx, min_beta_step
                )
                loss = loss_residuals(model, residuals2, epsilon, beta, 0, 0, weight)
        except np.linalg.LinAlgError:
            pass
    return alpha, beta
