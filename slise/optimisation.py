"""
    This script contains the loss functions and optimisation functions for SLISE.
"""

from math import log
from typing import Callable, Optional, Tuple
from warnings import catch_warnings, warn

import numpy as np
from lbfgs import LBFGSError, fmin_lbfgs
from numba import jit, get_num_threads, set_num_threads, threading_layer, float64
from scipy.optimize import brentq

from slise.utils import (
    SliseWarning,
    dlog_sigmoid,
    log_sigmoid,
    log_sum_special,
    mat_mul_inter,
    sigmoid,
)


def loss_smooth(
    alpha: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    epsilon: float,
    beta: float = 100,
    lambda1: float = 0,
    lambda2: float = 0,
    weight: Optional[np.ndarray] = None,
) -> float:
    """Smoothed version of the SLISE loss ([slise.optimisation.loss_sharp][]).

    Args:
        alpha (np.ndarray): Linear model coefficients.
        X (np.ndarray): Data matrix.
        Y (np.ndarray): Response vector.
        epsilon (float): Error tolerance.
        beta (float, optional): Sigmoid steepness. Defaults to 100.
        lambda1 (float, optional): LASSO/L1 regularisation coefficient. Defaults to 0.
        lambda2 (float, optional): Ridge/L2 regularisation coefficient. Defaults to 0.
        weight (Optional[np.ndarray], optional): Weight vector for the data items. Defaults to None.

    Returns:
        float: Loss value.
    """
    epsilon *= epsilon
    residual2 = ((X @ alpha) - Y) ** 2
    subset = sigmoid(beta * (epsilon - residual2))
    loss = 0.0
    if weight is None:
        residual2 = np.minimum(0, residual2 - epsilon * len(Y))
        loss += np.sum(subset * residual2) / len(Y)
    else:
        sumw = np.sum(weight)
        residual2 = np.minimum(0, residual2 - epsilon * sumw)
        loss += np.sum(subset * residual2 * weight) / sumw
    if lambda1 > 0:
        loss += lambda1 * np.sum(np.abs(alpha))
    if lambda2 > 0:
        loss += lambda2 * np.sum(alpha * alpha)
    return loss


def loss_residuals(
    alpha: np.ndarray,
    residuals2: np.ndarray,
    epsilon2: float,
    beta: float = 100.0,
    lambda1: float = 0.0,
    lambda2: float = 0.0,
    weight: Optional[np.ndarray] = None,
) -> float:
    """Smoothed version of the SLISE loss ([slise.optimisation.loss_smooth][]), that takes already calculated residuals.

    Args:
        alpha (np.ndarray): Linear model coefficients.
        residuals2 (np.ndarray): Squared residuals.
        epsilon2 (float): Squared error tolerance.
        beta (float, optional): Sigmoid steepness. Defaults to 100.
        lambda1 (float, optional): LASSO/L1 regularisation coefficient. Defaults to 0.
        lambda2 (float, optional): Ridge/L2 regularisation coefficient. Defaults to 0.
        weight (Optional[np.ndarray], optional): Weight vector for the data items. Defaults to None.

    Returns:
        float: Loss value.
    """
    alpha = np.ascontiguousarray(alpha, dtype=np.float64)
    residuals2 = np.ascontiguousarray(residuals2, dtype=np.float64)
    lambda1 = float(lambda1)
    lambda2 = float(lambda2)
    epsilon2 = float(epsilon2)
    beta = float(beta)
    if weight is None:
        return _loss_residuals(alpha, residuals2, epsilon2, beta, lambda1, lambda2)
    else:
        weight = np.ascontiguousarray(weight, dtype=np.float64)
        return _loss_residualsw(
            alpha, residuals2, epsilon2, beta, lambda1, lambda2, weight
        )


@jit(
    (float64[::1], float64[::1], float64, float64, float64, float64),
    nopython=True,
    fastmath=True,
    parallel=True,
    cache=True,
    nogil=True,
    boundscheck=False,
)
def _loss_residuals(
    alpha: np.ndarray,
    residuals2: np.ndarray,
    epsilon2: float,
    beta: float = 100,
    lambda1: float = 0,
    lambda2: float = 0,
) -> float:
    """
    Smoothed version of the SLISE loss, that takes already calculated squared residuals.
    _This function is sped up with numba._
    """
    subset = 1 / (1 + np.exp(-beta * (epsilon2 - residuals2)))  # Sigmoid
    residual2 = np.minimum(0, residuals2 - epsilon2 * len(residuals2))
    loss = np.sum(subset * residual2) / len(residuals2)
    if lambda1 > 0:
        loss = loss + lambda1 * np.sum(np.abs(alpha))
    if lambda2 > 0:
        loss = loss + lambda2 * np.sum(alpha * alpha)
    return loss


@jit(
    (
        float64[::1],
        float64[::1],
        float64,
        float64,
        float64,
        float64,
        float64[::1],
    ),
    nopython=True,
    fastmath=True,
    parallel=True,
    cache=True,
    nogil=True,
    boundscheck=False,
)
def _loss_residualsw(
    alpha: np.ndarray,
    residuals2: np.ndarray,
    epsilon2: float,
    beta: float,
    lambda1: float,
    lambda2: float,
    weight: np.ndarray,
) -> float:
    """
    Smoothed version of the SLISE loss, that takes already calculated squared residuals.
    _This function is sped up with numba._
    """
    subset = 1 / (1 + np.exp(-beta * (epsilon2 - residuals2)))  # Sigmoid
    sumw = np.sum(weight)
    residual2 = np.minimum(0, residuals2 - epsilon2 * sumw)
    loss = np.sum(subset * residual2 * weight) / sumw
    if lambda1 > 0:
        loss = loss + lambda1 * np.sum(np.abs(alpha))
    if lambda2 > 0:
        loss = loss + lambda2 * np.sum(alpha * alpha)
    return loss


def loss_sharp(
    alpha: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    epsilon: float,
    lambda1: float = 0,
    lambda2: float = 0,
    weight: Optional[np.ndarray] = None,
) -> float:
    """Exact version (no sigmoid smoothing) of the SLISE loss.

    Args:
        alpha (np.ndarray): Linear model coefficients.
        X (np.ndarray): Data matrix.
        Y (np.ndarray): Response vector.
        epsilon (float): Error tolerance.
        lambda1 (float, optional): LASSO/L1 regularisation coefficient. Defaults to 0.
        lambda2 (float, optional): Ridge/L2 regularisation coefficient. Defaults to 0.
        weight (Optional[np.ndarray], optional): Weight vector for the data items. Defaults to None.

    Returns:
        float: Loss value.
    """
    epsilon *= epsilon
    residual2 = (Y - mat_mul_inter(X, alpha)) ** 2
    if weight is None:
        loss = np.sum(residual2[residual2 <= epsilon] - (epsilon * len(Y))) / len(Y)
    else:
        sumw = np.sum(weight)
        mask = residual2 <= epsilon
        loss = np.sum((residual2[mask] - (epsilon * sumw)) * weight[mask]) / sumw
    if lambda1 > 0:
        loss += lambda1 * np.sum(np.abs(alpha))
    if lambda2 > 0:
        loss += lambda2 * np.sum(alpha * alpha)
    return loss


def loss_grad(
    alpha: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    epsilon: float,
    beta: float,
    lambda1: float = 0.0,
    lambda2: float = 0.0,
    weight: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray]:
    """Smoothed version of the SLISE loss ([slise.optimisation.loss_smooth][]), that also calculates the gradient.

    Args:
        alpha (np.ndarray): Linear model coefficients.
        X (np.ndarray): Data matrix.
        Y (np.ndarray): Response vector.
        epsilon (float): Error tolerance.
        beta (float): Sigmoid steepness.
        lambda1 (float): Lasso/L1 regularisation coefficient. Defaults to 0.0.
        lambda2 (float): Ridge/L2 regularisation coefficient. Defaults to 0.0.
        weight (Optional[np.ndarray]): Weight vector for the data items. Defaults to None.

    Returns:
        Tuple[float, np.ndarray]: Loss value and gradient vector.
    """
    alpha = np.ascontiguousarray(alpha, dtype=np.float64)
    X = np.ascontiguousarray(X, dtype=np.float64)
    Y = np.ascontiguousarray(Y, dtype=np.float64)
    assert X.shape[0] == len(Y), f"Different lengths {X.shape[0]} != {len(Y)}"
    assert X.shape[1] == len(alpha), f"Different lengths {X.shape[0]} != {len(alpha)}"
    lambda1 = float(lambda1)
    lambda2 = float(lambda2)
    epsilon = float(epsilon)
    beta = float(beta)
    if weight is None:
        loss, grad = _loss_grad(alpha, X, Y, epsilon, beta, lambda2)
    else:
        weight = np.ascontiguousarray(weight, dtype=np.float64)
        assert Y.shape == weight.shape, f"Different shapes {Y.shape} != {weight.shape}"
        loss, grad = _loss_gradw(alpha, X, Y, epsilon, beta, lambda2, weight)
    if lambda1 > 0:
        loss = loss + lambda1 * np.sum(np.abs(alpha))
        grad = grad + lambda1 * np.sign(alpha)
    return loss, grad


def loss_numba(*args, **kwargs):
    warn("Use `loss_grad` instead of `loss_numba`", DeprecationWarning)
    return loss_grad(*args, **kwargs)


@jit(
    (float64[::1], float64[:, ::1], float64[::1], float64, float64, float64),
    nopython=True,
    fastmath=True,
    parallel=True,
    cache=True,
    nogil=True,
    boundscheck=False,
)
def _loss_grad(
    alpha: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    epsilon: float,
    beta: float,
    lambda2: float,
) -> Tuple[float, np.ndarray]:
    """Smoothed version of the SLISE loss that also calculates the gradient.
    _This function is sped up with numba._
    """
    epsilon *= epsilon
    residuals = (X @ alpha) - Y
    residual2 = residuals**2
    n = residuals.dtype.type(len(Y))
    # Loss
    subset = 1 / (1 + np.exp(-beta * (epsilon - residual2)))  # Sigmoid
    residual2 = np.minimum(0, residual2 - (epsilon * n))
    loss = np.sum(subset * residual2) / n
    # Gradient
    grad = (
        np.expand_dims(
            subset
            * residuals
            * (2.0 / n - residual2 * (2.0 * beta / n) * (1.0 - subset))
            * (residual2 < 0.0).astype(X.dtype),
            0,
        )
        @ X
    )
    # Lambda
    if lambda2 > 0:
        loss = loss + lambda2 * np.sum(alpha * alpha)
        grad = grad + (lambda2 * 2) * alpha
    return loss, grad


@jit(
    (
        float64[::1],
        float64[:, ::1],
        float64[::1],
        float64,
        float64,
        float64,
        float64[::1],
    ),
    nopython=True,
    fastmath=True,
    parallel=True,
    cache=True,
    nogil=True,
    boundscheck=False,
)
def _loss_gradw(
    alpha: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    epsilon: float,
    beta: float,
    lambda2: float,
    weight: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """Smoothed version of the SLISE loss, that also calculates the gradient.
    _This function is sped up with numba._
    """
    epsilon *= epsilon
    residuals = (X @ alpha) - Y
    residual2 = residuals**2
    n = np.sum(weight)
    # Loss
    subset = 1 / (1 + np.exp(-beta * (epsilon - residual2)))  # Sigmoid
    residual2 = np.minimum(0, residual2 - (epsilon * n))
    loss = np.sum(subset * residual2 * weight) / n
    # Gradient
    grad = (
        np.expand_dims(
            subset
            * residuals
            * weight
            * (2.0 / n - residual2 * (2.0 * beta / n) * (1.0 - subset))
            * (residual2 < 0.0).astype(X.dtype),
            0,
        )
        @ X
    )
    # Lambda
    if lambda2 > 0:
        loss = loss + lambda2 * np.sum(alpha * alpha)
        grad = grad + (lambda2 * 2) * alpha
    return loss, grad


@jit(
    (float64[::1], float64[:, ::1], float64[::1], float64),
    nopython=True,
    fastmath=True,
    parallel=True,
    cache=True,
    nogil=True,
    boundscheck=False,
)
def _ridge_numba(
    alpha: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    lambda2: float = 0.0,
) -> Tuple[float, np.ndarray]:
    """
    Ridge regression (OLS + L2) loss, that also calculates the gradient.
    _This function is sped up with numba._
    """
    distances = (X @ alpha) - Y
    loss = np.sum(distances**2) / 2
    grad = np.expand_dims(distances, 0) @ X
    if lambda2 > 0.0:
        loss = loss + lambda2 * np.sum(alpha**2) / 2
        grad = grad + lambda2 * alpha
    return loss, grad


@jit(
    (float64[::1], float64[:, ::1], float64[::1], float64, float64[::1]),
    nopython=True,
    fastmath=True,
    parallel=True,
    cache=True,
    nogil=True,
    boundscheck=False,
)
def _ridge_numbaw(
    alpha: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    lambda2: float,
    weight: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """
    Ridge regression (OLS + L2) loss, that also calculates the gradient.
    _This function is sped up with numba._
    """
    distances = ((X @ alpha) - Y) * weight
    loss = np.sum(distances**2) / 2
    grad = np.expand_dims(distances, 0) @ X
    if lambda2 > 0.0:
        loss = loss + lambda2 * np.sum(alpha**2) / 2
        grad = grad + lambda2 * alpha
    return loss, grad


def owlqn(
    loss_grad_fn: Callable[[np.ndarray], Tuple[float, np.ndarray]],
    x0: np.ndarray,
    lambda1: float = 0,
    max_iterations: int = 200,
    **kwargs,
) -> np.ndarray:
    """Wrapper around owlqn that converts max_iter errors to warnings (see `lbfgs.fmin_lbfgs` from PyLBFGS).

    Args:
        loss_grad_fn (Callable[[np.ndarray], Tuple[float, np.ndarray]]): Function that calculates the loss and gradient.
        x0 (np.ndarray): Initial vector to be optimised.
        lambda1 (float, optional): LASSO/L1 regularisation coefficient. Defaults to 1e-6.
        max_iterations (int, optional): Maximum number of optimisation steps. Defaults to 200.

    Returns:
        np.ndarray: Optimised vector.
    """

    def f(x: np.ndarray, gradient: np.ndarray) -> float:
        loss, grad = loss_grad_fn(x)
        gradient[:] = grad
        return loss

    try:  # PyLBFGS throws an error if max_iterations is exceeded, this is a workaround to convert it into a warning

        def p(x, g, fx, xnorm, gnorm, step, k, num_eval, *args):
            if k >= max_iterations:
                x0[:] = x

        x0 = fmin_lbfgs(
            f=f,
            x0=x0,
            progress=p,
            orthantwise_c=lambda1,
            max_iterations=max_iterations,
            line_search="wolfe" if lambda1 > 0 else "default",
            **kwargs,
        )
    except LBFGSError as error:
        if (
            error.args[0]
            != "The algorithm routine reaches the maximum number of iterations."
        ):
            raise error
        else:
            warn(
                "LBFGS optimisation reaches the maximum number of iterations.",
                SliseWarning,
            )
    return x0


def regularised_regression(
    X: np.ndarray,
    Y: np.ndarray,
    lambda1: float = 1e-6,
    lambda2: float = 1e-6,
    weight: Optional[np.ndarray] = None,
    max_iterations: int = 200,
) -> np.ndarray:
    """Train a linear regression model with lasso (L1) and/or ridge (L2) regularisation.

    Args:
        X (np.ndarray): Data matrix.
        Y (np.ndarray): Response vector.
        lambda1 (float, optional): LASSO/L1 regularisation coefficient. Defaults to 1e-6.
        lambda2 (float, optional): Ridge/L2 regularisation coefficient. Defaults to 1e-6.
        weight (Optional[np.ndarray], optional): Weight vector for the data items. Defaults to None.
        max_iterations (int, optional): Maximum number of optimisation steps. Defaults to 200.

    Returns:
        np.ndarray: The coefficients of the linear model.
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    Y = np.ascontiguousarray(Y, dtype=np.float64)
    lambda1 = float(lambda1)
    lambda2 = float(lambda2)
    assert X.shape[0] == len(Y), f"Different lengths {X.shape[0]} != {len(Y)}"
    if weight is None:
        lf = lambda alpha: _ridge_numba(alpha, X, Y, lambda2)
    else:
        weight = np.ascontiguousarray(weight, dtype=np.float64)
        assert Y.shape == weight.shape, f"Different shapes {Y.shape} != {weight.shape}"
        lf = lambda alpha: _ridge_numbaw(alpha, X, Y, lambda2, weight)
    return owlqn(lf, np.zeros(X.shape[1], dtype=np.float64), lambda1, max_iterations)


def optimise_loss(
    alpha: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    epsilon: float = 0.1,
    beta: float = 100,
    lambda1: float = 0,
    lambda2: float = 0,
    weight: Optional[np.ndarray] = None,
    max_iterations: int = 200,
) -> np.ndarray:
    """Optimise a smoothed SLISE loss with `owl-qn`.

    Args:
        alpha (np.ndarray): Linear model coefficients.
        X (np.ndarray): Data matrix.
        Y (np.ndarray): Target vector
        epsilon (float, optional): Error tolerance. Defaults to 0.1.
        beta (float, optional): Sigmoid steepness. Defaults to 100.
        lambda1 (float, optional): LASSO/L1 regularisation coefficient. Defaults to 1e-6.
        lambda2 (float, optional): Ridge/L2 regularisation coefficient. Defaults to 1e-6.
        weight (Optional[np.ndarray], optional): Weight vector for the data items. Defaults to None.
        max_iterations (int, optional): Maximum number of optimisation steps. Defaults to 200.

    Returns:
        np.ndarray: The coefficients of the linear model.
    """
    alpha = np.ascontiguousarray(alpha, dtype=np.float64)
    X = np.ascontiguousarray(X, dtype=np.float64)
    Y = np.ascontiguousarray(Y, dtype=np.float64)
    assert X.shape[0] == len(Y), f"Different lengths {X.shape[0]} != {len(Y)}"
    assert X.shape[1] == len(alpha), f"Different lengths {X.shape[0]} != {len(alpha)}"
    lambda1 = float(lambda1)
    lambda2 = float(lambda2)
    epsilon = float(epsilon)
    beta = float(beta)
    if weight is None:
        lf = lambda alpha: _loss_grad(alpha, X, Y, epsilon, beta, lambda2)
    else:
        weight = np.ascontiguousarray(weight, dtype=np.float64)
        assert Y.shape == weight.shape, f"Different shapes {Y.shape} != {weight.shape}"
        lf = lambda alpha: _loss_gradw(alpha, X, Y, epsilon, beta, lambda2, weight)
    return owlqn(lf, alpha, lambda1, max_iterations)


def log_approximation_ratio(
    residuals2: np.ndarray,
    epsilon2: float,
    beta1: float,
    beta2: float,
    weight: Optional[np.ndarray] = None,
) -> float:
    """Calculate log(K), where K is the approximation ratio between two smoothed losses.

    Args:
        residuals2 (np.ndarray): Squared residuals.
        epsilon2 (float): Squared error tolerance.
        beta1 (float): Old sigmoid steepness.
        beta2 (float): New sigmoid steepness.
        weight (Optional[np.ndarray], optional): Weight vector. Defaults to None.

    Returns:
        float: log of the approximation ratio between `beta1` and `beta2` for the current solution.
    """
    if beta1 >= beta2:
        return 0
    log_f = lambda r, beta: log_sigmoid(beta * (epsilon2 - r))
    dlog_g = lambda r: -beta1 * dlog_sigmoid(
        beta1 * (epsilon2 - r)
    ) + beta2 * dlog_sigmoid(beta2 * (epsilon2 - r))
    if dlog_g(0) < 0:
        a = brentq(dlog_g, 0, epsilon2)
        log_k = min(
            log_f(0, beta1) - log_f(0, beta2), log_f(a, beta1) - log_f(a, beta2)
        )
    else:
        log_k = log_f(0, beta1) - log_f(0, beta2)
    if weight is None:
        phi = np.maximum(0, epsilon2 - residuals2 / len(residuals2))
    else:
        phi = np.maximum(0, epsilon2 - residuals2 / np.sum(weight)) * weight
    log_K = (
        log_sum_special(log_f(residuals2, beta1), phi)
        - log_k
        - log_sum_special(log_f(residuals2, beta2), phi)
    )
    return log_K


def next_beta(
    residuals2: np.ndarray,
    epsilon2: float = 0.01,
    beta: float = 0.0,
    weight: Optional[np.ndarray] = None,
    beta_max: float = 2500,
    log_max_approx: float = 0.14,
    min_beta_step: float = 0.0005,
) -> float:
    """Calculate the next beta for the graduated optimisation.

    Args:
        residuals2 (np.ndarray): Squared residuals.
        epsilon2 (float): Squared error tolerance. Defaults to 0.01.
        beta (float): Sigmoid steepness. Defaults to 0.
        weight (Optional[np.ndarray], optional): Weight vector. Defaults to None.
        beta_max (float, optional): Maximum `beta`. Defaults to 2500.
        log_max_approx (float, optional): Log-maximum approximation ratio. Defaults to 0.14.
        min_beta_step (float, optional): Minimum increase of `beta`. Defaults to 0.0005.

    Returns:
        float: next `beta`.
    """
    if beta >= beta_max:
        return beta
    log_approx = log_approximation_ratio(residuals2, epsilon2, beta, beta_max, weight)
    if log_approx <= log_max_approx:
        return beta_max
    else:
        f = (
            lambda b: log_approximation_ratio(residuals2, epsilon2, beta, b, weight)
            - log_max_approx
        )
        beta_min = beta + min_beta_step * (beta_max + beta)
        return max(brentq(f, beta, beta_max), beta_min)


def matching_epsilon(
    residuals2: np.ndarray,
    epsilon2: float,
    beta: float,
    weight: Optional[np.ndarray] = None,
) -> float:
    """Approximately calculate the epsilon that minimises the approximation ratio to the exact loss.

    Args:
        residuals2 (np.ndarray): Squared residuals.
        epsilon2 (float): Squared error tolerance.
        beta (float): Sigmoid steepness.
        weight (Optional[np.ndarray], optional): Weight vector. Defaults to None.

    Returns:
        float: (Approximatively) optimal epsilon for the exact loss (for the current solution).
    """
    if weight is None:
        residuals2 = np.sort(residuals2)
        loss = sigmoid(beta * (epsilon2 - residuals2))
        i = np.argmax(np.arange(1, 1 + len(residuals2)) * loss)
        return residuals2[i] ** 0.5
    else:
        order = np.argsort(residuals2)
        residuals2 = residuals2[order]
        loss = sigmoid(beta * (epsilon2 - residuals2))
        i = np.argmax(np.cumsum(weight[order]) * loss)
        return residuals2[i] ** 0.5


def _debug_log(
    alpha: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    epsilon: float = 0.1,
    beta: float = 0,
    lambda1: float = 0,
    lambda2: float = 0,
    weight: Optional[np.ndarray] = None,
):
    """
    Print the log statement for a graduated optimisation step.
    """
    residuals = (X @ alpha - Y) ** 2
    loss = loss_sharp(alpha, X, Y, epsilon, lambda1, lambda2, weight)
    bloss = loss_residuals(
        alpha, residuals, epsilon**2, beta, lambda1, lambda2, weight
    )
    epss = matching_epsilon(residuals, epsilon**2, beta, weight)
    beta = beta * epsilon**2
    print(
        f"beta: {beta:5.3f}    epsilon*: {epss:.3f}    Loss: {loss:6.2f}    B-Loss: {bloss:6.2f}"
    )


@np.errstate(over="ignore")
def graduated_optimisation(
    alpha: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    epsilon: float,
    beta: float = 0,
    lambda1: float = 0,
    lambda2: float = 0,
    weight: Optional[np.ndarray] = None,
    beta_max: float = 20,
    max_approx: float = 1.15,
    max_iterations: int = 200,
    debug: bool = False,
) -> np.ndarray:
    """Optimise `alpha` using graduated optimisation.

    Args:
        alpha (np.ndarray): Initial linear model coefficients.
        X (np.ndarray): Data matrix.
        Y (np.ndarray): Response vector.
        epsilon (float): Error tolerance.
        beta (float, optional): Initial sigmoid steepness. Defaults to 0.
        lambda1 (float, optional): L1 regularisation strength. Defaults to 0.
        lambda2 (float, optional): L2 regularisation strength. Defaults to 0.
        weight (Optional[np.ndarray], optional): Weight vector for the data items. Defaults to None.
        beta_max (float, optional): Maximum sigmoid steepness (the final beta). Defaults to 20.
        max_approx (float, optional): Target approximation ratio when increasing beta. Defaults to 1.15.
        max_iterations (int, optional): Maximum number of iterations for owl-qn. Defaults to 200.
        debug (bool, optional): Print debug logs after each optimisation step. Defaults to False.

    Returns:
        np.ndarray: Optimised `alpha`.
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    Y = np.ascontiguousarray(Y, dtype=np.float64)
    if weight is not None:
        weight = np.ascontiguousarray(weight, dtype=np.float64)
    beta_max = beta_max / epsilon**2
    max_approx = log(max_approx)
    with catch_warnings(record=True) as w:
        while beta < beta_max:
            alpha = optimise_loss(
                alpha, X, Y, epsilon, beta, lambda1, lambda2, weight, max_iterations
            )
            if debug:
                _debug_log(alpha, X, Y, epsilon, beta, lambda1, lambda2, weight)
            beta = next_beta(
                (X @ alpha - Y) ** 2, epsilon**2, beta, weight, beta_max, max_approx
            )
    alpha = optimise_loss(
        alpha, X, Y, epsilon, beta, lambda1, lambda2, weight, max_iterations * 4
    )
    if debug:
        _debug_log(alpha, X, Y, epsilon, beta, lambda1, lambda2, weight)
        if w:
            print("Warnings from intermediate steps:", w)
    return alpha


def set_threads(num: int = -1) -> int:
    """Set the number of numba threads.

    Args:
        num (int, optional): The number of threads (or -1 to keep the old value). Defaults to -1.

    Returns:
        int: The old number of theads (or -1 if unchanged).
    """
    if num > 0:
        old = get_num_threads()
        if old != num:
            set_num_threads(num)
        return old
    return -1


@jit((float64[::1],), nopython=True, fastmath=True, parallel=True, cache=True)
def _dummy_numba(
    x: np.ndarray,
) -> np.ndarray:
    """
    A dummy function to check the numba compilation (see check_threading_layer).
    """
    return x * x


def check_threading_layer():
    """
    Check which numba threading_layer is active, and warn if it is "workqueue".
    """
    _dummy_numba(np.ones(1))
    try:
        if threading_layer() == "workqueue":
            warn(
                'Using `numba.threading_layer()=="workqueue"` can be devastatingly slow!'
                " See https://numba.pydata.org/numba-doc/latest/user/threading-layer.html for alternatives.",
                SliseWarning,
            )
    except ValueError as e:
        warn(f"Numba: {e}", SliseWarning)
