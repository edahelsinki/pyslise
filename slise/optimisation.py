# This script contains the optimisations for SLISE (Graduated Optimisation and OWL-QN)

from math import log
from typing import Callable, Optional, Tuple
from warnings import catch_warnings, warn

import numpy as np
from lbfgs import LBFGSError, fmin_lbfgs
from numba import jit, get_num_threads, set_num_threads, threading_layer
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
    """
        Smoothed version of the loss.
    """
    epsilon *= epsilon
    distances = ((X @ alpha) - Y) ** 2
    subset = sigmoid(beta * (epsilon - distances))
    loss = 0.0
    if weight is None:
        residuals = np.minimum(0, distances - epsilon * len(Y))
        loss += np.sum(subset * residuals) / len(Y)
    else:
        sumw = np.sum(weight)
        residuals = np.minimum(0, distances - epsilon * sumw)
        loss += np.sum(subset * residuals * weight) / sumw
    if lambda1 > 0:
        loss += lambda1 * np.sum(np.abs(alpha))
    if lambda2 > 0:
        loss += lambda2 * np.sum(alpha * alpha)
    return loss


@jit(nopython=True, fastmath=True, parallel=True, cache=True)
def loss_residuals(
    alpha: np.ndarray,
    residuals2: np.ndarray,
    epsilon2: float,
    beta: float = 100,
    lambda1: float = 0,
    lambda2: float = 0,
    weight: Optional[np.ndarray] = None,
) -> float:
    """
        Smoothed version of the loss, that takes already calculated residuals.
        This function is sped up with numba.
        Assumes that the residuals and epsilon are squared.
    """
    subset = 1 / (1 + np.exp(-beta * (epsilon2 - residuals2)))  # Sigmoid
    if weight is None:
        residuals = np.minimum(0, residuals2 - epsilon2 * len(residuals2))
        loss = np.sum(subset * residuals) / len(residuals2)
    else:
        sumw = np.sum(weight)
        residuals = np.minimum(0, residuals2 - epsilon2 * sumw)
        loss = np.sum(subset * residuals * weight) / sumw
    if lambda1 > 0:
        loss += lambda1 * np.sum(np.abs(alpha))
    if lambda2 > 0:
        loss += lambda2 * np.sum(alpha * alpha)
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
    """
        Exact version of the loss.
    """
    epsilon *= epsilon
    distances = (Y - mat_mul_inter(X, alpha)) ** 2
    if weight is None:
        loss = np.sum(distances[distances <= epsilon] - (epsilon * len(Y))) / len(Y)
    else:
        sumw = np.sum(weight)
        mask = distances <= epsilon
        loss = np.sum((distances[mask] - (epsilon * sumw)) * weight[mask]) / sumw
    if lambda1 > 0:
        loss += lambda1 * np.sum(np.abs(alpha))
    if lambda2 > 0:
        loss += lambda2 * np.sum(alpha * alpha)
    return loss


@jit(nopython=True, fastmath=True, parallel=True, cache=True)
def loss_numba(
    alpha: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    epsilon: float,
    beta: float,
    lambda2: float,
    weight: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray]:
    """
        Smoothed version of the loss, that also calculates the gradient.
        This function is sped up with numba.
    """
    epsilon *= epsilon
    distances = (X @ alpha) - Y
    distances2 = distances ** 2
    # Loss
    subset = 1 / (1 + np.exp(-beta * (epsilon - distances2)))  # Sigmoid
    n = len(Y) if weight is None else np.sum(weight)
    residuals = np.minimum(0, distances2 - (epsilon * n))
    if weight is None:
        loss = np.sum(subset * residuals) / n
    else:
        loss = np.sum(subset * residuals * weight) / n
    # Gradient
    k1 = 2.0 / n
    k2 = (-2.0 * beta / n) * (subset - subset ** 2)
    distances[residuals == 0] = 0.0
    if weight is None:
        grad = ((subset * k1) + (residuals * k2)) * distances
    else:
        grad = ((subset * k1) + (residuals * k2)) * (distances * weight)
    grad = np.expand_dims(grad, 0) @ X
    # Lambda
    if lambda2 > 0:
        loss = loss + lambda2 * np.sum(alpha * alpha)
        grad = grad + (lambda2 * 2) * alpha
    return loss, grad


@jit(nopython=True, fastmath=True, parallel=True, cache=True)
def ridge_numba(
    alpha: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    lambda2: float = 0.0,
    weight: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray]:
    """
        Ridge regression (OLS + L2) loss, that also calculates the gradient.
        This function is sped up with numba.
    """
    distances = (X @ alpha) - Y
    if weight is not None:
        distances *= weight
    loss = np.sum(distances ** 2) / 2
    grad = np.expand_dims(distances, 0) @ X
    if lambda2 != 0.0:
        loss += lambda2 * np.sum(alpha ** 2) / 2
        grad += lambda2 * np.reshape(alpha, grad.shape)
    return loss, grad


def owlqn(
    loss_grad_fn: Callable[[np.ndarray], Tuple[float, np.ndarray]],
    x0: np.ndarray,
    lambda1: float = 0,
    max_iterations: int = 200,
    **kwargs,
) -> np.ndarray:
    """
        Wrapper around owlqn that converts max_iter errors to warnings (see `fmin_lbfgs`).
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
        weight (Optional[np.ndarray], optional): Weight vector for the data items. Defaults to None.
        lambda1 (float, optional): LASSO/L1 regularisation coefficient. Defaults to 1e-6.
        lambda2 (float, optional): Ridge/L2 regularisation coefficient. Defaults to 1e-6.
        max_iterations (int, optional): Maximum number of optimisation steps. Defaults to 200.

    Returns:
        np.ndarray: The coefficients of the linear model.
    """
    return owlqn(
        lambda alpha: ridge_numba(alpha, X, Y, lambda2, weight),
        np.zeros(X.shape[1]),
        lambda1,
        max_iterations,
    )


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
    """
        Optimise a smoothed SLISE loss with `owl-qn`.
    """
    return owlqn(
        lambda alpha: loss_numba(alpha, X, Y, epsilon, beta, lambda2, weight),
        alpha,
        lambda1,
        max_iterations,
    )


def log_approximation_ratio(
    residuals2: np.ndarray,
    epsilon2: float,
    beta1: float,
    beta2: float,
    weight: Optional[np.ndarray] = None,
) -> float:
    """
        Calculate log(K), where K is the approximation ratio between two smoothed losses.
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
    beta: float = 0,
    weight: Optional[np.ndarray] = None,
    beta_max: float = 2500,
    log_max_approx: float = 0.14,
    min_beta_step: float = 0.0005,
) -> float:
    """
        Calculate the next beta for the graduated optimisation.
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
    """
        Approximately calculate the epsilon that minimises the approximation ratio to the exact loss.
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


def debug_log(
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
        alpha, residuals, epsilon ** 2, beta, lambda1, lambda2, weight
    )
    epss = matching_epsilon(residuals, epsilon ** 2, beta, weight)
    beta = beta * epsilon ** 2
    print(
        f"beta: {beta:5.3f}    epsilon*: {epss:.3f}    Loss: {loss:6.2f}    B-Loss: {bloss:6.2f}"
    )


def set_threads(num: int = -1) -> int:
    """Set the number of numba threads

    Args:
        num (int, optional): The number of threads. Defaults to -1.

    Returns:
        int: The old number of theads (or -1 if unchanged).
    """
    if num > 0:
        old = get_num_threads()
        set_num_threads(num)
        return old
    return -1


def check_threading_layer():
    """Check which numba threading_layer is active, and warn if it is "workqueue".
    """
    loss_residuals(np.ones(1), np.ones(1), 1)
    if threading_layer() == "workqueue":
        warn(
            'Using `numba.threading_layer()=="workqueue"` can be devastatingly slow! See https://numba.pydata.org/numba-doc/latest/user/threading-layer.html for alternatives.',
            SliseWarning,
        )


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
    """Optimise alpha using graduated optimisation.

    Args:
        alpha (np.ndarray): Initial alpha.
        X (np.ndarray): Data matrix.
        Y (np.ndarray): Response vector.
        epsilon (float): Error tolerance.
        beta (float, optional): Initial beta. Defaults to 0.
        lambda1 (float, optional): L1 regularisation strength. Defaults to 0.
        lambda2 (float, optional): L2 regularisation strength. Defaults to 0.
        weight (Optional[np.ndarray], optional): Weight vector for the data items. Defaults to None.
        beta_max (float, optional): Maximum sigmoid steepness (the final beta). Defaults to 20.
        max_approx (float, optional): Target approximation ratio when increasing beta. Defaults to 1.15.
        max_iterations (int, optional): Maximum number of iterations for owl-qn. Defaults to 200.
        debug (bool, optional): Print debug logs after each optimisation step. Defaults to False.

    Returns:
        np.ndarray: Optimised alpha.
    """
    X = np.asfortranarray(X, dtype=np.float64)
    Y = np.asfortranarray(Y, dtype=np.float64)
    if weight is not None:
        weight = np.asfortranarray(weight, dtype=np.float64)
    beta_max = beta_max / epsilon ** 2
    max_approx = log(max_approx)
    with catch_warnings(record=True) as w:
        while beta < beta_max:
            alpha = optimise_loss(
                alpha, X, Y, epsilon, beta, lambda1, lambda2, weight, max_iterations
            )
            if debug:
                debug_log(alpha, X, Y, epsilon, beta, lambda1, lambda2, weight)
            beta = next_beta(
                (X @ alpha - Y) ** 2, epsilon ** 2, beta, weight, beta_max, max_approx
            )
    alpha = optimise_loss(
        alpha, X, Y, epsilon, beta, lambda1, lambda2, weight, max_iterations * 4
    )
    if debug:
        debug_log(alpha, X, Y, epsilon, beta, lambda1, lambda2, weight)
        if w:
            print("Warnings from intermediate steps:", w)
    return alpha
