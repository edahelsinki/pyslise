# This script contains the optimisations for SLISE (Graduated Optimisation and OWL-QN)

from math import log
import numpy as np
from lbfgs import fmin_lbfgs
from scipy.optimize import brentq
from slise.utils import sigmoid, log_sigmoid, dlog_sigmoid, log_sum_special

def loss_smooth(alpha: np.ndarray, X: np.ndarray, Y: np.ndarray, epsilon: float = 0.1,
        lambda1: float = 0, lambda2: float = 0, beta: float = 100) -> float:
    epsilon *= epsilon
    distances = ((X @ alpha) - Y)**2
    subset = sigmoid(beta * (epsilon - distances))
    residuals = np.minimum(0, distances - epsilon * len(Y))
    loss = np.sum(subset * residuals) / len(Y)
    if lambda1 > 0:
        loss += lambda1 * np.sum(np.abs(alpha))
    if lambda2 > 0:
        loss += lambda2 * np.sum(alpha * alpha)
    return loss

def loss_residuals(alpha: np.ndarray, residuals2: np.ndarray, epsilon2: float = 0.01,
        lambda1: float = 0, lambda2: float = 0, beta: float = 100) -> float:
    # Takes squared residuals and epsilons
    subset = sigmoid(beta * (epsilon2 - residuals2))
    residuals = np.minimum(0, residuals2 - epsilon2 * len(residuals2))
    loss = np.sum(subset * residuals) / len(residuals2)
    if lambda1 > 0:
        loss += lambda1 * np.sum(np.abs(alpha))
    if lambda2 > 0:
        loss += lambda2 * np.sum(alpha * alpha)
    return loss

def loss_sharp(alpha: np.ndarray, X: np.ndarray, Y: np.ndarray, epsilon: float = 0.1,
        lambda1: float = 0, lambda2: float = 0) -> float:
    epsilon *= epsilon
    distances = ((X @ alpha) - Y)**2
    loss = np.sum(distances[distances < epsilon] - (epsilon * len(Y))) / len(Y)
    if lambda1 > 0:
        loss += lambda1 * np.sum(np.abs(alpha))
    if lambda2 > 0:
        loss += lambda2 * np.sum(alpha * alpha)
    return loss

def loss_owlqn(alpha: np.ndarray, gradient: np.ndarray, X: np.ndarray, Y: np.ndarray,
        epsilon: float = 0.1, lambda2: float = 0, beta: float = 100) -> float:
    np.nan_to_num(alpha, False, 0.0)
    epsilon *= epsilon
    distances = (X @ alpha) - Y
    distances2 = distances**2
    # Loss
    subset = sigmoid(beta * (epsilon - distances2))
    residuals = np.minimum(0, distances2 - epsilon * len(Y))
    loss = np.sum(subset * residuals) / len(Y)
    # Gradient
    k1 = 2.0 / len(Y)
    k2 = (-2.0 * beta / len(Y)) * (subset - subset**2)
    distances = np.where(residuals < 0, distances, 0)
    grad = ((subset * k1) + (residuals * k2))[np.newaxis, :] @ (X * distances[:, np.newaxis])
    # Lambda
    if lambda2 > 0:
        loss += lambda2 * np.sum(alpha * alpha)
        grad += lambda2 * 2 * alpha
    gradient[:] = grad
    return loss

def owlqn(alpha, X, Y, epsilon = 0.1, lambda1 = 0, lambda2 = 0, beta = 3, max_iterations = 200):
    assert lambda1 >= 0
    line_search = "wolfe" if lambda1 > 0 else "default"
    return fmin_lbfgs(f = loss_owlqn, x0 = alpha, args = (X, Y, epsilon, lambda2, beta),
        orthantwise_c = lambda1, max_iterations = max_iterations, line_search = line_search)

def log_approximation_ratio(residuals2: np.ndarray, epsilon2: float, beta1: float, beta2: float) -> float:
    if beta1 >= beta2:
        return 0                 
    log_f = lambda r, beta: log_sigmoid(beta * (epsilon2 - r))
    dlog_g = lambda r: - beta1 * dlog_sigmoid(beta1 * (epsilon2 - r)) + beta2 * dlog_sigmoid(beta2 * (epsilon2 - r))
    if dlog_g(0) < 0:
        a = brentq(dlog_g, 0, epsilon2)
        log_k = min(log_f(0, beta1) - log_f(0, beta2), log_f(a, beta1) - log_f(a, beta2))
    else:
        log_k = log_f(0, beta1) - log_f(0, beta2)
    phi = np.maximum(0, epsilon2 - residuals2 / len(residuals2))
    log_K = log_sum_special(log_f(residuals2, beta1), phi) - log_k - log_sum_special(log_f(residuals2, beta2), phi)
    return log_K

def matching_epsilon(residuals2: np.ndarray, epsilon2: float, beta: float) -> float:
    residuals2 = np.sort(residuals2)
    loss = sigmoid(beta * (epsilon2 - residuals2))
    i = np.argmax(np.arange(len(residuals2)) * loss)
    return residuals2[i]**0.5

def next_beta(residuals2: np.ndarray, epsilon2: float = 0.01, beta: float = 0, beta_max: float = 2500,
        log_max_approx: float = log(1.15), min_beta_step: float = 0.0005):
    if (beta >= beta_max):
        return beta
    log_approx = log_approximation_ratio(residuals2, epsilon2, beta, beta_max)
    if log_approx <= log_max_approx:
        return beta_max
    else:
        f = lambda b: log_approximation_ratio(residuals2, epsilon2, beta, b) - log_max_approx
        beta_min = beta + min_beta_step * (beta_max + beta)
        return max(brentq(f, beta, beta_max), beta_min)

def graduated_optimisation(alpha, X, Y, epsilon = 0.1, lambda1 = 0, lambda2 = 0, beta = 0, beta_max = 25,
        max_approx = 1.15, max_iterations = 200, min_beta_step = 0.0005, *args):
    max_approx = log(max_approx)
    beta_max = beta_max / epsilon**2
    while beta < beta_max:
        alpha = owlqn(alpha, X, Y, epsilon, lambda1, lambda2, beta, max_iterations)
        beta = next_beta((X @ alpha - Y)**2, epsilon**2, beta, beta_max, max_approx, min_beta_step)
    alpha = owlqn(alpha, X, Y, epsilon, lambda1, lambda2, beta, max_iterations * 4)
    return alpha
