# This script contains the optimisations for SLISE (Graduated Optimisation and OWL-QN)

import numpy as np
from lbfgs import fmin_lbfgs
from slise.utils import sigmoid

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

def optimise_owlqn(alpha, X, Y, epsilon = 0.1, lambda1 = 0, lambda2 = 0, beta = 3, max_iterations = 250):
    assert lambda1 >= 0
    line_search = "wolfe" if lambda1 > 0 else "default"
    return fmin_lbfgs(f = loss_owlqn, x0 = alpha, args = (X, Y, epsilon, lambda2, beta),
        orthantwise_c = lambda1, max_iterations = max_iterations, line_search = line_search)
