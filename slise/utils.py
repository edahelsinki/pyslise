# This script contains some utility functions

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    # Computes probabilities from logits
    return 1 / (1 + np.exp(-x))

def logit(p: np.ndarray, stab : float = 0.001) -> np.ndarray:
    # Computes logits from probabilities
    # p is limited to [stab, 1-stab] for numerical stability
    p = np.clip(p, stab, 1 - stab)
    return np.log(p / (1.0 - p))

def dsigmoid(x: np.ndarray) -> np.ndarray:
    # Derivative of the sigmoid function
    s = sigmoid(x)
    return s * (1 - s)

def log_sigmoid(x: np.ndarray) -> np.ndarray:
    # Numerically stable calculation of log(sigmoid(x)):
    #   ifelse(x >= 0, - log(1 + exp(-x)), x - log(1 + exp(x)))
    y = -np.sign(x)
    return (y * 0.5 + 0.5) * x - np.log1p(np.exp(y * x))

def dlog_sigmoid(x: np.ndarray) -> np.ndarray:
    # Derivative of the log_sigmoid function
    return 1 - sigmoid(x)

def sparsity(x: np.ndarray, treshold: float = 0) -> float:
    # Count the number of abs(x) > treshold
    if treshold > 0:
        return np.count_nonzero(np.abs(x) > treshold)
    else:
        return np.count_nonzero(x)

def log_sum(x: np.ndarray) -> float:
    # Computes log(sum(exp(x))) in a numerically robust way
    xmax = np.max(x)
    return xmax + np.log(np.sum(np.exp(x - xmax)))
