# This script contains some utility functions

from math import log
import numpy as np
from scipy.special import expit as sigmoid

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
    return xmax + log(np.sum(np.exp(x - xmax)))

def log_sum_special(x: np.ndarray, y: np.ndarray) -> float:
    # Computes log(sum(exp(x) * y)),
    # or log(sum(exp(x))) if all(y == 0),
    # in a numerically robust way
    xmax = np.max(x)
    xexp = np.exp(x - xmax)
    xsum = np.sum(xexp * y)
    if xsum == 0:
        xsum = np.sum(xexp)
    return xmax + log(xsum)
