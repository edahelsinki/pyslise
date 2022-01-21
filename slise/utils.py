# This script contains some utility functions

from typing import Union

import numpy as np
from scipy.special import expit as sigmoid


class SliseWarning(RuntimeWarning):
    """
        Custom tag for warnings
    """


class SliseException(Exception):
    """
        Custom tag for exceptions
    """


def limited_logit(
    p: Union[np.ndarray, float], stab: float = 0.001
) -> Union[np.ndarray, float]:
    """Computes the logits from probabilities

    Args:
        p (Union[np.ndarray, float]): probability vector
        stab (float, optional): limit p to [stab, 1-stab] for numerical stability. Defaults to 0.001.

    Returns:
        Union[np.ndarray, float]: logit(clamp(p, stab, 1-stab))
    """
    p = np.minimum(1.0 - stab, np.maximum(stab, p))
    return np.log(p / (1.0 - p))


def dsigmoid(x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """
        Derivative of the sigmoid function
    """
    s = sigmoid(x)
    return s * (1 - s)


def log_sigmoid(x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """
        Numerically stable calculation of log(sigmoid(x)):
            ifelse(x >= 0, - log(1 + exp(-x)), x - log(1 + exp(x)))
    """
    y = -np.sign(x)
    return (y * 0.5 + 0.5) * x - np.log1p(np.exp(y * x))


def dlog_sigmoid(x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """
        Derivative of the log_sigmoid function
    """
    return 1 - sigmoid(x)


def sparsity(x: Union[np.ndarray, float], treshold: float = 0) -> int:
    """
        Count the number of abs(x) > treshold
    """
    if treshold > 0:
        return np.count_nonzero(np.abs(x) > treshold)
    else:
        return np.count_nonzero(x)


def log_sum_exp(x: np.ndarray) -> float:
    """
        Computes log(sum(exp(x))) in a numerically stable way
    """
    xmax = np.max(x)
    return xmax + np.log(np.sum(np.exp(x - xmax)))


def log_sum_special(x: np.ndarray, y: np.ndarray) -> float:
    """
        Computes log(sum(exp(x) * y)), or log(sum(exp(x)))
            if all(y == 0), in a numerically robust way
    """
    xmax = np.max(x)
    xexp = np.exp(x - xmax)
    xsum = np.sum(xexp * y)
    if xsum == 0:
        xsum = np.sum(xexp)
    return xmax + np.log(xsum)


def mat_mul_inter(X: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """
        Matrix multiplication, but check and handle potential intercepts in alpha
    """
    alpha = np.atleast_1d(alpha)
    if len(X.shape) == 1:
        if len(alpha) == X.size:
            return np.sum(alpha[1:] * X)
        if len(alpha) == X.size + 1:
            return alpha[0] + np.sum(alpha[1:] * X)
        else:
            X = np.reshape(X, X.shape + (1,))
    if len(alpha) == X.shape[1] + 1:
        return X @ alpha[1:] + alpha[0]
    else:
        return X @ alpha
