"""
    This script contains some utility functions.
"""

from typing import Union

import numpy as np
from scipy.special import expit as sigmoid


class SliseWarning(RuntimeWarning):
    """
    Custom tag for warnings.
    """


class SliseException(Exception):
    """
    Custom tag for exceptions.
    """


def limited_logit(
    p: Union[np.ndarray, float], stab: float = 0.001
) -> Union[np.ndarray, float]:
    """Computes logits from probabilities.

    Args:
        p (Union[np.ndarray, float]): Probability vector or scalar.
        stab (float, optional): Limit p to [stab, 1-stab] for numerical stability. Defaults to 0.001.

    Returns:
        Union[np.ndarray, float]: `logit(clamp(p, stab, 1-stab))`.
    """
    p = np.minimum(1.0 - stab, np.maximum(stab, p))
    return np.log(p / (1.0 - p))


def dsigmoid(x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """Derivative of the sigmoid function.

    Args:
        x (Union[np.ndarray, float]): Real vector or scalar.

    Returns:
        Union[np.ndarray, float]: Derivative of `sigmoid(x)`.
    """
    s = sigmoid(x)
    return s * (1 - s)


def log_sigmoid(x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """Computes `log(sigmoid(x))` in a numerically stable way.

    Args:
        x (Union[np.ndarray, float]): Real vector or scalar.

    Returns:
        Union[np.ndarray, float]: `log(sigmoid(x))`
    """
    y = -np.sign(x)
    return (y * 0.5 + 0.5) * x - np.log1p(np.exp(y * x))


def dlog_sigmoid(x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """Derivative of `log(sigmoid(x))`.

    Args:
        x (Union[np.ndarray, float]): Real vector or scalar.

    Returns:
        Union[np.ndarray, float]: Derivative of `log(sigmoid(x))`
    """
    return 1 - sigmoid(x)


def sparsity(x: Union[np.ndarray, float], treshold: float = 0) -> int:
    """Count the number of `abs(x) > treshold`.

    Args:
        x (Union[np.ndarray, float]): Real vector or scalar.
        treshold (float, optional): Threshold non-inclusive. Defaults to 0.

    Returns:
        int: The number of `abs(x) > treshold`.
    """
    if treshold > 0:
        return np.count_nonzero(np.abs(x) > treshold)
    else:
        return np.count_nonzero(x)


def log_sum_exp(x: np.ndarray) -> float:
    """Computes `log(sum(exp(x)))` in a numerically stable way.

    Args:
        x (np.ndarray): Real vector.

    Returns:
        float: `log(sum(exp(x)))`
    """
    xmax = np.max(x)
    return xmax + np.log(np.sum(np.exp(x - xmax)))


def log_sum_special(x: np.ndarray, y: np.ndarray) -> float:
    """Computes `log(sum(exp(x) * y))` (or `log(sum(exp(x)))` if `all(y == 0)`), in a numerically stable way.

    Args:
        x (np.ndarray): Real vector.
        y (np.ndarray): Real vector.

    Returns:
        float: `log(sum(exp(x) * y))`.
    """
    xmax = np.max(x)
    xexp = np.exp(x - xmax)
    xsum = np.sum(xexp * y)
    if xsum == 0:
        xsum = np.sum(xexp)
    return xmax + np.log(xsum)


def mat_mul_inter(X: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Matrix multiplication, but check and handle potential intercepts in `alpha`.

    Args:
        X (np.ndarray): Real matrix or vector.
        alpha (np.ndarray): Real vector.

    Returns:
        np.ndarray: `X @ alpha` or `X @ alpha[1:] + alpha[0]`.
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
