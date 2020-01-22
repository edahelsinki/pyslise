# This script contains some utility functions

from random import randrange
from math import log
from warnings import warn
import numpy as np
from scipy.special import expit as sigmoid
from lbfgs import fmin_lbfgs


class SliseWarning(RuntimeWarning):
    """
        Custom tag for the warnings
    """


def dsigmoid(x: np.ndarray) -> np.ndarray:
    """
        Derivative of the sigmoid function
    """
    s = sigmoid(x)
    return s * (1 - s)

def log_sigmoid(x: np.ndarray) -> np.ndarray:
    """
        Numerically stable calculation of log(sigmoid(x)):
            ifelse(x >= 0, - log(1 + exp(-x)), x - log(1 + exp(x)))
    """
    y = -np.sign(x)
    return (y * 0.5 + 0.5) * x - np.log1p(np.exp(y * x))

def dlog_sigmoid(x: np.ndarray) -> np.ndarray:
    """
        Derivative of the log_sigmoid function
    """
    return 1 - sigmoid(x)

def sparsity(x: np.ndarray, treshold: float = 0) -> float:
    """
        Count the number of abs(x) > treshold
    """
    if treshold > 0:
        return np.count_nonzero(np.abs(x) > treshold)
    else:
        return np.count_nonzero(x)

def log_sum(x: np.ndarray) -> float:
    """
        Computes log(sum(exp(x))) in a numerically stable way
    """
    xmax = np.max(x)
    return xmax + log(np.sum(np.exp(x - xmax)))

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
    return xmax + log(xsum)

def random_sample_int(n: int, k: int) -> list:
    """
        Get k random, but unique, integers from the interval [0,n)
    """
    if n < k:
        raise Exception("random_sample_int: n must be equal or larger than k")
    if n == k:
        return list(range(k))
    indices = [randrange(0, n)] * k
    for i in range(1, k):
        new = randrange(0, n)
        while new in indices[:k]:
            new = randrange(0, n)
        indices[i] = new
    return indices

def ridge_regression(X: np.ndarray, Y: np.ndarray, lambda2:float = 1e-10) -> np.ndarray:
    """Train a linear ridge regression model

    Arguments:
        X {np.ndarray} -- the data
        Y {np.ndarray} -- the response

    Keyword Arguments:
        lambda2 {float} -- the L2 regularisation coefficient (default: {1e-6})

    Returns:
        np.ndarray -- the linear model weights
    """
    def f(alpha: np.ndarray, gradient: np.ndarray) -> float:
        residuals = X @ alpha - Y
        gradient[:] = 2 * (X.T @ residuals) + (lambda2 * 2) * alpha
        return np.sum(residuals**2) + lambda2 * np.sum(alpha**2)
    return fmin_lbfgs(f = f, x0 = np.zeros(X.shape[1]))

def fill_column_names(names: list = None, amount: int = -1, intercept: bool = False) -> list:
    """Make sure the list of column names is of the correct size

    Keyword Arguments:
        names {list} -- prefilled list of column names (default: {None})
        amount {int} -- the number of columns, including intercept (default: {-1})
        intercept {bool} -- is the first column an intercept column (default: {False})

    Returns:
        list -- list of column names
    """
    if amount < 1:
        return names
    if names is None:
        if intercept:
            return ["Intercept"] + ["Col %d"%i for i in range(amount-1)]
        else:
            return ["Col %d"%i for i in range(amount)]
    elif len(names) == amount:
        if intercept:
            warn("No room to add the name for the intercept column", SliseWarning)
        return names
    elif len(names) == amount - 1 and intercept:
        return ["Intercept"] + names
    elif len(names) > amount:
        warn("Too many column names given", SliseWarning)
        return names[:amount]
    else:
        warn("Too few column names given", SliseWarning)
        if intercept:
            names = ["Intercept"] + names
        return names + ["Col %d"%i for i in range(len(names), amount)]

def fill_prediction_str(y: float, class_names: list = None, decimals: int = 3) -> str:
    """Fill a string with the prediction meassage for explanations

    Arguments:
        y {float} -- the prediction

    Keyword Arguments:
        class_names {list} -- list of class names, if classification (default: {None})
        decimals {int} -- the decimal precision (default: {3})

    Returns:
        str -- the formatted message
    """
    if class_names is not None:
        if len(class_names) > 1:
            if y >= 0.0 and y <= 1.0:
                if y > 0.5:
                    return f"Predicted: {y*100:.{decimals}f} % {class_names[1]}"
                else:
                    return f"Predicted: {(1-y)*100:.{decimals}f} % {class_names[0]}"
            else:
                if y > 0:
                    return f"Predicted: {y:.{decimals}f} {class_names[1]}"
                else:
                    return f"Predicted: {-y:.{decimals}f} {class_names[0]}"
        else:
            if y >= 0.0 and y <= 1.0:
                return f"Predicted: {y*100:.{decimals}f} % {class_names[0]}"
            else:
                return f"Predicted: {y:.{decimals}f} {class_names[0]}"
    else:
        return f"Predicted: {y:.{decimals}f}"
