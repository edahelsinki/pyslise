from typing import Tuple
import numpy as np
import numpy.random as npr


def data_create(n: int, d: int, c: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    X = npr.normal(npr.normal(size=d)[np.newaxis,], 1.0, [n, d])
    if d > c:
        X[:, c] = 0
    Y = npr.normal(size=n) + npr.normal()
    return X, Y


def data_create2(
    n: int, d: int, c: int = 2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = npr.normal(npr.normal(size=d)[np.newaxis,], 1.0, [n, d])
    mod = npr.normal(size=d)
    if d > c:
        X[:, c] = 0
        mod[c] = 0
    Y = X @ mod + npr.normal(size=n, scale=0.05)
    return X, Y, mod


def numeric_grad(x: np.ndarray, fn, eps=1e-6) -> np.ndarray:
    grad = x * 0
    for i, _ in enumerate(grad):
        x1 = x.copy()
        x1[i] += eps
        val1 = fn(x1)
        x1[i] = x[i] - eps
        val2 = fn(x1)
        grad[i] = (val1 - val2) / (eps * 2)
    return grad
