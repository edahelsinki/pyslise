import numpy as np
import numpy.random as npr


def data_create(n: int, d: int) -> (np.ndarray, np.ndarray):
    X = npr.normal(npr.normal(size=d)[np.newaxis,], 1.0, [n, d])
    Y = npr.normal(size=n) + npr.normal()
    return X, Y


def data_create2(n: int, d: int) -> (np.ndarray, np.ndarray):
    X = npr.normal(npr.normal(size=d)[np.newaxis,], 1.0, [n, d])
    mod = npr.normal(size=d)
    Y = X @ mod + npr.normal(size=n, scale=0.05)
    return X, Y, mod
