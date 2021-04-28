import numpy as np
from slise.utils import ridge_regression
from slise.optimisation import loss_smooth
from slise.data import add_intercept_column
from slise.initialisation import (
    initialise_candidates,
    initialise_candidates2,
    initialise_lasso,
    initialise_ols,
    initialise_zeros,
)
from slise.slise import regression, explain

from .utils import *


def test_initialise_old():
    print("Testing old initialisations")
    X, Y = data_create(20, 5)
    alpha, beta = initialise_lasso(X, Y)
    assert beta == 0
    assert len(alpha) == 5
    alpha, beta = initialise_zeros(X, Y)
    assert beta > 0
    assert len(alpha) == 5
    alpha, beta = initialise_ols(X, Y)
    assert beta > 0
    assert len(alpha) == 5


def test_initialise():
    print("Testing initialisation")
    X, Y = data_create(20, 5)
    zero = np.zeros(5)
    alpha, beta = initialise_candidates(X, Y)
    assert beta > 0
    assert loss_smooth(alpha, X, Y, beta=beta) <= loss_smooth(zero, X, Y, beta=beta)
    X, Y = data_create(20, 12)
    zero = np.zeros(12)
    alpha, beta = initialise_candidates(X, Y)
    assert beta > 0
    assert loss_smooth(alpha, X, Y, beta=beta) <= loss_smooth(zero, X, Y, beta=beta)
    X, Y = data_create(20, 11)
    X = add_intercept_column(X)
    zero = np.zeros(12)
    alpha, beta = initialise_candidates(X, Y)
    assert beta > 0
    assert loss_smooth(alpha, X, Y, beta=beta) <= loss_smooth(zero, X, Y, beta=beta)
    X, Y = data_create(20, 12)
    x = np.random.normal(size=12)
    X = X - x[None, :]
    zero = np.zeros(12)
    alpha, beta = initialise_candidates(X, Y)
    assert beta > 0
    assert loss_smooth(alpha, X, Y, beta=beta) <= loss_smooth(zero, X, Y, beta=beta)


def test_initialise2():
    print("Testing initialisation2")
    X, Y = data_create(20, 5)
    zero = np.zeros(5)
    alpha, beta = initialise_candidates2(X, Y)
    assert beta > 0
    assert loss_smooth(alpha, X, Y, beta=beta) <= loss_smooth(zero, X, Y, beta=beta)
    X, Y = data_create(20, 12)
    zero = np.zeros(12)
    alpha, beta = initialise_candidates2(X, Y)
    assert beta > 0
    assert loss_smooth(alpha, X, Y, beta=beta) <= loss_smooth(zero, X, Y, beta=beta)
    X, Y = data_create(20, 11)
    X = add_intercept_column(X)
    zero = np.zeros(12)
    alpha, beta = initialise_candidates2(X, Y)
    assert beta > 0
    assert loss_smooth(alpha, X, Y, beta=beta) <= loss_smooth(zero, X, Y, beta=beta)
    X, Y = data_create(20, 12)
    x = np.random.normal(size=12)
    X = X - x[None, :]
    zero = np.zeros(12)
    alpha, beta = initialise_candidates2(X, Y)
    assert beta > 0
    assert loss_smooth(alpha, X, Y, beta=beta) <= loss_smooth(zero, X, Y, beta=beta)


def test_ridge():
    print("Testing ridge regression")
    X, Y, mod = data_create2(20, 5)
    alpha = ridge_regression(X, Y, 1e-10)
    Y2 = X @ alpha
    assert np.allclose(Y, Y2, atol=0.3), f"Ridge Y not close: {Y - Y2}"
    assert np.allclose(mod, alpha, atol=0.2), f"Ridge alpha not close: {mod - alpha}"


def test_slise_reg():
    print("Testing slise regression")
    X, Y, mod = data_create2(20, 5)
    reg = regression(
        X,
        Y,
        epsilon=0.1,
        lambda1=0.01,
        lambda2=0.01,
        intercept=True,
        scale_x=True,
        scale_y=True,
    )
    assert reg.score() <= 0, f"Slise loss should usually be <=0 ({reg.score():.2f})"
    reg = regression(
        X,
        Y,
        epsilon=0.1,
        lambda1=0.01,
        lambda2=0.01,
        intercept=True,
        scale_x=False,
        scale_y=False,
    )
    assert reg.score() <= 0, f"Slise loss should usually be <=0 ({reg.score():.2f})"
    reg = regression(
        X,
        Y,
        epsilon=0.1,
        lambda1=0,
        lambda2=0,
        intercept=True,
        scale_x=False,
        scale_y=False,
    )
    assert reg.score() <= 0, f"Slise loss should usually be <=0 ({reg.score():.2f})"


# def test_slise_exp():
#     print("Testing slise explanation")
#     X, Y, mod = data_create2(20, 5)
#     x = np.random.normal(size=5)
#     y = np.random.normal()
#     reg = explain(
#         X, Y, x, y, epsilon=0.1, lambda1=0.1, lambda2=0.1, scale_x=True, scale_y=True
#     )
#     assert reg.score() <= 0, f"Slise loss should usually be <=0 ({reg.score():.2f})"
#     reg = explain(
#         X, Y, x, y, epsilon=0.1, lambda1=0.1, lambda2=0.1, scale_x=False, scale_y=False
#     )
#     assert reg.score() <= 0, f"Slise loss should usually be <=0 ({reg.score():.2f})"
#     reg = explain(
#         X, Y, x, y, epsilon=0.1, lambda1=0, lambda2=0, scale_x=False, scale_y=False
#     )
#     assert reg.score() <= 0, f"Slise loss should usually be <=0 ({reg.score():.2f})"
#     reg = explain(
#         X, Y, 19, epsilon=0.1, lambda1=0.1, lambda2=0.1, scale_x=True, scale_y=True
#     )
#     assert reg.score() <= 0, f"Slise loss should usually be <=0 ({reg.score():.2f})"
#     reg = explain(
#         X, Y, 19, epsilon=0.1, lambda1=0.1, lambda2=0.1, scale_x=False, scale_y=False
#     )
#     assert reg.score() <= 0, f"Slise loss should usually be <=0 ({reg.score():.2f})"
#     reg = explain(
#         X, Y, 19, epsilon=0.1, lambda1=0, lambda2=0, scale_x=False, scale_y=False
#     )
#     assert reg.score() <= 0, f"Slise loss should usually be <=0 ({reg.score():.2f})"
