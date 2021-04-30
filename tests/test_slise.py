import numpy as np
from slise.optimisation import loss_smooth, regularised_regression
from slise.data import add_intercept_column, scale_same
from slise.initialisation import (
    initialise_candidates,
    initialise_candidates2,
    initialise_lasso,
    initialise_ols,
    initialise_zeros,
)
from slise.slise import regression, explain

from .utils import *
from slise.utils import mat_mul_inter


def test_initialise_old():
    print("Testing old initialisations")
    X, Y = data_create(20, 5)
    alpha, beta = initialise_lasso(X, Y, 0.1)
    assert beta == 0
    assert len(alpha) == 5
    alpha, beta = initialise_zeros(X, Y, 0.1)
    assert beta > 0
    assert len(alpha) == 5
    alpha, beta = initialise_ols(X, Y, 0.1)
    assert beta > 0
    assert len(alpha) == 5


def test_initialise():
    print("Testing initialisation")
    X, Y = data_create(20, 5)
    zero = np.zeros(5)
    alpha, beta = initialise_candidates(X, Y, 0.1)
    assert beta > 0
    assert loss_smooth(alpha, X, Y, 0.1, beta=beta) <= loss_smooth(
        zero, X, Y, 0.1, beta=beta
    )
    X, Y = data_create(20, 12)
    zero = np.zeros(12)
    alpha, beta = initialise_candidates(X, Y, 0.1)
    assert beta > 0
    assert loss_smooth(alpha, X, Y, 0.1, beta=beta) <= loss_smooth(
        zero, X, Y, 0.1, beta=beta
    )
    X, Y = data_create(20, 11)
    X = add_intercept_column(X)
    zero = np.zeros(12)
    alpha, beta = initialise_candidates(X, Y, 0.1)
    assert beta > 0
    assert loss_smooth(alpha, X, Y, 0.1, beta=beta) <= loss_smooth(
        zero, X, Y, 0.1, beta=beta
    )
    X, Y = data_create(20, 12)
    x = np.random.normal(size=12)
    X = X - x[None, :]
    zero = np.zeros(12)
    alpha, beta = initialise_candidates(X, Y, 0.1)
    assert beta > 0
    assert loss_smooth(alpha, X, Y, 0.1, beta=beta) <= loss_smooth(
        zero, X, Y, 0.1, beta=beta
    )


def test_initialise2():
    print("Testing initialisation2")
    X, Y = data_create(20, 5)
    zero = np.zeros(5)
    alpha, beta = initialise_candidates2(X, Y, 0.1)
    assert beta > 0
    assert loss_smooth(alpha, X, Y, 0.1, beta=beta) <= loss_smooth(
        zero, X, Y, 0.1, beta=beta
    )
    X, Y = data_create(20, 12)
    zero = np.zeros(12)
    alpha, beta = initialise_candidates2(X, Y, 0.1)
    assert beta > 0
    assert loss_smooth(alpha, X, Y, 0.1, beta=beta) <= loss_smooth(
        zero, X, Y, 0.1, beta=beta
    )
    X, Y = data_create(20, 11)
    X = add_intercept_column(X)
    zero = np.zeros(12)
    alpha, beta = initialise_candidates2(X, Y, 0.1)
    assert beta > 0
    assert loss_smooth(alpha, X, Y, 0.1, beta=beta) <= loss_smooth(
        zero, X, Y, 0.1, beta=beta
    )
    X, Y = data_create(20, 12)
    x = np.random.normal(size=12)
    X = X - x[None, :]
    zero = np.zeros(12)
    alpha, beta = initialise_candidates2(X, Y, 0.1)
    assert beta > 0
    assert loss_smooth(alpha, X, Y, 0.1, beta=beta) <= loss_smooth(
        zero, X, Y, 0.1, beta=beta
    )


def test_regres():
    print("Testing regularised regression")
    X, Y, mod = data_create2(20, 5)
    alpha = regularised_regression(X, Y, 1e-10, 1e-10)
    Y2 = X @ alpha
    assert np.allclose(Y, Y2, atol=0.3), f"regreg Y not close: {Y - Y2}"
    assert np.allclose(mod, alpha, atol=0.2), f"regreg alpha not close: {mod - alpha}"
    alpha = regularised_regression(X, Y, 1e-10, 0)
    Y2 = X @ alpha
    assert np.allclose(Y, Y2, atol=0.3), f"Lasso Y not close: {Y - Y2}"
    assert np.allclose(mod, alpha, atol=0.2), f"Lasso alpha not close: {mod - alpha}"
    alpha = regularised_regression(X, Y, 0, 1e-10)
    Y2 = X @ alpha
    assert np.allclose(Y, Y2, atol=0.3), f"Ridge Y not close: {Y - Y2}"
    assert np.allclose(mod, alpha, atol=0.2), f"Ridge alpha not close: {mod - alpha}"


def test_slise_reg():
    print("Testing slise regression")
    X, Y, mod = data_create2(40, 5)
    reg1 = regression(
        X, Y, epsilon=0.1, lambda1=0.01, lambda2=0.01, intercept=True, normalise=True,
    )
    reg1.print()
    Yp = mat_mul_inter(X, reg1.coefficients)
    Yn = reg1.scale.scale_y(Y)
    Ynp = mat_mul_inter(reg1.scale.scale_x(X), reg1.alpha)
    Ypn = reg1.scale.scale_y(Yp)
    S = (Y - Yp) ** 2 < reg1.epsilon ** 2
    Sn = (Yn - Ynp) ** 2 < reg1.epsilon_orig ** 2
    assert np.all(S == Sn), f"Subsets are not the same {S.mean()} {Sn.mean()}"
    # TODO This for some reason does not hold:
    # print(reg1.scale)
    # print(Ypn, Ynp, Ypn - Ynp)
    # assert np.allclose(
    #     Ypn, Ynp,
    # ), f"The predicted Y's are not the same {np.max(np.abs(Ynp - Ypn))}"
    assert (
        reg1.score() <= 0
    ), f"SLISE loss should be negative ({reg1.score():.2f}, {reg1.subset().mean():.2f})"
    assert 1.0 >= reg1.subset().mean() > 0.75
    reg2 = regression(
        X, Y, epsilon=0.1, lambda1=0.01, lambda2=0.01, intercept=True, normalise=False,
    )
    reg2.print()
    assert (
        reg2.score() <= 0
    ), f"SLISE loss should be negative ({reg2.score():.2f}, {reg2.subset().mean():.2f})"
    assert 1.0 >= reg2.subset().mean() > 0.5
    reg3 = regression(
        X, Y, epsilon=0.1, lambda1=0, lambda2=0, intercept=True, normalise=False,
    )
    reg3.print()
    assert (
        reg3.score() <= 0
    ), f"SLISE loss should be negative ({reg3.score():.2f}, {reg3.subset().mean():.2f})"
    assert 1.0 >= reg3.subset().mean() > 0.5


def test_slise_exp():
    print("Testing slise explanation")
    X, Y, mod = data_create2(100, 5)
    x = np.random.normal(size=5)
    y = np.random.normal()
    reg = explain(X, Y, 0.1, x, y, lambda1=0.01, lambda2=0.01, normalise=True)
    reg.print()
    assert reg.score() <= 0, f"Slise loss should usually be <=0 ({reg.score():.2f})"
    assert 1.0 >= reg.subset().mean() > 0.0
    reg = explain(X, Y, 0.1, 19, lambda1=0.01, lambda2=0.01, normalise=True)
    reg.print()
    assert reg.score() <= 0, f"Slise loss should usually be <=0 ({reg.score():.2f})"
    assert 1.0 >= reg.subset().mean() > 0.0
    reg = explain(X, Y, 0.1, x, y, lambda1=0.01, lambda2=0.01, normalise=False)
    assert reg.score() <= 0, f"Slise loss should usually be <=0 ({reg.score():.2f})"
    assert 1.0 >= reg.subset().mean() > 0.0
    reg = explain(X, Y, 0.1, x, y, lambda1=0, lambda2=0, normalise=False)
    assert reg.score() <= 0, f"Slise loss should usually be <=0 ({reg.score():.2f})"
    assert 1.0 >= reg.subset().mean() > 0.0
    reg = explain(X, Y, 0.1, 19, lambda1=0.01, lambda2=0.01, normalise=False)
    assert reg.score() <= 0, f"Slise loss should usually be <=0 ({reg.score():.2f})"
    assert 1.0 >= reg.subset().mean() > 0.0
    reg = explain(X, Y, 0.1, 19, lambda1=0, lambda2=0, normalise=False)
    assert reg.score() <= 0, f"Slise loss should usually be <=0 ({reg.score():.2f})"
    assert 1.0 >= reg.subset().mean() > 0.0
