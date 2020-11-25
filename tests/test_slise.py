import numpy as np
from slise.utils import (
    sigmoid,
    log_sigmoid,
    sparsity,
    log_sum,
    log_sum_special,
    ridge_regression,
)
from slise.optimisation import (
    loss_smooth,
    loss_sharp,
    loss_numba,
    owlqn,
    graduated_optimisation,
)
from slise.data import add_intercept_column, local_into
from slise.initialisation import initialise_candidates
from slise.slise import regression, explain

from .utils import *


def test_utils():
    print("Testing util functions")
    x = np.arange(-6, 6)
    assert np.allclose(np.log(sigmoid(x)), log_sigmoid(x))
    assert sparsity(x) == len(x) - 1
    assert sparsity(x, 1.2) == len(x) - 3
    assert log_sum(x) == np.log(np.sum(np.exp(x)))
    assert log_sum(x) == log_sum_special(x, 0)
    assert log_sum(x) == log_sum_special(x, 1)
    assert log_sum(x) + np.log(2) == log_sum_special(x, 2)


def test_loss():
    print("Testing loss functions")
    X, Y = data_create(20, 5)
    alpha = np.random.normal(size=5)
    assert loss_smooth(alpha, X, Y) <= 0
    assert loss_sharp(alpha, X, Y) <= 0
    # assert loss_numba(alpha, X, Y)[0] <= 0
    assert loss_sharp(alpha, X, Y) <= 0
    assert loss_smooth(alpha, X, Y, 10) < 0
    assert loss_sharp(alpha, X, Y, 10) < 0
    # assert loss_numba(alpha, X, Y, 10)[0] < 0
    assert np.allclose(loss_smooth(alpha, X, Y, beta=1000000), loss_sharp(alpha, X, Y))
    assert np.allclose(
        loss_smooth(alpha, X, Y, lambda1=0.5, beta=1000000),
        loss_sharp(alpha, X, Y, lambda1=0.5),
    )
    assert np.allclose(
        loss_smooth(alpha, X, Y, lambda2=0.5, beta=1000000),
        loss_sharp(alpha, X, Y, lambda2=0.5),
    )
    # assert np.allclose(loss_smooth(alpha, X, Y, beta=100), loss_numba(alpha, X, Y, beta=100)[0])
    # assert np.allclose(loss_smooth(alpha, X, Y, beta=100, lambda2 = 0.5), loss_numba(alpha, X, Y, beta=100, lambda2 = 0.5)[0])


def test_owlqn():
    print("Testing owlqn")
    X, Y = data_create(20, 5)
    alpha = np.random.normal(size=5)
    alpha2 = owlqn(alpha, X, Y, beta=100)
    assert loss_smooth(alpha, X, Y, beta=100) >= loss_smooth(alpha2, X, Y, beta=100)
    alpha2 = owlqn(alpha, X, Y, beta=100, lambda1=0.5)
    assert loss_smooth(alpha, X, Y, beta=100, lambda1=0.5) > loss_smooth(
        alpha2, X, Y, beta=100, lambda1=0.5
    )
    alpha2 = owlqn(alpha, X, Y, beta=100, lambda2=0.5)
    assert loss_smooth(alpha, X, Y, beta=100, lambda2=0.5) > loss_smooth(
        alpha2, X, Y, beta=100, lambda2=0.5
    )


# def test_gradopt():
#     print("Testing graduated optimisation")
#     X, Y = data_create(20, 5)
#     alpha = np.random.normal(size=5)
#     alpha2 = graduated_optimisation(alpha, X, Y)
#     assert loss_smooth(alpha, X, Y, beta=100) >= loss_smooth(alpha2, X, Y, beta=100)
#     alpha2 = graduated_optimisation(alpha, X, Y, beta=100, lambda1=0.5)
#     assert loss_smooth(alpha, X, Y, beta=100, lambda1=0.5) > loss_smooth(
#         alpha2, X, Y, beta=100, lambda1=0.5
#     )
#     alpha2 = graduated_optimisation(alpha, X, Y, beta=100, lambda2=0.5)
#     assert loss_smooth(alpha, X, Y, beta=100, lambda2=0.5) > loss_smooth(
#         alpha2, X, Y, beta=100, lambda2=0.5
#     )


def test_initialise():
    print("Testing initialisation")
    X, Y = data_create(20, 5)
    zero = np.zeros(5)
    alpha, beta = initialise_candidates(X, Y, None)
    assert beta > 0
    assert loss_smooth(alpha, X, Y, beta=beta) <= loss_smooth(zero, X, Y, beta=beta)
    X, Y = data_create(20, 12)
    zero = np.zeros(12)
    alpha, beta = initialise_candidates(X, Y, None)
    assert beta > 0
    assert loss_smooth(alpha, X, Y, beta=beta) <= loss_smooth(zero, X, Y, beta=beta)
    X, Y = data_create(20, 11)
    X = add_intercept_column(X)
    zero = np.zeros(12)
    alpha, beta = initialise_candidates(X, Y, None, intercept=True)
    assert beta > 0
    assert loss_smooth(alpha, X, Y, beta=beta) <= loss_smooth(zero, X, Y, beta=beta)
    X, Y = data_create(20, 12)
    x = np.random.normal(size=12)
    X = local_into(X, x)
    zero = np.zeros(12)
    alpha, beta = initialise_candidates(X, Y, x)
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


def test_slise_exp():
    print("Testing slise explanation")
    X, Y, mod = data_create2(20, 5)
    x = np.random.normal(size=5)
    y = np.random.normal()
    reg = explain(
        X, Y, x, y, epsilon=0.1, lambda1=0.1, lambda2=0.1, scale_x=True, scale_y=True
    )
    assert reg.score() <= 0, f"Slise loss should usually be <=0 ({reg.score():.2f})"
    reg = explain(
        X, Y, x, y, epsilon=0.1, lambda1=0.1, lambda2=0.1, scale_x=False, scale_y=False
    )
    assert reg.score() <= 0, f"Slise loss should usually be <=0 ({reg.score():.2f})"
    reg = explain(
        X, Y, x, y, epsilon=0.1, lambda1=0, lambda2=0, scale_x=False, scale_y=False
    )
    assert reg.score() <= 0, f"Slise loss should usually be <=0 ({reg.score():.2f})"
    reg = explain(
        X, Y, 19, epsilon=0.1, lambda1=0.1, lambda2=0.1, scale_x=True, scale_y=True
    )
    assert reg.score() <= 0, f"Slise loss should usually be <=0 ({reg.score():.2f})"
    reg = explain(
        X, Y, 19, epsilon=0.1, lambda1=0.1, lambda2=0.1, scale_x=False, scale_y=False
    )
    assert reg.score() <= 0, f"Slise loss should usually be <=0 ({reg.score():.2f})"
    reg = explain(
        X, Y, 19, epsilon=0.1, lambda1=0, lambda2=0, scale_x=False, scale_y=False
    )
    assert reg.score() <= 0, f"Slise loss should usually be <=0 ({reg.score():.2f})"


#     old = np.seterr(over="ignore")
#     TESTS HERE
#     np.seterr(**old)
