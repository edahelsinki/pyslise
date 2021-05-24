import numpy as np
import pytest
from slise.utils import (
    sigmoid,
    log_sigmoid,
    sparsity,
    log_sum,
    log_sum_special,
)
from slise.optimisation import (
    loss_smooth,
    loss_sharp,
    loss_numba,
    optimise_loss,
    graduated_optimisation,
    regularised_regression,
)
from slise.data import add_intercept_column
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
    assert loss_smooth(alpha, X, Y, 0.1) <= 0
    assert loss_sharp(alpha, X, Y, 0.1) <= 0
    assert loss_numba(alpha, X, Y, 0.1, 0, 0)[0] <= 0
    assert loss_smooth(alpha, X, Y, 10) < 0
    assert loss_sharp(alpha, X, Y, 10) < 0
    assert loss_numba(alpha, X, Y, 10, 0, 0)[0] < 0
    assert np.allclose(
        loss_smooth(alpha, X, Y, 0.1, beta=1000000), loss_sharp(alpha, X, Y, 0.1)
    )
    assert np.allclose(
        loss_smooth(alpha, X, Y, 0.1, 0.5, beta=1000000),
        loss_sharp(alpha, X, Y, 0.1, 0.5),
    )
    assert np.allclose(
        loss_smooth(alpha, X, Y, 0.1, 0.5, beta=1000000),
        loss_sharp(alpha, X, Y, 0.1, 0.5),
    )
    assert (
        pytest.approx(loss_smooth(alpha, X, Y, 0.1, 0.0, 0.0, 20), 1e-8)
        == loss_numba(alpha, X, Y, 0.1, 0.0, 20)[0]
    )
    assert (
        pytest.approx(loss_smooth(alpha, X, Y, 0.1, 0.0, 0.5, 20), 1e-8)
        == loss_numba(alpha, X, Y, 0.1, 0.5, 20)[0]
    )


def test_owlqn():
    print("Testing owlqn")
    X, Y = data_create(20, 5)
    alpha = np.random.normal(size=5)
    alpha2 = optimise_loss(alpha, X, Y, 0.1, beta=100)
    assert loss_smooth(alpha, X, Y, 0.1, beta=100) >= loss_smooth(
        alpha2, X, Y, 0.1, beta=100
    )
    alpha2 = optimise_loss(alpha, X, Y, 0.1, beta=100, lambda1=0.5)
    assert loss_smooth(alpha, X, Y, 0.1, beta=100, lambda1=0.5) > loss_smooth(
        alpha2, X, Y, 0.1, beta=100, lambda1=0.5
    )
    alpha2 = optimise_loss(alpha, X, Y, 0.1, beta=100, lambda2=0.5)
    assert loss_smooth(alpha, X, Y, 0.1, beta=100, lambda2=0.5) > loss_smooth(
        alpha2, X, Y, 0.1, beta=100, lambda2=0.5
    )


def test_gradopt():
    print("Testing graduated optimisation")
    X, Y = data_create(20, 5)
    alpha = np.random.normal(size=5)
    alpha2 = graduated_optimisation(alpha, X, Y, 0.1,)
    assert loss_smooth(alpha, X, Y, 0.1, beta=100) >= loss_smooth(
        alpha2, X, Y, 0.1, beta=100
    )
    alpha2 = graduated_optimisation(alpha, X, Y, 0.1, beta=100, lambda1=0.5)
    assert loss_smooth(alpha, X, Y, 0.1, beta=100, lambda1=0.5) > loss_smooth(
        alpha2, X, Y, 0.1, beta=100, lambda1=0.5
    )
    alpha2 = graduated_optimisation(alpha, X, Y, 0.1, beta=100, lambda2=0.5)
    assert loss_smooth(alpha, X, Y, 0.1, beta=100, lambda2=0.5) > loss_smooth(
        alpha2, X, Y, 0.1, beta=100, lambda2=0.5
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

