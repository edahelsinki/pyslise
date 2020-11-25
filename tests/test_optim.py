import numpy as np
import pytest
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
    assert loss_smooth(alpha, X, Y, 0.1) <= 0
    assert loss_sharp(alpha, X, Y, 0.1) <= 0
    assert loss_numba(alpha, X, Y, 0.1, 0, 0)[0] <= 0
    assert loss_smooth(alpha, X, Y, 10) < 0
    assert loss_sharp(alpha, X, Y, 10) < 0
    assert loss_numba(alpha, X, Y, 10, 0, 0)[0] < 0
    assert np.allclose(loss_smooth(alpha, X, Y, beta=1000000), loss_sharp(alpha, X, Y))
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


def test_gradopt():
    print("Testing graduated optimisation")
    X, Y = data_create(20, 5)
    alpha = np.random.normal(size=5)
    alpha2 = graduated_optimisation(alpha, X, Y)
    assert loss_smooth(alpha, X, Y, beta=100) >= loss_smooth(alpha2, X, Y, beta=100)
    alpha2 = graduated_optimisation(alpha, X, Y, beta=100, lambda1=0.5)
    assert loss_smooth(alpha, X, Y, beta=100, lambda1=0.5) > loss_smooth(
        alpha2, X, Y, beta=100, lambda1=0.5
    )
    alpha2 = graduated_optimisation(alpha, X, Y, beta=100, lambda2=0.5)
    assert loss_smooth(alpha, X, Y, beta=100, lambda2=0.5) > loss_smooth(
        alpha2, X, Y, beta=100, lambda2=0.5
    )
