import numpy as np
import pytest
from pytest import approx
from slise.optimisation import (
    check_threading_layer,
    graduated_optimisation,
    log_approximation_ratio,
    loss_grad,
    loss_sharp,
    loss_smooth,
    matching_epsilon,
    next_beta,
    optimise_loss,
    regularised_regression,
)
from slise.utils import log_sigmoid, log_sum_exp, log_sum_special, sigmoid, sparsity

from .utils import *


def test_utils():
    print("Testing util functions")
    x = np.arange(-6, 6)
    assert np.allclose(np.log(sigmoid(x)), log_sigmoid(x))
    assert sparsity(x) == len(x) - 1
    assert sparsity(x, 1.2) == len(x) - 3
    assert log_sum_exp(x) == np.log(np.sum(np.exp(x)))
    assert log_sum_exp(x) == log_sum_special(x, 0)
    assert log_sum_exp(x) == log_sum_special(x, 1)
    assert log_sum_exp(x) + np.log(2) == log_sum_special(x, 2)


def test_loss():
    print("Testing loss functions")
    X, Y = data_create(20, 5)
    w = np.random.uniform(size=20)
    alpha = np.random.normal(size=5)
    assert loss_smooth(alpha, X, Y, 0.1) <= 0
    assert loss_sharp(alpha, X, Y, 0.1) <= 0
    assert loss_grad(alpha, X, Y, 0.1, lambda2=0, beta=0)[0] <= 0
    assert loss_smooth(alpha, X, Y, 10) < 0
    assert loss_sharp(alpha, X, Y, 10) < 0
    assert loss_grad(alpha, X, Y, 10, lambda2=0, beta=0)[0] < 0
    assert np.allclose(
        loss_smooth(alpha, X, Y, 0.1, beta=1000000), loss_sharp(alpha, X, Y, 0.1)
    )
    assert np.allclose(
        loss_smooth(alpha, X, Y, 0.1, lambda1=0.5, beta=1000000),
        loss_sharp(alpha, X, Y, 0.1, lambda1=0.5),
    )
    assert np.allclose(
        loss_smooth(alpha, X, Y, 0.1, lambda2=0.5, beta=1000000),
        loss_sharp(alpha, X, Y, 0.1, lambda2=0.5),
    )
    assert loss_smooth(alpha, X, Y, 0.1, beta=20, lambda1=0.0, lambda2=0.0) == approx(
        loss_grad(alpha, X, Y, 0.1, lambda2=0.0, beta=20)[0], 1e-8
    )
    assert loss_smooth(alpha, X, Y, 0.1, beta=20, lambda1=0.0, lambda2=0.5) == approx(
        loss_grad(alpha, X, Y, 0.1, lambda2=0.5, beta=20)[0], 1e-8
    )
    # With weight
    assert loss_smooth(alpha, X, Y, 0.1, weight=w) <= 0
    assert loss_sharp(alpha, X, Y, 0.1, weight=w) <= 0
    assert loss_grad(alpha, X, Y, 0.1, lambda2=0, beta=0, weight=w)[0] <= 0
    assert loss_smooth(alpha, X, Y, 10, weight=w) < 0
    assert loss_sharp(alpha, X, Y, 10, weight=w) < 0
    assert loss_grad(alpha, X, Y, 10, lambda2=0, beta=0, weight=w)[0] < 0
    assert np.allclose(
        loss_smooth(alpha, X, Y, 0.1, beta=1000000, weight=w),
        loss_sharp(alpha, X, Y, 0.1, weight=w),
    )
    assert np.allclose(
        loss_smooth(alpha, X, Y, 0.1, lambda1=0.5, beta=1000000, weight=w),
        loss_sharp(alpha, X, Y, 0.1, lambda1=0.5, weight=w),
    )
    assert np.allclose(
        loss_smooth(alpha, X, Y, 0.1, lambda2=0.5, beta=1000000, weight=w),
        loss_sharp(alpha, X, Y, 0.1, lambda2=0.5, weight=w),
    )
    assert loss_smooth(alpha, X, Y, 0.1, beta=20, weight=w, lambda2=0.0) == approx(
        loss_grad(alpha, X, Y, 0.1, lambda2=0.0, weight=w, beta=20)[0], 1e-8
    )
    assert loss_smooth(alpha, X, Y, 0.1, beta=20, weight=w, lambda2=0.5) == approx(
        loss_grad(alpha, X, Y, 0.1, lambda2=0.5, weight=w, beta=20)[0], 1e-8
    )


def test_grad_numerically():
    print("Comparing the manual gradient to a numeric gradient")
    for _ in range(3):
        X, Y = data_create(20, 5)
        alpha = np.random.normal(size=5)
        _, grad = loss_grad(alpha, X, Y, 10, lambda2=0, beta=0)
        grad2 = numeric_grad(
            alpha, lambda x: loss_grad(x, X, Y, 10, lambda2=0, beta=0)[0]
        )
        assert np.allclose(grad, grad2, atol=1e-4)
        w = np.random.uniform(size=20)
        _, grad = loss_grad(alpha, X, Y, 10, lambda2=0, beta=0, weight=w)
        grad2 = numeric_grad(
            alpha, lambda x: loss_grad(x, X, Y, 10, lambda2=0, beta=0, weight=w)[0]
        )
        assert np.allclose(grad, grad2, atol=1e-4)


def test_owlqn():
    print("Testing owlqn")
    X, Y = data_create(20, 5)
    alpha = np.random.normal(size=5)
    alpha2 = optimise_loss(alpha, X, Y, 0.1, beta=100)
    assert loss_smooth(alpha, X, Y, 0.1, beta=100) >= loss_smooth(
        alpha2, X, Y, 0.1, beta=100
    )
    alpha2 = optimise_loss(alpha, X, Y, 0.1, beta=100, lambda1=0.5)
    assert loss_smooth(alpha, X, Y, 0.1, beta=100, lambda1=0.5) >= loss_smooth(
        alpha2, X, Y, 0.1, beta=100, lambda1=0.5
    )
    alpha2 = optimise_loss(alpha, X, Y, 0.1, beta=100, lambda2=0.5)
    assert loss_smooth(alpha, X, Y, 0.1, beta=100, lambda2=0.5) >= loss_smooth(
        alpha2, X, Y, 0.1, beta=100, lambda2=0.5
    )
    # With weight
    w = np.random.uniform(size=20)
    alpha2 = optimise_loss(alpha, X, Y, 0.1, beta=100, weight=w)
    assert loss_smooth(alpha, X, Y, 0.1, beta=100, weight=w) >= loss_smooth(
        alpha2, X, Y, 0.1, beta=100, weight=w
    )
    alpha2 = optimise_loss(alpha, X, Y, 0.1, beta=100, lambda1=0.5, weight=w)
    assert loss_smooth(
        alpha, X, Y, 0.1, beta=100, lambda1=0.5, weight=w
    ) >= loss_smooth(alpha2, X, Y, 0.1, beta=100, lambda1=0.5, weight=w)
    alpha2 = optimise_loss(alpha, X, Y, 0.1, beta=100, lambda2=0.5, weight=w)
    assert loss_smooth(
        alpha, X, Y, 0.1, beta=100, lambda2=0.5, weight=w
    ) >= loss_smooth(alpha2, X, Y, 0.1, beta=100, lambda2=0.5, weight=w)


def test_gradopt():
    print("Testing graduated optimisation")
    X, Y = data_create(20, 5)
    alpha = np.random.normal(size=5)
    alpha2 = graduated_optimisation(alpha, X, Y, 0.1, beta=100)
    assert loss_smooth(alpha, X, Y, 0.1, beta=100) >= loss_smooth(
        alpha2, X, Y, 0.1, beta=100
    )
    alpha2 = graduated_optimisation(alpha, X, Y, 0.1, beta=100, lambda1=0.5)
    assert loss_smooth(alpha, X, Y, 0.1, beta=100, lambda1=0.5) >= loss_smooth(
        alpha2, X, Y, 0.1, beta=100, lambda1=0.5
    )
    alpha2 = graduated_optimisation(alpha, X, Y, 0.1, beta=100, lambda2=0.5)
    assert loss_smooth(alpha, X, Y, 0.1, beta=100, lambda2=0.5) >= loss_smooth(
        alpha2, X, Y, 0.1, beta=100, lambda2=0.5
    )
    # With weight
    w = np.random.uniform(size=20)
    alpha2 = graduated_optimisation(alpha, X, Y, 0.1, beta=100, weight=w)
    assert loss_smooth(alpha, X, Y, 0.1, beta=100, weight=w) >= loss_smooth(
        alpha2, X, Y, 0.1, beta=100, weight=w
    )
    alpha2 = graduated_optimisation(alpha, X, Y, 0.1, beta=100, lambda1=0.5, weight=w)
    assert loss_smooth(
        alpha, X, Y, 0.1, beta=100, lambda1=0.5, weight=w
    ) >= loss_smooth(alpha2, X, Y, 0.1, beta=100, lambda1=0.5, weight=w)
    alpha2 = graduated_optimisation(alpha, X, Y, 0.1, beta=100, lambda2=0.5, weight=w)
    assert loss_smooth(
        alpha, X, Y, 0.1, beta=100, lambda2=0.5, weight=w
    ) >= loss_smooth(alpha2, X, Y, 0.1, beta=100, lambda2=0.5, weight=w)


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
    w = np.ones(20)
    alpha = regularised_regression(X, Y, 1e-10, 1e-10, w)
    Y2 = X @ alpha
    assert np.allclose(Y, Y2, atol=0.3), f"regregw Y not close: {Y - Y2}"
    assert np.allclose(mod, alpha, atol=0.2), f"regregw alpha not close: {mod - alpha}"


def test_weights():
    X, Y, mod = data_create2(20, 5)
    w1 = None
    w2 = np.ones(20)
    w3 = w2 * 2
    X2 = np.concatenate((X, X), 0)
    Y2 = np.concatenate((Y, Y), 0)
    alpha = np.random.normal(size=5)
    assert np.allclose(
        loss_sharp(alpha, X, Y, 0.1, weight=w1),
        loss_sharp(alpha, X, Y, 0.1, weight=w2),
    )
    assert np.allclose(
        loss_sharp(alpha, X2, Y2, 0.1, weight=w1),
        loss_sharp(alpha, X, Y, 0.1, weight=w3),
    )
    assert np.allclose(
        loss_smooth(alpha, X, Y, 0.1, 10, weight=w1),
        loss_smooth(alpha, X, Y, 0.1, 10, weight=w2),
    )
    assert np.allclose(
        loss_smooth(alpha, X2, Y2, 0.1, 10, weight=w1),
        loss_smooth(alpha, X, Y, 0.1, 10, weight=w3),
    )
    assert np.allclose(
        graduated_optimisation(alpha, X, Y, 0.1, beta=100, lambda2=0.5, weight=w1),
        graduated_optimisation(alpha, X, Y, 0.1, beta=100, lambda2=0.5, weight=w2),
    )
    assert np.allclose(
        graduated_optimisation(alpha, X2, Y2, 0.1, beta=100, lambda2=0.5, weight=w1),
        graduated_optimisation(alpha, X, Y, 0.1, beta=100, lambda2=0.5, weight=w3),
    )
    assert np.allclose(
        matching_epsilon((X @ alpha - Y) ** 2, 0.01, 10, w1),
        matching_epsilon((X @ alpha - Y) ** 2, 0.01, 10, w2),
    )
    assert np.allclose(
        matching_epsilon((X2 @ alpha - Y2) ** 2, 0.01, 100, w1),
        matching_epsilon((X @ alpha - Y) ** 2, 0.01, 100, w3),
    )
    assert np.allclose(
        next_beta((X @ alpha - Y) ** 2, 0.01, 100, w1),
        next_beta((X @ alpha - Y) ** 2, 0.01, 100, w2),
    )
    assert np.allclose(
        next_beta((X2 @ alpha - Y2) ** 2, 0.01, 100, w1),
        next_beta((X @ alpha - Y) ** 2, 0.01, 100, w3),
    )
    assert np.allclose(
        log_approximation_ratio((X @ alpha - Y) ** 2, 0.01, 1, 100, w1),
        log_approximation_ratio((X @ alpha - Y) ** 2, 0.01, 1, 100, w2),
    )
    assert np.allclose(
        log_approximation_ratio((X2 @ alpha - Y2) ** 2, 0.01, 1, 100, w1),
        log_approximation_ratio((X @ alpha - Y) ** 2, 0.01, 1, 100, w3),
    )
    assert np.allclose(
        regularised_regression(X, Y, 1e-4, 1e-4, weight=w1),
        regularised_regression(X, Y, 1e-4, 1e-4, weight=w2),
    )
    assert np.allclose(
        regularised_regression(X2, Y2, 1e-4, 1e-4, weight=w1, max_iterations=300),
        regularised_regression(X, Y, 1e-4, 1e-4, weight=w3, max_iterations=300),
    )


def test_check_threading_layer():
    check_threading_layer()
