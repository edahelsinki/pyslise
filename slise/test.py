# This script contains unit tests

import numpy as np
from slise.utils import sigmoid, log_sigmoid, sparsity, log_sum, log_sum_special
from slise.optimisation import loss_smooth, loss_sharp, loss_owlqn, owlqn, graduated_optimisation

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
    X = np.random.normal(size=[20, 5])
    Y = np.random.normal(size=20)
    alpha = np.random.normal(size=5)
    grad = alpha.copy()
    assert loss_smooth(alpha, X, Y) <= 0
    assert loss_sharp(alpha, X, Y) <= 0
    assert loss_owlqn(alpha, grad, X, Y) <= 0
    assert loss_sharp(alpha, X, Y) <= 0
    assert loss_smooth(alpha, X, Y, 10) < 0
    assert loss_sharp(alpha, X, Y, 10) < 0
    assert loss_owlqn(alpha, grad, X, Y, 10) < 0
    assert np.allclose(loss_smooth(alpha, X, Y, beta=1000000), loss_sharp(alpha, X, Y))
    assert np.allclose(loss_smooth(alpha, X, Y, lambda1 = 0.5, beta=1000000), loss_sharp(alpha, X, Y, lambda1 = 0.5))
    assert np.allclose(loss_smooth(alpha, X, Y, lambda2 = 0.5, beta=1000000), loss_sharp(alpha, X, Y, lambda2 = 0.5))
    assert np.allclose(loss_smooth(alpha, X, Y, beta=100), loss_owlqn(alpha, grad, X, Y, beta=100))
    assert np.allclose(loss_smooth(alpha, X, Y, beta=100, lambda2 = 0.5), loss_owlqn(alpha, grad, X, Y, beta=100, lambda2 = 0.5))

def test_owlqn():
    print("Testing owlqn")
    X = np.random.normal(size=[20, 5])
    Y = np.random.normal(size=20)
    alpha = np.random.normal(size=5)
    alpha2 = owlqn(alpha, X, Y, beta = 100)
    assert loss_smooth(alpha, X, Y, beta = 100) >= loss_smooth(alpha2, X, Y, beta = 100)
    alpha2 = owlqn(alpha, X, Y, beta = 100, lambda1 = 0.5)
    assert loss_smooth(alpha, X, Y, beta = 100, lambda1 = 0.5) >= loss_smooth(alpha2, X, Y, beta = 100, lambda1 = 0.5)
    alpha2 = owlqn(alpha, X, Y, beta = 100, lambda2 = 0.5)
    assert loss_smooth(alpha, X, Y, beta = 100, lambda2 = 0.5) >= loss_smooth(alpha2, X, Y, beta = 100, lambda2 = 0.5)

def test_gradopt():
    print("Testing graduated optimisation")
    X = np.random.normal(size=[20, 5])
    Y = np.random.normal(size=20)
    alpha = np.random.normal(size=5)
    alpha2 = graduated_optimisation(alpha, X, Y)
    assert loss_smooth(alpha, X, Y, beta = 100) >= loss_smooth(alpha2, X, Y, beta = 100)
    alpha2 = graduated_optimisation(alpha, X, Y, beta = 100, lambda1 = 0.5)
    assert loss_smooth(alpha, X, Y, beta = 100, lambda1 = 0.5) >= loss_smooth(alpha2, X, Y, beta = 100, lambda1 = 0.5)
    alpha2 = graduated_optimisation(alpha, X, Y, beta = 100, lambda2 = 0.5)
    assert loss_smooth(alpha, X, Y, beta = 100, lambda2 = 0.5) >= loss_smooth(alpha2, X, Y, beta = 100, lambda2 = 0.5)

if __name__ == "__main__":
    old = np.seterr(over='ignore')
    test_utils()
    test_loss()
    test_owlqn()
    test_gradopt()
    np.seterr(**old)
