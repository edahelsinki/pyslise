# This script contains unit tests

import numpy as np
from slise.utils import sigmoid, log_sigmoid, sparsity, log_sum, log_sum_special
from slise.optimisation import loss_smooth, loss_sharp, loss_numba, owlqn, graduated_optimisation
from slise.data import ScalerNormal, ScalerRange, ScalerLocal, ScalerRemoveConstant, pca_simple,\
    pca_invert, pca_rotate, DataScaler, pca_invert_model, pca_rotate_model, add_intercept_column,\
    ScalerNested, DataScaler, ScalerIdentity, local_into
from slise.initialisation import initialise_candidates


def data_create(n:int, d:int) -> (np.ndarray, np.ndarray):
    X = np.random.normal(size=[n, d]) + np.random.normal(size=d)[np.newaxis, ]
    Y = np.random.normal(size=n) + np.random.normal()
    return X, Y


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
    assert loss_numba(alpha, X, Y)[0] <= 0
    assert loss_sharp(alpha, X, Y) <= 0
    assert loss_smooth(alpha, X, Y, 10) < 0
    assert loss_sharp(alpha, X, Y, 10) < 0
    assert loss_numba(alpha, X, Y, 10)[0] < 0
    assert np.allclose(loss_smooth(alpha, X, Y, beta=1000000), loss_sharp(alpha, X, Y))
    assert np.allclose(loss_smooth(alpha, X, Y, lambda1 = 0.5, beta=1000000), loss_sharp(alpha, X, Y, lambda1 = 0.5))
    assert np.allclose(loss_smooth(alpha, X, Y, lambda2 = 0.5, beta=1000000), loss_sharp(alpha, X, Y, lambda2 = 0.5))
    assert np.allclose(loss_smooth(alpha, X, Y, beta=100), loss_numba(alpha, X, Y, beta=100)[0])
    assert np.allclose(loss_smooth(alpha, X, Y, beta=100, lambda2 = 0.5), loss_numba(alpha, X, Y, beta=100, lambda2 = 0.5)[0])

def test_owlqn():
    print("Testing owlqn")
    X, Y = data_create(20, 5)
    alpha = np.random.normal(size=5)
    alpha2 = owlqn(alpha, X, Y, beta = 100)
    assert loss_smooth(alpha, X, Y, beta = 100) >= loss_smooth(alpha2, X, Y, beta = 100)
    alpha2 = owlqn(alpha, X, Y, beta = 100, lambda1 = 0.5)
    assert loss_smooth(alpha, X, Y, beta = 100, lambda1 = 0.5) > loss_smooth(alpha2, X, Y, beta = 100, lambda1 = 0.5)
    alpha2 = owlqn(alpha, X, Y, beta = 100, lambda2 = 0.5)
    assert loss_smooth(alpha, X, Y, beta = 100, lambda2 = 0.5) > loss_smooth(alpha2, X, Y, beta = 100, lambda2 = 0.5)

def test_gradopt():
    print("Testing graduated optimisation")
    X, Y = data_create(20, 5)
    alpha = np.random.normal(size=5)
    alpha2 = graduated_optimisation(alpha, X, Y)
    assert loss_smooth(alpha, X, Y, beta = 100) >= loss_smooth(alpha2, X, Y, beta = 100)
    alpha2 = graduated_optimisation(alpha, X, Y, beta = 100, lambda1 = 0.5)
    assert loss_smooth(alpha, X, Y, beta = 100, lambda1 = 0.5) > loss_smooth(alpha2, X, Y, beta = 100, lambda1 = 0.5)
    alpha2 = graduated_optimisation(alpha, X, Y, beta = 100, lambda2 = 0.5)
    assert loss_smooth(alpha, X, Y, beta = 100, lambda2 = 0.5) > loss_smooth(alpha2, X, Y, beta = 100, lambda2 = 0.5)

def test_scaling():
    print("Testing scaling")
    X, Y = data_create(20, 5)
    X = add_intercept_column(X)
    scalers = [ScalerNormal(), ScalerRange(), ScalerRemoveConstant(), ScalerIdentity(),
        ScalerLocal(np.random.normal(size=6)), ScalerNested(ScalerNormal(), ScalerRange())]
    for sc in scalers:
        X2 = sc.fit(X)
        X3 = sc.scale(X)
        X4 = sc.unscale(X2)
        X5 = sc.unscale(sc.scale(X[1,:]))
        assert np.allclose(X, X4), f"scale-unscale failed for {sc}"
        assert np.allclose(X2, X3), f"unscale-scale failed for {sc}"
        assert np.allclose(X[1, :], X5)
    scalers = [ScalerNormal(), ScalerRange(), ScalerRemoveConstant(), ScalerIdentity(),
        ScalerLocal(np.random.normal(size=1)), ScalerNested(ScalerNormal(), ScalerRange())]
    for sc in scalers:
        X2 = sc.fit(Y)
        X3 = sc.scale(Y)
        X4 = sc.unscale(X2)
        X5 = sc.unscale(sc.scale(Y[1]))
        assert np.allclose(Y, X4)
        assert np.allclose(X2, X3)
        assert np.allclose(Y[1], X5)

def test_pca():
    print("Testing pca")
    X = np.random.normal(size=[20, 5])
    X2, v = pca_simple(X, 5)
    assert np.allclose(X, pca_invert(X2, v))
    X3 = np.concatenate((X*2, X), 1)
    assert pca_simple(X3, 10)[1].shape == (5, 10)
    assert np.allclose(X[0, :], pca_invert(pca_rotate(X[0, :], v), v))
    mod = np.random.normal(size=5)
    assert np.allclose(X @ mod, X2 @ pca_rotate_model(mod, v))
    assert np.allclose(X @ pca_invert_model(mod, v), X2 @ mod)
    X4, v = pca_simple(X.T, 4)


def test_data_scaler():
    print("Testing model scaling")
    X, Y = data_create(20, 5)
    Y = np.random.uniform(0, 1, 5)
    sc = DataScaler(True, True, True, True)
    X2, Y2 = sc.fit(X, Y)
    X3, Y3 = sc.unscale(X2, Y2)
    assert np.allclose(X, X3)
    assert np.allclose(Y, Y3)
    mod = np.random.normal(size=5)
    mod2 = sc.scale_model(mod)
    mod3 = sc.unscale_model(mod2)
    mod4 = sc.unscale_model(mod)
    mod5 = sc.scale_model(mod4)
    assert np.allclose(mod, mod3[1:])
    assert np.allclose(mod, mod5[1:])

def test_initialise():
    print("Testing initialisation")
    X, Y = data_create(20, 5)
    zero = np.zeros(5)
    alpha, beta = initialise_candidates(X, Y, None)
    assert beta > 0
    assert loss_smooth(alpha, X, Y, beta = beta) <= loss_smooth(zero, X, Y, beta = beta)
    X, Y = data_create(20, 12)
    zero = np.zeros(12)
    alpha, beta = initialise_candidates(X, Y, None)
    assert beta > 0
    assert loss_smooth(alpha, X, Y, beta = beta) <= loss_smooth(zero, X, Y, beta = beta)
    X, Y = data_create(20, 11)
    X = add_intercept_column(X)
    zero = np.zeros(12)
    alpha, beta = initialise_candidates(X, Y, None, intercept=True)
    assert beta > 0
    assert loss_smooth(alpha, X, Y, beta = beta) <= loss_smooth(zero, X, Y, beta = beta)
    X, Y = data_create(20, 12)
    x = np.random.normal(size=12)
    X = local_into(X, x)
    zero = np.zeros(12)
    alpha, beta = initialise_candidates(X, Y, x)
    assert beta > 0
    assert loss_smooth(alpha, X, Y, beta = beta) <= loss_smooth(zero, X, Y, beta = beta)

#TODO: tests for slise.slise

if __name__ == "__main__":
    old = np.seterr(over='ignore')
    test_utils()
    test_loss()
    test_owlqn()
    test_gradopt()
    test_scaling()
    test_pca()
    test_data_scaler()
    test_initialise()
    np.seterr(**old)
    print("All tests completed")
