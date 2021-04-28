import numpy as np
from slise.data import (
    pca_simple,
    pca_invert,
    pca_rotate,
    pca_invert_model,
    pca_rotate_model,
    add_intercept_column,
    remove_intercept_column,
    remove_constant_columns,
    add_constant_columns,
    scale_robust,
    scale_same,
    mat_mul_with_intercept,
    unscale_model,
)

from .utils import *


def test_scaling():
    print("Testing scaling")
    for i in (4, 6, 8):
        X, Y = data_create(i * 30, i)
        X2, center, scale = scale_robust(X)
        assert np.allclose(scale_same(X, center, scale), X2)
        assert np.allclose(X2[0,], scale_same(X[0,], center, scale))
        X3 = add_intercept_column(X2)
        assert np.allclose(X2, remove_intercept_column(X3))
        X4, mask = remove_constant_columns(X3)
        assert np.allclose(X2, X4)
        assert np.allclose(mask, np.array([False] + [True] * i))
        assert np.allclose(X3[:, 1:], add_constant_columns(X2, mask)[:, 1:])
        Y2, center2, scale2 = scale_robust(Y)
        assert np.allclose(scale_same(Y, center2, scale2), Y2)
        assert np.allclose(scale_same(Y[0], center2, scale2), Y2[0])
        mod = np.random.normal(0, 1, i + 1)
        Y3 = mat_mul_with_intercept(X3, mod)
        mod2 = unscale_model(mod, center, scale, 0.0, 1.0)
        assert len(mod) == len(mod2)
        Y4 = mat_mul_with_intercept(X, mod2)
        assert np.allclose(Y3, Y4)


def test_pca():
    print("Testing pca")
    X = np.random.normal(size=[20, 5])
    X2, v = pca_simple(X, 5)
    assert np.allclose(X, pca_invert(X2, v))
    X3 = np.concatenate((X * 2, X), 1)
    assert pca_simple(X3, 10)[1].shape == (5, 10)
    assert np.allclose(X[0, :], pca_invert(pca_rotate(X[0, :], v), v))
    mod = np.random.normal(size=5)
    assert np.allclose(X @ mod, X2 @ pca_rotate_model(mod, v))
    assert np.allclose(X @ pca_invert_model(mod, v), X2 @ mod)
    X4, v = pca_simple(X.T, 4)

