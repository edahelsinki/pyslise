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
    normalise_robust,
    scale_same,
    unscale_model,
)
from slise.utils import mat_mul_inter

from .utils import *


def test_scaling():
    print("Testing scaling")
    for i in (4, 6, 8):
        X, Y = data_create(i * 30, i, 100000)
        X2, center, scale = normalise_robust(X)
        assert np.allclose(scale_same(X, center, scale), X2)
        assert np.allclose(X2[0,], scale_same(X[0,], center, scale))
        X3 = add_intercept_column(X2)
        assert np.allclose(X2, remove_intercept_column(X3))
        X4, mask = remove_constant_columns(X3)
        assert np.allclose(X2, X4)
        assert np.allclose(mask, np.array([False] + [True] * i))
        assert np.allclose(X3[:, 1:], add_constant_columns(X2, mask)[:, 1:])
        Y2, center2, scale2 = normalise_robust(Y)
        assert np.allclose(scale_same(Y, center2, scale2), Y2)
        assert np.allclose(scale_same(Y[0], center2, scale2), Y2[0])


def test_model_scaling():
    print("Testing model scaling")
    for i in (4, 6, 8):
        X, Y, model2 = data_create2(i * 30, i)
        X2, x_center, x_scale = normalise_robust(X)
        Y2, y_center, y_scale = normalise_robust(Y)
        model2 = np.random.normal(size=i)
        model = unscale_model(model2, x_center, x_scale, y_center, y_scale)
        Z1 = mat_mul_inter(X, model)
        Z2 = mat_mul_inter(X2, model2)
        Z3 = scale_same(Z1, y_center, y_scale)
        assert np.allclose(Z2, Z3), f"Max Diff {np.max(np.abs(Z2 - Z3))}"
        model2 = np.random.normal(size=i + 1)
        model = unscale_model(model2, x_center, x_scale, y_center, y_scale)
        Z1 = mat_mul_inter(X, model)
        Z2 = mat_mul_inter(X2, model2)
        Z3 = scale_same(Z1, y_center, y_scale)
        assert np.allclose(Z2, Z3), f"Max Diff {np.max(np.abs(Z2 - Z3))}"


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

