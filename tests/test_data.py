import numpy as np
from slise.data import (
    ScalerNormal,
    ScalerRange,
    ScalerLocal,
    ScalerRemoveConstant,
    pca_simple,
    pca_invert,
    pca_rotate,
    DataScaler,
    pca_invert_model,
    pca_rotate_model,
    add_intercept_column,
    ScalerNested,
    ScalerIdentity,
    local_into,
)

from .utils import *


def test_scaling():
    print("Testing scaling")
    X, Y = data_create(20, 5)
    X = add_intercept_column(X)
    scalers = [
        ScalerNormal(),
        ScalerRange(),
        ScalerRemoveConstant(),
        ScalerIdentity(),
        ScalerLocal(np.random.normal(size=6)),
        ScalerNested(ScalerNormal(), ScalerRange()),
    ]
    for sc in scalers:
        X2 = sc.fit(X)
        X3 = sc.scale(X)
        X4 = sc.unscale(X2)
        X5 = sc.unscale(sc.scale(X[1, :]))
        assert np.allclose(X, X4), f"scale-unscale X failed for {sc}"
        assert np.allclose(X2, X3), f"unscale-scale X failed for {sc}"
        assert np.allclose(X[1, :], X5), f"scale-unscale X vector failed for {sc}"
    scalers = [
        ScalerNormal(),
        ScalerRange(),
        ScalerRemoveConstant(),
        ScalerIdentity(),
        ScalerLocal(np.random.normal(size=1)),
        ScalerNested(ScalerNormal(), ScalerRange()),
    ]
    for sc in scalers:
        X2 = sc.fit(Y)
        X3 = sc.scale(Y)
        X4 = sc.unscale(X2)
        X5 = sc.unscale(sc.scale(Y[1]))
        assert np.allclose(Y, X4), f"scale-unscale Y failed for {sc}"
        assert np.allclose(X2, X3), f"unscale-scale Y failed for {sc}"
        assert np.allclose(Y[1], X5), f"scale-unscale Y scalar failed for {sc}"


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
