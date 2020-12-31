"""
    This is an attempt at modelling SLISE (or something similar)
    using MIP (mixed integer programming) to find exact solutions.
    The idea would be to compare these solutions to the ones given
    by the SLISE algorithm. However, this is __A LOT__ slower than
    the graduated optimisation of SLISE.
"""
from zipfile import ZipFile
import mip
import numpy as np
import pandas as pd


def model_sq(X, Y, epsilon, lambda1):
    # This is the "correct" SLISE model, but it does
    #   not work since it is not a linear (MIP) model.
    mod = mip.Model("SLISE")
    alpha = mod.add_var_tensor((X.shape[1],), name="alpha", lb=-mip.INF)
    subset = mod.add_var_tensor((X.shape[0],), name="subset", var_type=mip.BINARY)
    residuals2 = (X @ alpha - Y) ** 2

    mod.objective = mip.minimize(
        mip.xsum(subset * (residuals2 / X.shape[0] - epsilon ^ 2))
        + lambda1 * mip.xsum(alpha)
    )
    mod += residuals2 * subset <= epsilon ^ 2

    return mod


def model_ln(X, Y, epsilon: float, lambda1: float = 0.0) -> mip.Model:
    # This is a linear model (which means it works).
    # However, it only gives the subset and does not support sparsity.
    mod = mip.Model("SLISE")
    alpha = mod.add_var_tensor((X.shape[1],), name="alpha", lb=-mip.INF)
    subset = mod.add_var_tensor(
        (X.shape[0],), name="subset", ub=1.0, var_type=mip.BINARY
    )
    residuals = X @ alpha - Y
    M = (Y.max() - Y.min()) * 100  # This should be a very large number

    mod.objective = mip.maximize(mip.xsum(subset))
    mod += residuals - epsilon <= M * (1 - subset)
    mod += -residuals - epsilon <= M * (1 - subset)

    return mod


def preprocess(X, Y, intercept=True):
    Y2 = Y - np.median(Y)
    Y2 = Y2 / np.median(np.abs(Y2))
    X2 = X - np.median(X, 0, keepdims=True)
    X2 = X2 / np.median(np.abs(X2), 0, keepdims=True)
    if intercept:
        X2 = np.concatenate((np.ones(Y.shape + (1,)), X2), 1)
    return X2, Y2


def data_air_quality(path="experiments/data/air_quality.zip", file="AirQualityUCI.csv"):
    with ZipFile(path) as zf:
        with zf.open(file) as df:
            df = pd.read_csv(
                df,
                sep=";",
                delimiter=";",
                decimal=",",
                usecols=(2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14),
                na_values=-200,
                dtype=np.float32,
            ).dropna()
            return df[df.columns[1:]].to_numpy(), df[df.columns[0]].to_numpy()


if __name__ == "__main__":
    X, Y = data_air_quality()
    X, Y = preprocess(X, Y)
    model = model_ln(X, Y, 0.5, 1e-6)
    res = model.optimize(max_seconds=1200)
    print("Subset size:", np.array([x.x for x in model.vars[X.shape[1] :]]).sum())
    # Result: This is very slow (> 20 minutes) and does not find as good of a solution as SLISE (< 1 second)
