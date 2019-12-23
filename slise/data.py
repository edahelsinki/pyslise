# This script contains functions for modifying data

import numpy as np

def add_intercept_column(X: np.ndarray) -> np.ndarray:
    if len(X.shape) == 1:
        return np.concatenate(([1.0], X))
    return np.concatenate((np.ones((X.shape[0], 1)), X), 1)

def scale_normal(X: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    if len(X.shape) == 1:
        mean = np.mean(X)
        X = X - mean
        scale = np.sqrt(np.sum(np.square(X)) / (len(X) - 1))
        if scale == 0:
            scale = 1.0
        X = X / scale
        return X, mean, scale, 0
    else:
        mean = np.mean(X, 0)
        X = X - mean[np.newaxis, :]
        scale = np.sqrt(np.sum(np.square(X), 0) / (X.shape[0] - 1))
        mask = np.nonzero(scale)
        if isinstance(mask, tuple):
            mask = mask[-1]
        if len(mask) == 0:
            mask = np.arange(len(scale))
        elif len(mask) != len(scale):
            scale = scale[mask]
            X = X[:, mask]
        X = X / scale[np.newaxis, :]
        return X, mean, scale, mask

def scale_range(X: np.ndarray, quantiles: list = [0.05, 0.5, 0.95]) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    if len(X.shape) == 1:
        qs = np.quantile(X, quantiles)
        mean = np.mean(qs)
        X = X - mean
        scale = 0.5 * np.max(qs) - 0.5 * np.min(qs)
        if scale == 0:
            scale = 1.0
        X = X / scale
        return X, mean, scale, 0
    else:
        qs = np.quantile(X, quantiles, 0)
        mean = np.mean(qs, 0)
        X = X - mean[np.newaxis, :]
        scale = 0.5 * np.max(qs, 0) - 0.5 * np.min(qs, 0)
        mask = np.nonzero(scale)
        if isinstance(mask, tuple):
            mask = mask[-1]
        if len(mask) == 0:
            mask = np.arange(len(scale))
        elif len(mask) != len(scale):
            scale = scale[mask]
            X = X[:, mask]
        X = X / scale[np.newaxis, :]
        return X, mean, scale, mask

def scale_identity(X: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    if len(X.shape) == 1:
        return X, 0.0, 1.0, 0
    else:
        return X, np.zeros(X.shape[1]), np.ones(X.shape[1]), np.arange(X.shape[1])

def unscale(X: np.ndarray, mean: np.ndarray, scale: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if len(X.shape) == 1:
        if isinstance(mean, float):
            return X * scale + mean
        if len(mean) > len(mask):
            X2 = mean.copy()
            X2[mask] += X * scale
            return X2
        return X * scale + mean
    else:
        if len(mean) > len(mask):
            X2 = np.repeat([mean], X.shape[0], axis=0)
            X2[:, mask] += X * scale[np.newaxis, :]
            return X2
        return X * scale[np.newaxis, :] + mean[np.newaxis, :]

def unscale_extend(X: np.ndarray, mean: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if isinstance(mean, float) or len(mean) == len(mask):
        return X
    if len(X.shape) == 1:
        if len(X) > len(mask):
            out = np.zeros(len(mean) + 1)
            out[0] = X[0]
            out[mask + 1] = X[1:]
            return out
        else:
            out = np.zeros(len(mean))
            out[mask] = X
            return out
    else:
        if X.shape[1] > len(mask):
            out = np.zeros((X.shape[0], len(mean) + 1))
            out[:, 0] = X[:, 0]
            out[:, mask + 1] = X[:, 1:]
            return out
        else:
            out = np.zeros((X.shape[0], len(mean)))
            out[:, mask] = X
            return out


def scale_model(alpha: np.ndarray, mean_x: np.ndarray, scale_x: np.ndarray,
        mask: np.ndarray, mean_y: np.ndarray, scale_y: np.ndarray) -> np.ndarray:
    if isinstance(alpha, float):
        alpha = np.array([0.0, alpha])
    elif isinstance(mean_x, float):
        mean_x = (mean_x,)
    elif len(alpha) == len(mean_x):
        alpha = np.concatenate(([0.0], alpha[mask]))
    else:
        alpha = alpha[np.concatenate(([0], mask + 1))]
    alpha[0] = (alpha[0] - mean_y + sum(alpha[1:] * mean_x[mask])) / scale_y
    alpha[1:] = alpha[1:] / scale_y * scale_x
    return alpha

def unscale_model(alpha: np.ndarray, mean_x: np.ndarray, scale_x: np.ndarray,
        mask: np.ndarray, mean_y: np.ndarray, scale_y: np.ndarray) -> np.ndarray:
    if isinstance(mean_x, float):
        mean_x = (mean_x,)
    out = np.zeros(len(mean_x) + 1)
    if isinstance(alpha, float):
        out[mask + 1] = alpha
    elif isinstance(scale_x, float) or len(alpha) > len(scale_x):
        out[0] = alpha[0]
        out[mask + 1] = alpha[1:]
    else:
        out[mask + 1] = alpha
    out[0] = (out[0] - sum(out[mask + 1] * mean_x[mask] / scale_x)) * scale_y + mean_y
    out[mask + 1] = out[mask + 1] / scale_x * scale_y
    return out


def pca_simple(X: np.ndarray, dimensions: int = 10, tolerance: float = 1e-10) -> (np.ndarray, np.ndarray):
    if len(X.shape) == 1:
        return X, 1.0
    dimensions = min(dimensions, *X.shape)
    u, s, v = np.linalg.svd(X, False, True, False)
    dimensions = np.sum(s[:min(dimensions, len(s))] > s[0] * tolerance)
    if dimensions < v.shape[0]:
        v = v[:dimensions, :]
    return u[:, :dimensions].dot(np.diag(s[:dimensions])), v

def pca_rotate(X: np.ndarray, v: np.ndarray) -> np.ndarray:
    return X @ v.T

def pca_invert(X: np.ndarray, v: np.ndarray) -> np.ndarray:
    return X @ v

def pca_rotate_model(alpha: np.ndarray, v: np.ndarray) -> np.ndarray:
    if len(alpha) > v.shape[1]:
        return np.concatenate((alpha[:1], v @ alpha[1:]))
    return v @ alpha

def pca_invert_model(alpha: np.ndarray, v: np.ndarray) -> np.ndarray:
    if len(alpha) > v.shape[0]:
        return np.concatenate((alpha[:1], v.T @ alpha[1:]))
    return v.T @ alpha


def local_scale(X: np.ndarray, x:np.ndarray) -> np.ndarray:
    if len(X.shape) > 1:
        return X - x[np.newaxis, :]
    return X - x

def local_unscale(X: np.ndarray, x:np.ndarray) -> np.ndarray:
    if len(X.shape) > 1:
        return X + x[np.newaxis, :]
    return X + x

def local_unscale_model(alpha: np.ndarray, x:np.ndarray, y:np.ndarray) -> np.ndarray:
    return np.concatenate((y - np.sum(alpha * x, keepdims=True), alpha))


if __name__ =="__main__":
    scale_normal(np.random.normal(size=(20,5)))
