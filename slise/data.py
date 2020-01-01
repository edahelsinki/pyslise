# This script contains functions for modifying data

from abc import ABC, abstractmethod
import numpy as np
from scipy.special import logit, expit as sigmoid


def add_intercept_column(X: np.ndarray) -> np.ndarray:
    """
        Add a constant (first) column of ones to the matrix
    """
    if len(X.shape) == 1:
        return np.concatenate(([1.0], X))
    return np.concatenate((np.ones((X.shape[0], 1)), X), 1)

def remove_intercept_column(X: np.ndarray) -> np.ndarray:
    """
        Remove the first column
    """
    if len(X.shape) == 1:
        return X[1:]
    return X[:, 1:]

def mat_mul_with_intercept(X: np.ndarray, alpha: np.ndarray):
    """
        Matrix multiplication, but check and handle potential intercepts in alpha
    """
    alpha = np.atleast_1d(alpha)
    if len(X.shape) == 1:
        X.shape += (1,)
    if len(alpha) == X.shape[1] + 1:
        return X @ alpha[1:] + alpha[0]
    else:
        return X @ alpha


class AScaler(ABC):
    """
        Abstract base class for all built-in scalers
    """

    def __init__(self):
        """
            All scalers are of the type X' = (X - mean)[mask] / stddv
        """
        self.mean = 0.0
        self.stddv = 1.0
        self.mask = 0

    @abstractmethod
    def fit(self, X: np.ndarray) -> np.ndarray:
        """
            Fit the parameters, and return the scaled np.ndarray
        """
        if len(X.shape) == 1:
            self.mean = 0.0
            self.stddv = 1.0
            self.mask = 0
        else:
            self.mean = np.zeros(X.shape[1])
            self.stddv = np.ones(X.shape[1])
            self.mask = np.arange(X.shape[1])
        return X

    def scale(self, X:np.ndarray) -> np.ndarray:
        """
           Scale X using the fitted parameters
        """
        if isinstance(self.mean, float) or len(self.mean) == 1:
            return (X - self.mean) / self.stddv
        elif len(X.shape) > 1:
            return (X[:, self.mask] - self.mean[np.newaxis, self.mask]) / self.stddv[np.newaxis, :]
        elif len(self.mean) == len(X):
            return (X - self.mean)[self.mask] / self.stddv
        else:
            raise Exception("Wrong dimensions of X (this method is for additional scaling, use 'fit' for finding the scaling parameters)")

    def unscale(self, X: np.ndarray) -> np.ndarray:
        """
           Unscale X using the fitted parameters
        """
        if isinstance(self.mean, float) or len(self.mean) == 1:
            return X * self.stddv + self.mean
        elif len(X.shape) > 1:
            if len(self.mean) > len(self.mask):
                X2 = np.repeat([self.mean], X.shape[0], axis=0)
                X2[:, self.mask] = X2[:, self.mask] + X * self.stddv[np.newaxis, :]
                return X2
            return X * self.stddv[np.newaxis, :] + self.mean[np.newaxis, :]
        elif len(self.stddv) == len(X):
            if len(self.mean) > len(self.mask):
                X2 = self.mean.copy()
                X2[self.mask] = X2[self.mask] + X * self.stddv
                return X2
            return X * self.stddv + self.mean
        else:
            raise Exception("Wrong dimensions of X (use fit first)")


class ScalerNormal(AScaler):
    """
        A "normal" scaler that subtracts the (columnwise) mean and divides by the (columnwise) deviation.
        Additionally it removes all constant columns.
    """

    def fit(self, X: np.ndarray) -> np.ndarray:
        """
            Fit the parameters, and return the scaled np.ndarray
        """
        if len(X.shape) == 1:
            self.mean = np.mean(X)
            X = X - self.mean
            self.stddv = np.sqrt(np.sum(np.square(X)) / (len(X) - 1))
            if self.stddv == 0:
                self.stddv = 1.0
            X = X / self.stddv
            self.mask = 0
        else:
            self.mean = np.mean(X, 0)
            X = X - self.mean[np.newaxis, :]
            self.stddv = np.sqrt(np.sum(np.square(X), 0) / (X.shape[0] - 1))
            self.mask = np.nonzero(self.stddv)
            if isinstance(self.mask, tuple):
                self.mask = self.mask[-1]
            if len(self.mask) == 0:
                self.mask = np.arange(len(self.stddv))
            elif len(self.mask) != len(self.stddv):
                self.stddv = self.stddv[self.mask]
                X = X[:, self.mask]
            X = X / self.stddv[np.newaxis, :]
        return X


class ScalerRange(AScaler):
    """
        A scaler that scales the columns to have spread of one:
            min(quantile) - max(quantile) = 1.0
    """

    def __init__(self, quantiles: list = [0.05, 0.95]):
        """A scaler that scales the columns to have spread of one:
        
        Keyword Arguments:
            quantiles {list} -- The quantiles for calculating the spread (default: {[0.05, 0.95]})
        """
        super().__init__()
        self.quantiles = quantiles

    def fit(self, X: np.ndarray) -> np.ndarray:
        """
            Fit the parameters, and return the scaled np.ndarray
        """
        if len(X.shape) == 1:
            qs = np.quantile(X, self.quantiles)
            self.mean = 0.0
            self.stddv = 0.5 * np.max(qs) - 0.5 * np.min(qs)
            if self.stddv == 0:
                self.stddv = 1.0
            X = X / self.stddv
            self.mask = 0
            return X
        else:
            qs = np.quantile(X, self.quantiles, 0)
            self.mean = np.zeros(X.shape[1])
            self.stddv = 0.5 * np.max(qs, 0) - 0.5 * np.min(qs, 0)
            self.mask = np.nonzero(self.stddv)
            if isinstance(self.mask, tuple):
                self.mask = self.mask[-1]
            if len(self.mask) == 0:
                self.mask = np.arange(len(self.stddv))
            elif len(self.mask) != len(self.stddv):
                self.stddv = self.stddv[self.mask]
                self.mean = np.mean(qs, 0)
                self.mean[self.mask] = 0.0
                X = X[:, self.mask]
            X = X / self.stddv[np.newaxis, :]
            return X

class ScalerIdentity(AScaler):
    """
        A scaler that does nothing
    """

    def fit(self, X: np.ndarray) -> np.ndarray:
        """
            Fit the parameters, and return the scaled np.ndarray
        """
        if len(X.shape) == 1:
            self.mean = 0.0
            self.stddv = 1.0
            self.mask = 0
        else:
            self.mean = np.zeros(X.shape[1])
            self.stddv = np.ones(X.shape[1])
            self.mask = np.arange(X.shape[1])
        return X

    def scale(self, X):
        return X

    def unscale(self, X):
        return X


class ScalerRemoveConstant(AScaler):
    """
        A scaler that only removes constant columns
    """

    def fit(self, X: np.ndarray) -> np.ndarray:
        """
            Fit the parameters, and return the scaled np.ndarray
        """
        if len(X.shape) == 1:
            self.mean = 0.0
            self.stddv = 1.0
            self.mask = 0
        else:
            self.mean = np.mean(X, 0)
            self.stddv = np.ones(X.shape[1])
            var = np.sum(np.square(X - self.mean[np.newaxis, :]), 0)
            self.mask = np.nonzero(var)
            if isinstance(self.mask, tuple):
                self.mask = self.mask[-1]
            if len(self.mask) == 0:
                self.mask = np.arange(len(self.stddv))
            elif len(self.mask) != len(self.stddv):
                self.stddv = self.stddv[self.mask]
                X = X[:, self.mask]
            self.mean[self.mask] = 0.0
        return X


def local_into(X: np.ndarray, x:np.ndarray) -> np.ndarray:
    """Center a np.ndarray on a selected point
    
    Arguments:
        X {np.ndarray} -- the np.ndarray
        x {np.ndarray} -- the point
    
    Returns:
        np.ndarray -- the centered np.ndarray
    """
    if len(X.shape) > 1:
        return X - x[np.newaxis, :]
    return X - x

def local_from(X: np.ndarray, x:np.ndarray) -> np.ndarray:
    """Uncenter a np.ndarray on a selected point
    
    Arguments:
        X {np.ndarray} -- the np.ndarray
        x {np.ndarray} -- the point
    
    Returns:
        np.ndarray -- the centered np.ndarray
    """
    if len(X.shape) > 1:
        return X + x[np.newaxis, :]
    return X + x

def local_model(alpha: np.ndarray, x:np.ndarray, y:np.ndarray) -> np.ndarray:
    """Make sure a linear model passes through a point
    
    Arguments:
        alpha {np.ndarray} -- the linear model
        x {np.ndarray} -- the point
        y {np.ndarray} -- the response
    
    Returns:
        np.ndarray -- the localised model
    """
    if len(alpha) == len(x) + 1:
        alpha = alpha.copy()
        alpha[0] = y - np.sum(alpha[1:] * x)
        return alpha
    return np.concatenate((y - np.sum(alpha * x, keepdims=True), alpha))

class ScalerLocal(AScaler):
    """
        A scaler that centers the data on a selected point
    """

    def __init__(self, x):
        """A scaler that centers the data on a selected point
        
        Arguments:
            x -- the point to center on
        """
        super().__init__()
        self.mean = x
        self.stddv = np.ones_like(x)
        self.mask = np.arange(len(x))

    def fit(self, X):
        return local_into(X, self.mean)

    def scale(self, X):
        return local_into(X, self.mean)

    def unscale(self, X):
        return local_from(X, self.mean)


class ScalerNested(AScaler):
    """
        A scaler that combines to other AScalers
    """

    def __init__(self, outer: AScaler, inner: AScaler):
        """A scaler that combines to other AScalers
        
        Arguments:
            outer {AScaler} -- The scaler that operates on the data
            inner {AScaler} -- The scaler that operates on the ouput of the outer scaler
        """
        super().__init__()
        self.inner = inner
        self.outer = outer

    def fit(self, X):
        """
            Fit the parameters, and return the scaled np.ndarray
        """
        X1 = self.outer.fit(X)
        X2 = self.inner.fit(X1)
        self.mask = np.atleast_1d(self.outer.mask)[self.inner.mask]
        self.stddv = np.atleast_1d(self.outer.stddv)[self.inner.mask] * self.inner.stddv
        self.mean = np.atleast_1d(self.outer.mean)
        self.mean[self.outer.mask] = self.mean[self.outer.mask] + self.outer.stddv * self.inner.mean
        return X2


class DataScaler():
    """
        This class holds two scalers, one for X and one for Y
    """

    def __init__(self, scaler_x: AScaler = False, scaler_y: AScaler = False, intercept: bool = False, logit: bool = False):
        """This class holds two scalers, one for X and one for Y
        
        Keyword Arguments:
            scaler_x {AScaler} -- The scaler to use for X, or True for ScalerNormal and False for ScalerRemoveConstant (default: {False})
            scaler_y {AScaler} -- The scaler to use for Y, or True for ScalerRange and False for ScalerIdentity (default: {False})
            intercept {bool} -- Should an intercept column be added to X (default: {False})
            logit {bool} -- Should Y be passed through a logit function (default: {False})
        """
        if scaler_x == True:
            self.scaler_x = ScalerNormal()
        elif scaler_x:
            self.scaler_x = scaler_x
        else:
            self.scaler_x = ScalerRemoveConstant()
        if scaler_y == True:
            self.scaler_y = ScalerRange()
        elif scaler_y:
            self.scaler_y = scaler_y
        else:
            self.scaler_y = ScalerIdentity()
        self.logit = logit
        self.intercept = intercept

    def fit(self, X: np.ndarray, Y: np.ndarray) -> (np.ndarray, np.ndarray):
        """
            Fit the scalers for X and Y
        """
        if X is not None:
            X = self.scaler_x.fit(X)
            if self.intercept:
                X = add_intercept_column(X)
        if Y is not None:
            if self.logit:
                Y = logit(Y)
            Y = self.scaler_y.fit(Y)
        return X, Y

    def scale(self, X: np.ndarray, Y: np.ndarray) -> (np.ndarray, np.ndarray):
        """
            Scale X and Y
        """
        if X is not None:
            X = self.scaler_x.scale(X)
            if self.intercept:
                X = add_intercept_column(X)
        if Y is not None:
            if self.logit:
                Y = logit(Y)
            Y = self.scaler_y.scale(Y)
        return X, Y

    def unscale(self, X: np.ndarray, Y: np.ndarray) -> (np.ndarray, np.ndarray):
        """
            Unscale X and Y
        """
        if X is not None:
            if self.intercept:
                X = remove_intercept_column(X)
            X = self.scaler_x.unscale(X)
        if Y is not None:
            Y = self.scaler_y.unscale(Y)
            if self.logit:
                Y = sigmoid(Y)
        return X, Y

    def scale_model(self, alpha: np.ndarray) -> np.ndarray:
        """
            Scale a linear model
        """
        if isinstance(alpha, float):
            alpha = np.array([0.0, alpha])
        elif len(alpha) == len(self.scaler_x.mean):
            alpha = np.concatenate(([0.0], alpha[self.scaler_x.mask]))
        elif len(alpha) == len(self.scaler_x.mean) + 1: # Assuming intercept
            alpha = alpha[np.concatenate(([0], np.atleast_1d(self.scaler_x.mask + 1)))]
        else:
            raise Exception("wrong size for alpha")
        alpha[0] = (alpha[0] - self.scaler_y.mean + sum(alpha[1:] * self.scaler_x.mean[self.scaler_x.mask])) / self.scaler_y.stddv
        alpha[1:] = alpha[1:] * self.scaler_x.stddv / self.scaler_y.stddv
        return alpha

    def unscale_model(self, alpha: np.ndarray) -> np.ndarray:
        """
            Unscale a linear model
        """
        if len(np.atleast_1d(alpha)) == len(np.atleast_1d(self.scaler_x.mask)):
            alpha = np.concatenate(([0], alpha))
        else:
            alpha = alpha.copy()
        alpha[0] = (alpha[0] - np.sum(alpha[1:] * self.scaler_x.mean[self.scaler_x.mask] / self.scaler_x.stddv)) * self.scaler_y.stddv + self.scaler_y.mean
        alpha[1:] = alpha[1:] * self.scaler_y.stddv / self.scaler_x.stddv
        return self.extend_model(alpha)

    def extend_model(self, alpha: np.ndarray) -> np.ndarray:
        """
            If constant columns where removed, readd them to a linear model, but don't unscale
        """
        mask = np.atleast_1d(self.scaler_x.mask)
        mean = np.atleast_1d(self.scaler_x.mean)
        if len(mask) == len(mean):
            return alpha
        a2 = np.atleast_1d(alpha)
        if len(a2) == len(mask):
            a2 = np.zeros_like(mean)
            a2[mask] = alpha
            return a2
        if len(a2) == len(mask) + 1:
            a2 = np.zeros(len(mean) + 1)
            a2[np.concatenate(([0], mask + 1))] = alpha
            return a2
        return alpha


def pca_simple(X: np.ndarray, dimensions: int = 10, tolerance: float = 1e-10) -> (np.ndarray, np.ndarray):
    """Fit and use PCA for dimensionality reduction
    
    Arguments:
        X {np.ndarray} -- The matrix to reduce
    
    Keyword Arguments:
        dimensions {int} -- the number of output dimensions (default: {10})
        tolerance {float} -- relative treshold for accepted dimensions (default: {1e-10})
    
    Returns:
        (np.ndarray, np.ndarray): -- The reduced matrix, and the PCA-rotation
    """
    if len(X.shape) == 1:
        return X, 1.0
    dimensions = min(dimensions, *X.shape)
    u, s, v = np.linalg.svd(X, False, True, False)
    dimensions = np.sum(s[:min(dimensions, len(s))] > s[0] * tolerance)
    if dimensions < v.shape[0]:
        v = v[:dimensions, :]
    return u[:, :dimensions].dot(np.diag(s[:dimensions])), v

def pca_rotate(X: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Use PCA for dimensionality reduction
    
    Arguments:
        X {np.ndarray} -- the np.ndarray
        v {np.ndarray} -- the PCA-rotation
    
    Returns:
        np.ndarray -- the reduced X
    """
    return X @ v.T

def pca_invert(X: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Invert PCA for dimensionality expansion
    
    Arguments:
        X {np.ndarray} -- the np.ndarray
        v {np.ndarray} -- the PCA-rotation
    
    Returns:
        np.ndarray -- the expanded X
    """
    return X @ v

def pca_rotate_model(alpha: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Use PCA for dimensionality reduction
    
    Arguments:
        alpha {np.ndarray} -- the linear model to rotate
        v {np.ndarray} -- the PCA-rotation
    
    Returns:
        np.ndarray -- the rotated alpha
    """
    if len(alpha) > v.shape[1]:
        return np.concatenate((alpha[:1], v @ alpha[1:]))
    return v @ alpha

def pca_invert_model(alpha: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Invert PCA for dimensionality expansion
    
    Arguments:
        alpha {np.ndarray} -- the linear model in reduced space
        v {np.ndarray} -- the PCA-rotation
    
    Returns:
        np.ndarray -- the expanded linear model
    """
    if len(alpha) > v.shape[0]:
        return np.concatenate((alpha[:1], v.T @ alpha[1:]))
    return v.T @ alpha
