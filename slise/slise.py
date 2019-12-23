# This script contains the main slise functions

from math import log
from warnings import warn
import numpy as np
from scipy.special import logit, expit as sigmoid
from slise.data import add_intercept_column, DataScaler,\
    pca_simple, pca_invert_model, pca_rotate
from slise.optimisation import graduated_optimisation, loss_sharp
from slise.initialisation import initialise_candidates


def slise_raw(X: np.ndarray, Y: np.ndarray, alpha: np.ndarray = None, beta: float = 0.0, 
                epsilon: float = 0.1, lambda1: float = 0, lambda2: float = 0,
                beta_max: float = 25, max_approx: float = 1.15, max_iterations: int = 200,
                pca_treshold: int = 10, inits: int = 500) -> np.ndarray:
    if alpha is None:
        alpha, beta = initialise_candidates(X, Y, x=None, epsilon=epsilon,
            intercept=False, beta_max=beta_max * 0.2, max_approx=max_approx,
            pca_treshold=pca_treshold, inits=inits)
    return graduated_optimisation(alpha, X, Y, epsilon=epsilon, lambda1=lambda1,
            lambda2=lambda2, beta=beta, beta_max=beta_max,  max_approx=max_approx,
            max_iterations=max_iterations)

class SliseWarning(RuntimeWarning):
    pass

class SliseRegression():
    def __init__(self, epsilon: float = 0.1, lambda1: float = 0, lambda2: float = 0,
                intercept: bool = True, logit: bool = False, scale_x = False, scale_y = False,
                beta_max: float = 25, max_approx: float = 1.15, max_iterations: int = 200, 
                pca_treshold: int = 10, inits: int = 500):
        self.epsilon = epsilon
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.scaler = DataScaler(scale_x, scale_y, intercept, logit)
        self.beta_max = beta_max
        self.max_approx = max_approx
        self.max_iterations = max_iterations
        self.pca_treshold = pca_treshold
        self.inits = inits
        self.alpha = 0.0
        self.coefficients = 0.0

    def fit(self, X: np.ndarray, Y: np.ndarray):
        if len(X.shape) == 1:
            X.shape += (1,)
        X, Y = self.scaler.fit(X, Y)
        alpha, beta = initialise_candidates(X, Y, x=None, epsilon=self.epsilon,
            intercept=self.scaler.intercept, beta_max=self.beta_max * 0.2, max_approx=self.max_approx,
            pca_treshold=self.pca_treshold, inits=self.inits)
        self.alpha = graduated_optimisation(alpha, X, Y, epsilon=self.epsilon, lambda1=self.lambda1,
            lambda2=self.lambda2, beta=beta, beta_max=self.beta_max,  max_approx=self.max_approx,
            max_iterations=self.max_iterations)
        self.coefficients = self.scaler.unscale_model(self.alpha)
        if not self.scaler.intercept:
            if np.abs(self.coefficients[0]) > 1e-8:
                warn("Intercept introduced due to scaling (consider setting scale_*=False, or intercept=True)", SliseWarning)
            else:
                self.coefficients = self.coefficients[1:]
        return self

    def get_params(self, scaled=False):
        return self.alpha if scaled else self.coefficients

    def predict(self, X):
        if len(X.shape) == 1:
            X.shape += (1,)
        Y = X @ self.coefficients[1:] + self.coefficients[0]
        if self.scaler.logit:
            Y = sigmoid(Y)
        return Y

    def score(self, X, Y):
        if len(X.shape) == 1:
            X.shape += (1,)
        X, Y = self.scaler.scale(X, Y)
        return loss_sharp(self.alpha, X, Y, self.epsilon, self.lambda1, self.lambda2)

    def set_params(self, params):
        self.alpha = params

    def print(self, column_names: list = None, decimals: int = 3):
        alpha = np.atleast_1d(self.scaler.extend_model(self.alpha))
        coeff = np.atleast_1d(self.coefficients)
        if len(alpha) < len(coeff):
            alpha = np.concatenate(([0.0], alpha))
        if column_names is None:
            column_names = ["Col %d"%i for i in range(len(np.atleast_1d(self.scaler.scaler_x.mean)))]
        if len(column_names) < len(coeff):
            column_names = ["Intercept"] + column_names
        alpha = ["%%.%df"%decimals%a for a in alpha]
        coeff = ["%%.%df"%decimals%a for a in coeff]
        col_len = max(8,
            np.max([len(s) for s in column_names]),
            np.max([len(a) for a in alpha]),
            np.max([len(a) for a in coeff])) + 1
        assert len(alpha) == len(coeff)
        assert len(alpha) == len(column_names)
        print("Variables:    ", end="")
        for s in column_names:
            print(" %%%ds"%col_len%s, end="")
        print("\nCoefficients: ", end="")
        for a in coeff:
            print(" %%%ds"%col_len%a, end="")
        print("\nScaled Alpha: ", end="")
        for a in alpha:
            print(" %%%ds"%col_len%a, end="")

#TODO SliseExplanation