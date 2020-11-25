"""
    This script contains the main slise functions, and classes
"""

from __future__ import annotations
from warnings import warn
import numpy as np
from scipy.special import expit as sigmoid
from slise.data import DataScaler, local_into, local_model, mat_mul_with_intercept
from slise.optimisation import graduated_optimisation, loss_sharp
from slise.initialisation import initialise_candidates
from slise.utils import SliseWarning
from slise.plot import (
    plot_regression_2D,
    fill_column_names,
    fill_prediction_str,
    plot_explanation_tabular,
    plot_explanation_dist,
    plot_explanation_image,
)


def slise_raw(
    X: np.ndarray,
    Y: np.ndarray,
    alpha: np.ndarray = None,
    beta: float = 0.0,
    epsilon: float = 0.1,
    lambda1: float = 0,
    lambda2: float = 0,
    beta_max: float = 25,
    max_approx: float = 1.15,
    max_iterations: int = 200,
    pca_treshold: int = 10,
    inits: int = 500,
    debug: bool = False,
) -> np.ndarray:
    """Use SLISE "raw" without any preprocessing

    Arguments:
        X {np.ndarray} -- the data matrix
        Y {np.ndarray} -- the response vector

    Keyword Arguments:
        alpha {np.ndarray} -- the starting linear model (None means using initialisation) (default: {None})
        beta {float} -- the starting sigmoid steepness (default: {0.0})
        epsilon {float} -- the error tolerance (default: {0.1})
        lambda1 {float} -- the L1 regularistaion strength (default: {0})
        lambda2 {float} -- the L2 regularisation strength (default: {0})
        beta_max {float} -- the stopping sigmoid steepness (default: {25})
        max_approx {float} -- the target approximation ratio for the graduated optimisation (default: {1.15})
        max_iterations {int} -- the maximum iterations of OWL-QN per graduated optimisation step (default: {200})
        pca_treshold {int} -- the treshold for using pca in the initialisation (default: {10})
        inits {int} -- the number of candidates to generate in the initialisation (default: {500})
        debug {bool} -- print debug statements each graduated optimisation step (default: {False})

    Returns:
        np.ndarray -- the linear model coefficients
    """
    if alpha is None:
        alpha, beta = initialise_candidates(
            X,
            Y,
            x=None,
            epsilon=epsilon,
            intercept=False,
            beta_max=beta_max * 0.2,
            max_approx=max_approx,
            pca_treshold=pca_treshold,
            inits=inits,
        )
    return graduated_optimisation(
        alpha,
        X,
        Y,
        epsilon=epsilon,
        lambda1=lambda1,
        lambda2=lambda2,
        beta=beta,
        beta_max=beta_max,
        max_approx=max_approx,
        max_iterations=max_iterations,
        debug=debug,
    )


def regression(X: np.ndarray, Y: np.ndarray, **kwargs) -> SliseRegression:
    """Use SLISE for robust regression

    Arguments:
        X {np.ndarray} -- The data matrix
        Y {np.ndarray} -- The response vector

    Keyword Arguments:
        epsilon {float} -- the error tolerance (default: {0.1})
        lambda1 {float} -- the L1 regularistaion strength (default: {0})
        lambda2 {float} -- the L2 regularisation strength (default: {0})
        intercept {bool} -- add an intercept term (default: {True})
        logit {bool} -- do a logit transformation on the Y vector, this is recommended if Y is probabilities (default: {False})
        scale_x {bool or slise.data.AScaler} -- should the X matrix be scaled by subtracting the mean and dividing by the standard deviation, or specific scaler to use (default: {False})
        scale_y {bool or slise.data.AScaler} -- should the Y vector be scaled to have a range of one, or specific scaler to use (default: {False})
        beta_max {float} -- the stopping sigmoid steepness (default: {25})
        max_approx {float} -- the target approximation ratio for the graduated optimisation (default: {1.15})
        max_iterations {int} -- the maximum iterations of OWL-QN per graduated optimisation step (default: {200})
        pca_treshold {int} -- the treshold for using pca in the initialisation (default: {10})
        inits {int} -- the number of candidates to generate in the initialisation (default: {500})
        debug {bool} -- print debug statements each graduated optimisation step (default: {False})

    Returns:
        SliseRegression -- object containing the regression result
    """
    return SliseRegression(**kwargs).fit(X, Y)


def explain(
    X: np.ndarray, Y: np.ndarray, x: np.ndarray, y: float = None, **kwargs
) -> SliseExplainer:
    """Use SLISE for explaining predictions from black box models

    Arguments:
        X {np.ndarray} -- the data matrix
        Y {np.ndarray} -- the vector of predictions
        x {np.ndarray} -- the data item to explain, or the row/index from X
        y {float} -- the prediction to explain, or None if x is an index (default: {None})

    Keyword Arguments:
        epsilon {float} -- the error tolerance (default: {0.1})
        lambda1 {float} -- the L1 regularistaion strength (default: {0})
        lambda2 {float} -- the L2 regularisation strength (default: {0})
        logit {bool} -- do a logit transformation on the Y vector, this is recommended if Y is probabilities (default: {False})
        scale_x {bool or slise.data.AScaler} -- should the X matrix be scaled by subtracting the mean and dividing by the standard deviation, or specific scaler to use (default: {False})
        scale_y {bool or slise.data.AScaler} -- should the Y vector be scaled to have a range of one, or specific scaler to use (default: {False})
        beta_max {float} -- the stopping sigmoid steepness (default: {25})
        max_approx {float} -- the target approximation ratio for the graduated optimisation (default: {1.15})
        max_iterations {int} -- the maximum iterations of OWL-QN per graduated optimisation step (default: {200})
        pca_treshold {int} -- the treshold for using pca in the initialisation (default: {10})
        inits {int} -- the number of candidates to generate in the initialisation (default: {500})
        debug {bool} -- print debug statements each graduated optimisation step (default: {False})

    Returns:
        SliseExplainer -- object containing the explanation result
    """
    return SliseExplainer(X, Y, **kwargs).explain(x, y)


class SliseRegression:
    """
        Class for holding the result from using SLISE for regression.
        Can also be used sklearn-style to do regression.
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        lambda1: float = 0,
        lambda2: float = 0,
        intercept: bool = True,
        logit: bool = False,
        scale_x=False,
        scale_y=False,
        beta_max: float = 25,
        max_approx: float = 1.15,
        max_iterations: int = 200,
        pca_treshold: int = 10,
        inits: int = 500,
        debug: bool = False,
    ):
        """Use SLISE for robust regression

        Keyword Arguments:
            alpha {np.ndarray} -- the starting linear model (None means using initialisation) (default: {None})
            beta {float} -- the starting sigmoid steepness (default: {0.0})
            epsilon {float} -- the error tolerance (default: {0.1})
            lambda1 {float} -- the L1 regularistaion strength (default: {0})
            lambda2 {float} -- the L2 regularisation strength (default: {0})
            intercept {bool} -- add an intercept term (default: {True})
            logit {bool} -- do a logit transformation on the Y vector (default: {False})
            scale_x {bool or slise.data.AScaler} -- should the X matrix be scaled by subtracting the mean and dividing by the standard deviation, or specific scaler to use (default: {False})
            scale_y {bool or slise.data.AScaler} -- should the Y vector be scaled to have a range of one, or specific scaler to use (default: {False})
            beta_max {float} -- the stopping sigmoid steepness (default: {25})
            max_approx {float} -- the target approximation ratio for the graduated optimisation (default: {1.15})
            max_iterations {int} -- the maximum iterations of OWL-QN per graduated optimisation step (default: {200})
            pca_treshold {int} -- the treshold for using pca in the initialisation (default: {10})
            inits {int} -- the number of candidates to generate in the initialisation (default: {500})
            debug {bool} -- print debug statements each graduated optimisation step (default: {False})
        """
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
        self.debug = debug
        self.X = np.array([[0.0]])
        self.Y = np.array([0])

    def fit(self, X: np.ndarray, Y: np.ndarray) -> SliseRegression:
        """Fit a robust regression

        Arguments:
            X {np.ndarray} -- The data matrix
            Y {np.ndarray} -- The response vector

        Returns:
            SliseRegression -- self, containing the regression result
        """
        if len(X.shape) == 1:
            X.shape += (1,)
        X, Y = self.scaler.fit(X, Y)
        self.X = X
        self.Y = Y
        alpha, beta = initialise_candidates(
            X,
            Y,
            x=None,
            epsilon=self.epsilon,
            intercept=self.scaler.intercept,
            beta_max=self.beta_max * 0.2,
            max_approx=self.max_approx,
            pca_treshold=self.pca_treshold,
            inits=self.inits,
        )
        self.alpha = graduated_optimisation(
            alpha,
            X,
            Y,
            epsilon=self.epsilon,
            lambda1=self.lambda1,
            lambda2=self.lambda2,
            beta=beta,
            beta_max=self.beta_max,
            max_approx=self.max_approx,
            max_iterations=self.max_iterations,
            debug=self.debug,
        )
        self.coefficients = self.scaler.unscale_model(self.alpha)
        if not self.scaler.intercept:
            if np.abs(self.coefficients[0]) > 1e-8:
                warn(
                    "Intercept introduced due to scaling (consider setting scale_*=False, or intercept=True)",
                    SliseWarning,
                )
            else:
                self.coefficients = self.coefficients[1:]
        return self

    def get_params(self, scaled: bool = False) -> np.ndarray:
        """Get the coefficients of the linear model

        Keyword Arguments:
            scaled {bool} -- return the model that fits the scaled data (if the data is being scaled within SLISE) (default: {False})

        Returns:
            np.ndarray -- the coefficients of the linear model
        """
        return self.alpha if scaled else self.coefficients

    def predict(self, X: np.ndarray = None) -> np.ndarray:
        """Use the fitted model to predict new responses

        Keyword Arguments:
            X {np.ndarray} -- the data to predict, or None for using the give dataset (default: {None})

        Returns:
            np.ndarray -- the response
        """
        if X is None:
            Y = mat_mul_with_intercept(self.X, self.alpha)
            Y = self.scaler.scaler_y.unscale(Y)
        else:
            Y = mat_mul_with_intercept(X, self.coefficients)
        if self.scaler.logit:
            Y = sigmoid(Y)
        return Y

    def score(self, X: np.ndarray = None, Y: np.ndarray = None) -> float:
        """Calculate the loss

        Keyword Arguments:
            X {np.ndarray} -- the data, None for the given dataset (default: {None})
            Y {np.ndarray} -- the response, None for the given dataset (default: {None})

        Returns:
            float -- the loss
        """
        if X is None or Y is None:
            X = self.X
            Y = self.Y
        else:
            X, Y = self.scaler.scale(X, Y)
        return loss_sharp(self.alpha, X, Y, self.epsilon, self.lambda1, self.lambda2)

    def subset(self, X: np.ndarray = None, Y: np.ndarray = None) -> np.ndarray:
        """Get the subset as a boolean mask

        Keyword Arguments:
            X {np.ndarray} -- the data, None for the given dataset (default: {None})
            Y {np.ndarray} -- the response, None for the given dataset (default: {None})

        Returns:
            np.ndarray -- the subset as a boolean mask
        """
        if X is None or Y is None:
            X = self.X
            Y = self.Y
        else:
            X, Y = self.scaler.scale(X, Y)
        Y = mat_mul_with_intercept(X, self.alpha) - Y
        return Y ** 2 < self.epsilon ** 2

    def set_params(self, alpha: np.ndarray) -> SliseRegression:
        """Set the coefficient for the current linear model

        Arguments:
            alpha {np.ndarray} -- the coefficients

        Returns:
            SliseRegression -- self, with the new linear model
        """
        self.alpha = alpha
        return self

    def print(self, column_names: list = None, decimals: int = 3) -> SliseRegression:
        """Print the current robust regression result

        Keyword Arguments:
            column_names {list} -- names of the variables/columns in X (default: {None})
            decimals {int} -- the precision to use for printing (default: {3})

        Returns:
            SliseRegression -- self
        """
        alpha = np.atleast_1d(self.scaler.extend_model(self.alpha))
        coeff = np.atleast_1d(self.coefficients)
        if len(alpha) < len(coeff):
            alpha = np.concatenate(([0.0], alpha))
        column_names = fill_column_names(
            column_names,
            len(coeff),
            len(np.atleast_1d(self.scaler.scaler_x.mean)) < len(coeff),
        )
        alpha = ["%%.%df" % decimals % a for a in alpha]
        coeff = ["%%.%df" % decimals % a for a in coeff]
        col_len = max(
            8,
            np.max([len(s) for s in column_names]),
            np.max([len(a) for a in alpha]),
            np.max([len(a) for a in coeff]),
        )
        assert len(alpha) == len(coeff)
        assert len(alpha) == len(column_names)
        print("Variables:   ", " ".join([f"{s:>{col_len}}" for s in column_names]))
        print("Coefficients:", " ".join([f"{s:>{col_len}}" for s in coeff]))
        print("Scaled Alpha:", " ".join([f"{s:>{col_len}}" for s in alpha]))
        print(f"Loss:         {self.score():>{col_len}.{decimals}f}")
        print(f"Subset:       {self.subset().mean():>{col_len}.{decimals}f}")
        return self

    def plot(
        self, label_x: str = "x", label_y: str = "y", decimals: int = 3
    ) -> SliseRegression:
        """Plot 1D data in a 2D scatter plot, with a line for the regression model

        Keyword Arguments:
            label_x {str} -- the name of the dependent value (default: "x")
            label_y {str} -- the name of the predicted value (default: "y")
            decimals {int} -- the number of decimals for the axes (default: {2})

        Raises:
            Exception: if the data is not 1D (intercept allowed)

        Returns:
            SliseRegression -- self
        """
        plot_regression_2D(
            self.X,
            self.Y,
            self.alpha,
            self.epsilon,
            self.scaler,
            label_x,
            label_y,
            decimals,
        )
        return self


class SliseExplainer:
    """
        Class for holding the result from using SLISE as an explainer.
        Can also be used sklearn-style to create explanations.
    """

    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        epsilon: float = 0.1,
        lambda1: float = 0,
        lambda2: float = 0,
        logit: bool = False,
        scale_x=False,
        scale_y=False,
        beta_max: float = 25,
        max_approx: float = 1.15,
        max_iterations: int = 200,
        pca_treshold: int = 10,
        inits: int = 500,
        debug: bool = False,
    ):
        """Use SLISE for explaining outcomes from black box models

        Arguments:
            X {np.ndarray} -- the data matrix
            Y {np.ndarray} -- the vector of predictions

        Keyword Arguments:
            epsilon {float} -- the error tolerance (default: {0.1})
            lambda1 {float} -- the L1 regularistaion strength (default: {0})
            lambda2 {float} -- the L2 regularisation strength (default: {0})
            logit {bool} -- do a logit transformation on the Y vector, this is recommended if Y is probabilities (default: {False})
            scale_x {bool or slise.data.AScaler} -- should the X matrix be scaled by subtracting the mean and dividing by the standard deviation, or specific scaler to use (default: {False})
            scale_y {bool or slise.data.AScaler} -- should the Y vector be scaled to have a range of one, or specific scaler to use (default: {False})
            beta_max {float} -- the stopping sigmoid steepness (default: {25})
            max_approx {float} -- the target approximation ratio for the graduated optimisation (default: {1.15})
            max_iterations {int} -- the maximum iterations of OWL-QN per graduated optimisation step (default: {200})
            pca_treshold {int} -- the treshold for using pca in the initialisation (default: {10})
            inits {int} -- the number of candidates to generate in the initialisation (default: {500})
            debug {bool} -- print debug statements each graduated optimisation step (default: {False})
        """
        self.epsilon = epsilon
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.scaler = DataScaler(scale_x, scale_y, False, logit)
        self.beta_max = beta_max
        self.max_approx = max_approx
        self.max_iterations = max_iterations
        self.pca_treshold = pca_treshold
        self.inits = inits
        self.alpha = 0.0
        self.coefficients = 0.0
        self.debug = debug
        if len(X.shape) == 1:
            X.shape += (1,)
        X, Y = self.scaler.fit(X, Y)
        self.X = X
        self.Y = Y
        self.x = X[0, :] * 0
        self.y = 0

    def explain(self, x: np.ndarray, y: float = None) -> SliseExplainer:
        """Explain an outcome from a black box model

        Arguments:
            x {np.ndarray} -- the data item to explain, or an index from the dataset X

        Keyword Arguments:
            y {float} -- the prediction from the black box model, or None if x is an index (default: {None})

        Returns:
            SliseExplainer -- self, with values set to the explanation
        """
        if y is None:
            self.y = self.Y[x]
            self.x = self.X[x, :]
        else:
            self.x, self.y = self.scaler.scale(x, y)
        X = local_into(self.X, self.x)
        Y = local_into(self.Y, self.y)
        alpha, beta = initialise_candidates(
            X,
            Y,
            x=self.x,
            epsilon=self.epsilon,
            intercept=self.scaler.intercept,
            beta_max=self.beta_max * 0.2,
            max_approx=self.max_approx,
            pca_treshold=self.pca_treshold,
            inits=self.inits,
        )
        alpha = graduated_optimisation(
            alpha,
            X,
            Y,
            epsilon=self.epsilon,
            lambda1=self.lambda1,
            lambda2=self.lambda2,
            beta=beta,
            beta_max=self.beta_max,
            max_approx=self.max_approx,
            max_iterations=self.max_iterations,
            debug=self.debug,
        )
        self.alpha = local_model(alpha, self.x, self.y)
        self.coefficients = self.scaler.unscale_model(self.alpha)
        self.coefficients = local_model(
            self.coefficients, *self.scaler.unscale(self.x, self.y)
        )
        return self

    def get_params(self, scaled: bool = False) -> np.ndarray:
        """Get the approximating linear model

        Keyword Arguments:
            scaled {bool} -- get the scaled coefficients, if the dataset is being scaled within SLISE (default: {False})

        Returns:
            np.ndarray -- the linear model coefficients (first one is the intercept)
        """
        return self.alpha if scaled else self.coefficients

    def predict(self, X: np.ndarray = None) -> np.ndarray:
        """Use the approximating linear model to predict new outcomes

        Keyword Arguments:
            X {np.ndarray} -- the data to predict, None for the given dataset (default: {None})

        Returns:
            np.ndarray -- the (approximated) predictions
        """
        if X is None:
            Y = mat_mul_with_intercept(self.X, self.alpha)
            Y = self.scaler.scaler_y.unscale(Y)
        else:
            Y = mat_mul_with_intercept(X, self.coefficients)
        if self.scaler.logit:
            Y = sigmoid(Y)
        return Y

    def score(self, X: np.ndarray = None, Y: np.ndarray = None) -> float:
        """Calculate the loss

        Keyword Arguments:
            X {np.ndarray} -- the data, None for the given dataset (default: {None})
            Y {np.ndarray} -- the response, None for the given dataset (default: {None})

        Returns:
            float -- the loss
        """
        if X is None or Y is None:
            X = self.X
            Y = self.Y
        else:
            X, Y = self.scaler.scale(X, Y)
        X = local_into(self.X, self.x)
        Y = local_into(self.Y, self.y)
        return loss_sharp(
            self.alpha[1:], X, Y, self.epsilon, self.lambda1, self.lambda2
        )

    def subset(self, X: np.ndarray = None, Y: np.ndarray = None) -> np.ndarray:
        """Get the subset as a boolean mask

        Keyword Arguments:
            X {np.ndarray} -- the data, None for the given dataset (default: {None})
            Y {np.ndarray} -- the response, None for the given dataset (default: {None})

        Returns:
            np.ndarray -- the subset as a boolean mask
        """
        if X is None or Y is None:
            X = self.X
            Y = self.Y
        else:
            X, Y = self.scaler.scale(X, Y)
        Y = mat_mul_with_intercept(X, self.alpha) - Y
        return Y ** 2 < self.epsilon ** 2

    def set_params(self, alpha: np.ndarray, x: np.ndarray, y: float) -> SliseExplainer:
        """Override the current cached explanation

        Arguments:
            alpha {np.ndarray} -- the explanation as a linear model
            x {np.ndarray} -- the explained data item
            y {float} -- the explained prediction

        Returns:
            SliseExplainer -- self, with the new explanation
        """
        self.alpha = alpha
        self.x, self.y = self.scaler.scale(x, y)
        return self

    def print(
        self, column_names: list = None, class_names: list = None, decimals: int = 3
    ) -> SliseExplainer:
        """Print the current explanation

        Keyword Arguments:
            column_names {list} -- the names of the features/variables (default: {None})
            class_names {list} -- the names of the classes, if explaining a classifier (default: {None})
            decimals {int} -- the precision to use for printing (default: {3})

        Returns:
            SliseExplainer -- self
        """
        column_names = fill_column_names(
            column_names, len(np.atleast_1d(self.coefficients)), True
        )
        column_names = [
            column_names[i]
            for i in np.concatenate(([0], np.atleast_1d(self.scaler.scaler_x.mask + 1)))
        ]
        alpha = np.atleast_1d(self.alpha)
        impact = alpha * np.concatenate(([1.0], np.atleast_1d(self.x)))
        unscaled = self.scaler.unscale(self.x, None)[0][self.scaler.scaler_x.mask]
        alpha = ["%%.%df" % decimals % a for a in alpha]
        impact = ["%%.%df" % decimals % a for a in impact]
        unscaled = [""] + ["%%.%df" % decimals % a for a in unscaled]
        col_len = (
            max(
                8,
                np.max([len(s) for s in column_names]),
                np.max([len(a) for a in alpha]),
                np.max([len(a) for a in impact]),
                np.max([len(a) for a in unscaled]),
            )
            + 1
        )
        assert len(alpha) == len(impact)
        assert len(alpha) == len(unscaled)
        assert len(alpha) == len(column_names)
        subset = self.subset()
        print(
            fill_prediction_str(
                self.scaler.scaler_y.unscale(self.y), class_names, decimals
            )
        )
        print("Variables:", " ".join([f"{s:>{col_len}}" for s in column_names]))
        print("Values:   ", " ".join([f"{s:>{col_len}}" for s in unscaled]))
        print("Weights:  ", " ".join([f"{s:>{col_len}}" for s in alpha]))
        print("Impact:   ", " ".join([f"{s:>{col_len}}" for s in impact]))
        print(f"Loss:      {self.score():>{col_len}.{decimals}f}")
        print(f"Subset:    {subset.mean():>{col_len}.{decimals}f}")
        if self.scaler.logit:
            if isinstance(class_names, list) and len(class_names) == 2:
                print(
                    f"Class Balance: {(self.Y[subset] > 0.0).mean() * 100:>.{decimals}f}% {class_names[0]} / {(self.Y[subset] < 0.0).mean() * 100:>.{decimals}f}% {class_names[1]}"
                )
            else:
                print(
                    f"Class Balance: {(self.Y[subset] > 0.0).mean() * 100:>.{decimals}f}% / {(self.Y[subset] < 0.0).mean() * 100:>.{decimals}f}%"
                )
        return self

    def plot(
        self, column_names: list = None, class_names: list = None, decimals: int = 3
    ) -> SliseExplainer:
        """Plot the current explanation (for tabular data)

        This plots three bar-plots side-by-side. The first one is the values
        from the item being explained. Note that the text is the original
        values, while the bars are showing the (potentially) scaled values. The
        second one is the weights from the linear model (on the scaled data),
        which can (very loosely) be interpreted as the importance of the
        different variables for this particular prediction. Finally, the third
        plot is the impact: the (linear model) weights times the (scaled data
        item) values normalised so the absolute sums to one. The impact might be
        an intuitive way of looking at the explanation, since a negative value
        combined with a negative weight actually supports a positive prediction,
        which the impact captures.

        Keyword Arguments:
            column_names {list} -- the names of the features/variables (default: {None})
            class_names {str or list} -- the names of the class (str) / classes (list), if explaining a classifier (default: {None})
            decimals {int} -- the precision to use for printing (default: {3})

        Returns:
            SliseExplainer -- self
        """
        plot_explanation_tabular(
            self.x, self.y, self.alpha, self.scaler, column_names, class_names, decimals
        )
        return self

    def plot_image(
        self, width: int, height: int, class_names: list = None, decimals: int = 3
    ) -> SliseExplainer:
        """Plot the current explanation for a black and white image (MNIST like)

        Arguments:
            width {int} -- the width of the image
            height {int} -- the height of the image

        Keyword Arguments:
            class_names {str or list} -- the names of the class (str) / classes (list), if explaining a classifier (default: {None})
            decimals {int} -- the precision to use for printing (default: {3})
        """
        plot_explanation_image(
            self.x,
            self.y,
            self.alpha,
            width,
            height,
            self.scaler,
            class_names,
            decimals,
        )
        return self

    def plot_dist(
        self, column_names: list = None, class_names: list = None, decimals: int = 3
    ) -> SliseExplainer:
        """Plot the current explanation (for tabular data), with density plots of the dataset and subset


        Keyword Arguments:
            column_names {list} -- the names of the features/variables (default: {None})
            class_names {str or list} -- the names of the class (str) / classes (list), if explaining a classifier (default: {None})
            decimals {int} -- the precision to use for printing (default: {3})

        Returns:
            SliseExplainer -- self
        """
        plot_explanation_dist(
            self.x,
            self.y,
            self.X,
            self.Y,
            self.alpha,
            self.subset(),
            self.scaler,
            column_names,
            class_names,
            decimals,
        )
        return self
