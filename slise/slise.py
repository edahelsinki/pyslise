"""
    This script contains the main slise functions, and classes
"""

from __future__ import annotations
from typing import Union, Tuple, Callable, List
from warnings import warn
import numpy as np
from scipy.special import expit as sigmoid
from slise.data import (
    DataScaling,
    add_constant_columns,
    add_intercept_column,
    remove_constant_columns,
    scale_robust,
    scale_same,
    unscale_model,
)
from slise.optimisation import graduated_optimisation, loss_sharp
from slise.initialisation import initialise_candidates
from slise.utils import SliseWarning, mat_mul_inter, limited_logit
from slise.plot import (
    plot_regression_2D,
    fill_column_names,
    fill_prediction_str,
    plot_explanation_tabular,
    plot_explanation_dist,
    plot_explanation_image,
)


def regression(
    X: np.ndarray,
    Y: np.ndarray,
    epsilon: float,
    lambda1: float = 0,
    lambda2: float = 0,
    intercept: bool = True,
    normalise: bool = False,
    initialisation: Callable[
        np.ndarray, np.ndarray, float, ..., Tuple[np.ndarray, float]
    ] = initialise_candidates,
    beta_max: float = 20,
    max_approx: float = 1.15,
    max_iterations: int = 300,
    debug: bool = False,
) -> SliseRegression:
    """Use SLISE for robust regression
        It is highly recommended that you normalise the data, either before using SLISE or by setting normalise = TRUE.
        This is a wrapper that is equivalent to `SliseRegression(epsilon, **kwargs).fit(X, Y)`

        Args:
            X (np.ndarray): the data matrix
            Y (np.ndarray): the response vector
            epsilon (float): the error tolerance
            lambda1 (float, optional): the L1 regularistaion strength. Defaults to 0.
            lambda2 (float, optional): the L2 regularisation strength. Defaults to 0.
            intercept (bool, optional): add an intercept term. Defaults to True.
            normalise (bool, optional): should X and Y be normalised (note that epsilon will not be scaled). Defaults to False.
            initialisation (Callable[ np.ndarray, np.ndarray, ..., Tuple[np.ndarray, float] ], optional): function that takes X, Y and gives an initial values for alpha and beta. Defaults to initialise_candidates.
            beta_max (float, optional): the stopping sigmoid steepness. Defaults to 20.
            max_approx (float, optional): approximation ratio when selecting the next beta. Defaults to 1.15.
            max_iterations (int, optional): maximum number of OWL-QN iterations. Defaults to 300.
            debug (bool, optional): print debug statements each graduated optimisation step. Defaults to False.

        Returns:
            SliseRegression: object containing the regression result
    """
    return SliseRegression(
        epsilon,
        lambda1,
        lambda2,
        intercept,
        normalise,
        initialisation,
        beta_max,
        max_approx,
        max_iterations,
        debug,
    ).fit(X, Y)


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
        epsilon: float,
        lambda1: float = 0,
        lambda2: float = 0,
        intercept: bool = True,
        normalise: bool = False,
        initialisation: Callable[
            np.ndarray, np.ndarray, float, ..., Tuple[np.ndarray, float]
        ] = initialise_candidates,
        beta_max: float = 20,
        max_approx: float = 1.15,
        max_iterations: int = 300,
        debug: bool = False,
    ):
        """Use SLISE for robust regression
        It is highly recommended that you normalise the data, either before using SLISE or by setting normalise = TRUE.

        Args:
            epsilon (float): the error tolerance
            lambda1 (float, optional): the L1 regularistaion strength. Defaults to 0.
            lambda2 (float, optional): the L2 regularisation strength. Defaults to 0.
            intercept (bool, optional): add an intercept term. Defaults to True.
            normalise (bool, optional): should X and Y be normalised (note that epsilon will not be scaled). Defaults to False.
            initialisation (Callable[ np.ndarray, np.ndarray, ..., Tuple[np.ndarray, float] ], optional): function that takes (X, Y, epslion) and gives an initial values for alpha and beta. Defaults to initialise_candidates.
            beta_max (float, optional): the stopping sigmoid steepness. Defaults to 20.
            max_approx (float, optional): approximation ratio when selecting the next beta. Defaults to 1.15.
            max_iterations (int, optional): maximum number of OWL-QN iterations. Defaults to 300.
            debug (bool, optional): print debug statements each graduated optimisation step. Defaults to False.
        """
        self.epsilon_orig = epsilon
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.intercept = intercept
        self.normalise = normalise
        self.initialisation = initialisation
        self.beta_max = beta_max
        self.max_approx = max_approx
        self.max_iterations = max_iterations
        self.debug = debug
        self.alpha = None
        self.coefficients = None
        self.epsilon = epsilon
        self.X = None
        self.Y = None
        self.scale = None

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
        self.X = X
        self.Y = Y
        # Preprocessing
        if self.normalise:
            X, x_cols = remove_constant_columns(X)
            if self.X.shape[1] == X.shape[1]:
                x_cols = None
            X, x_center, x_scale = scale_robust(X)
            Y, y_center, y_scale = scale_robust(Y)
            self.scale = DataScaling(x_center, x_scale, y_center, y_scale, x_cols)
        if self.intercept:
            X = add_intercept_column(X)
        # Initialisation
        alpha, beta = self.initialisation(X, Y, self.epsilon_orig)
        # Optimisation
        self.alpha = graduated_optimisation(
            alpha,
            X,
            Y,
            epsilon=self.epsilon_orig,
            lambda1=self.lambda1,
            lambda2=self.lambda2,
            beta=beta,
            beta_max=self.beta_max,
            max_approx=self.max_approx,
            max_iterations=self.max_iterations,
            debug=self.debug,
        )
        if self.normalise:
            alpha2 = self.scale.unscale_model(alpha)
            if not self.intercept:
                if np.abs(alpha2[0]) > 1e-8:
                    warn(
                        "Intercept introduced due to scaling (consider setting scale_*=False, or intercept=True)",
                        SliseWarning,
                    )
                else:
                    alpha2 = alpha2[1:]
            self.coefficients = alpha2
            self.epsilon = self.epsilon_orig * y_scale
        else:
            self.coefficients = self.alpha
        return self

    def get_params(self, scaled: bool = False) -> np.ndarray:
        """Get the coefficients of the linear model

        Keyword Arguments:
            scaled {bool} -- return the model that fits the scaled data (if the data is being scaled within SLISE) (default: {False})

        Returns:
            np.ndarray -- the coefficients of the linear model
        """
        return self.alpha if scaled else self.coefficients

    def predict(self, X: Union[np.ndarray, None] = None) -> np.ndarray:
        """Use the fitted model to predict new responses

        Keyword Arguments:
            X {np.ndarray} -- the data to predict, or None for using the give dataset (default: {None})

        Returns:
            np.ndarray -- the response
        """
        if X is None:
            return mat_mul_inter(self.X, self.coefficients)
        else:
            return mat_mul_inter(X, self.coefficients)

    def score(
        self, X: Union[np.ndarray, None] = None, Y: Union[np.ndarray, None] = None
    ) -> float:
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
        if self.normalise:
            X = self.scale.scale_x(X)
            Y = self.scale.scale_y(Y)
        return loss_sharp(
            self.alpha, X, Y, self.epsilon_orig, self.lambda1, self.lambda2
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
        Y2 = mat_mul_inter(X, self.coefficients)
        return (Y2 - Y) ** 2 < self.epsilon ** 2

    def print(
        self, column_names: Union[List[str], None] = None, decimals: int = 3
    ) -> SliseRegression:
        """Print the current robust regression result

        Keyword Arguments:
            column_names {list} -- names of the variables/columns in X (default: {None})
            decimals {int} -- the precision to use for printing (default: {3})

        Returns:
            SliseRegression -- self
        """
        intercept = self.intercept or len(self.coefficients) > self.X.shape[1]
        alpha = self.alpha
        coeff = self.coefficients
        if self.normalise and self.scale.columns is not None:
            alpha = add_constant_columns(alpha, self.scale.columns, self.intercept)
        if len(alpha) < len(coeff):
            alpha = np.concatenate(([0.0], alpha))
        column_names = fill_column_names(column_names, len(coeff), intercept)
        alpha = ["%%.%df" % decimals % a for a in alpha]
        coeff = ["%%.%df" % decimals % a for a in coeff]
        col_len = max(
            8,
            max(len(s) for s in column_names),
            max(len(a) for a in alpha),
            max(len(a) for a in coeff),
        )
        assert len(alpha) == len(coeff)
        assert len(alpha) == len(column_names)
        print("SLISE Regression")
        print("Variables:   ", " ".join([f"{s:>{col_len}}" for s in column_names]))
        print("Coefficients:", " ".join([f"{s:>{col_len}}" for s in coeff]))
        if self.normalise:
            print("Scaled Alpha:", " ".join([f"{s:>{col_len}}" for s in alpha]))
        print(f"Loss:         {self.score():>{col_len}.{decimals}f}")
        print(f"Subset:       {self.subset().mean():>{col_len}.{decimals}f}")
        print(f"Epsilon:      {self.epsilon:>{col_len}.{decimals}f}")
        return self

    def plot(
        self, label_x: str = "x", label_y: str = "y", decimals: int = 3
    ) -> SliseRegression:
        """Plot 1D data in a 2D scatter plot, with a line for the regression model

        Keyword Arguments:
            label_x {str} -- the name of the dependent value (default: "x")
            label_y {str} -- the name of the predicted value (default: "y")
            decimals {int} -- the number of decimals for the axes (default: {3})

        Raises:
            Exception: if the data is not 1D (intercept allowed)

        Returns:
            SliseRegression -- self
        """
        plot_regression_2D(
            self.X,
            self.Y,
            self.coefficients,
            self.epsilon * (self.scale.y_scale if self.normalise else 1),
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
        epsilon: float,
        lambda1: float = 0,
        lambda2: float = 0,
        logit: bool = False,
        normalise=False,
        initialisation: Callable[
            np.ndarray, np.ndarray, float, ..., Tuple[np.ndarray, float]
        ] = initialise_candidates,
        beta_max: float = 20,
        max_approx: float = 1.15,
        max_iterations: int = 300,
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
            normalise (bool, optional): should X and Y be normalised (note that epsilon will not be scaled). Defaults to False.
            initialisation (Callable[ np.ndarray, np.ndarray, ..., Tuple[np.ndarray, float] ], optional): function that takes (X, Y, epslion) and gives an initial values for alpha and beta. Defaults to initialise_candidates.
            beta_max {float} -- the stopping sigmoid steepness (default: {25})
            max_approx {float} -- the target approximation ratio for the graduated optimisation (default: {1.15})
            max_iterations {int} -- the maximum iterations of OWL-QN per graduated optimisation step (default: {200})
            pca_treshold {int} -- the treshold for using pca in the initialisation (default: {10})
            inits {int} -- the number of candidates to generate in the initialisation (default: {500})
            debug {bool} -- print debug statements each graduated optimisation step (default: {False})
        """
        self.epsilon_orig = epsilon
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.logit = logit
        self.normalise = normalise
        self.initialisation = initialisation
        self.beta_max = beta_max
        self.max_approx = max_approx
        self.max_iterations = max_iterations
        self.debug = debug
        if len(X.shape) == 1:
            X.shape += (1,)
        self.X = X
        self.Y = Y
        self.x = None
        self.y = None
        self.alpha = None
        self.coefficients = None
        # Preprocess data
        if logit:
            Y = limited_logit(Y)
        if self.normalise:
            X2, x_cols = remove_constant_columns(X)
            if X.shape[1] == X2.shape[1]:
                x_cols = None
            X, x_center, x_scale = scale_robust(X2)
            Y, y_center, y_scale = scale_robust(Y)
            self.scale = DataScaling(x_center, x_scale, y_center, y_scale, x_cols)
            self.epsilon = epsilon * y_scale
        else:
            self.scale = None
            self.epsilon = epsilon
        self.X2 = X
        self.Y2 = Y

    def explain(self, x: np.ndarray, y: float = None) -> SliseExplainer:
        """Explain an outcome from a black box model

        Arguments:
            x {np.ndarray} -- the data item to explain, or an index from the dataset X

        Keyword Arguments:
            y {float} -- the prediction from the black box model, or None if x is an index (default: {None})

        Returns:
            SliseExplainer -- self, with values set to the explanation
        """
        if isinstance(x, int) and y is None:
            self.y = self.Y[x]
            self.x = self.X[x, :]
            y = self.Y2[x]
            x = self.X2[x, :]
        else:
            self.x = x
            self.y = y
            if self.logit:
                y = limited_logit(y)
            if self.normalise:
                x = self.scale.scale_x(x)
                y = self.scale.scale_y(y)
        X = self.X2 - x[None, :]
        Y = self.Y2 - y
        alpha, beta = self.initialisation(X, Y, self.epsilon_orig)
        alpha = graduated_optimisation(
            alpha,
            X,
            Y,
            epsilon=self.epsilon_orig,
            lambda1=self.lambda1,
            lambda2=self.lambda2,
            beta=beta,
            beta_max=self.beta_max,
            max_approx=self.max_approx,
            max_iterations=self.max_iterations,
            debug=self.debug,
        )
        alpha = np.concatenate(
            (y - np.sum(alpha * x, dtype=x.dtype, keepdims=True), alpha)
        )
        self.alpha = alpha
        if self.normalise:
            alpha2 = self.scale.unscale_model(alpha)
            alpha2[0] = self.y - np.sum(self.x * alpha2[1:])
            self.coefficients = alpha2
        else:
            self.coefficients = alpha
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
            Y = mat_mul_inter(self.X, self.coefficients)
        else:
            Y = mat_mul_inter(X, self.coefficients)
        if self.scaler.logit:
            Y = sigmoid(Y)
        return Y

    def score(
        self, X: Union[np.ndarray, None] = None, Y: Union[np.ndarray, None] = None
    ) -> float:
        """Calculate the loss

        Keyword Arguments:
            X {np.ndarray} -- the data, None for the given dataset (default: {None})
            Y {np.ndarray} -- the response, None for the given dataset (default: {None})

        Returns:
            float -- the loss
        """
        x = self.x
        y = self.y
        if self.logit:
            y = limited_logit(y)
        if self.normalise:
            x = self.scale.scale_x(x)
            y = self.scale.scale_y(y)
        if X is None or Y is None:
            X = self.X2
            Y = self.Y2
        else:
            if self.logit:
                Y = limited_logit(Y)
            if self.normalise:
                X = self.scale.scale_x(X)
                Y = self.scale.scale_y(Y)
        X = X - x[None, :]
        Y = Y - y
        return loss_sharp(
            self.alpha[1:], X, Y, self.epsilon_orig, self.lambda1, self.lambda2,
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
            res = mat_mul_inter(self.X2, self.alpha) - self.Y2
            return res ** 2 < self.epsilon_orig ** 2
        else:
            if self.logit:
                Y = limited_logit(Y)
            res = mat_mul_inter(X, self.coefficients) - Y
            return res ** 2 < self.epsilon ** 2

    def print(
        self,
        column_names: Union[List[str], None] = None,
        class_names: Union[List[str], None] = None,
        decimals: int = 3,
    ) -> SliseExplainer:
        """Print the current explanation

        Keyword Arguments:
            column_names {list} -- the names of the features/variables (default: {None})
            class_names {list} -- the names of the classes, if explaining a classifier (default: {None})
            decimals {int} -- the precision to use for printing (default: {3})

        Returns:
            SliseExplainer -- self
        """
        column_names = fill_column_names(column_names, len(self.coefficients), True)[1:]
        alpha = self.alpha[1:]
        unscaled = self.x
        scaled = unscaled
        if self.normalise:
            scaled = self.scale.scale_x(unscaled)
            if self.scale.columns is not None:
                unscaled = unscaled[self.scale.columns]
                column_names = column_names[self.scale.columns]
        impact = scaled * alpha
        alpha = ["%%.%df" % decimals % a for a in alpha]
        impact = ["%%.%df" % decimals % a for a in impact]
        unscaled = ["%%.%df" % decimals % a for a in unscaled]
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
        print("SLISE Explanation")
        print(fill_prediction_str(self.y, class_names, decimals))
        print("Variables:", " ".join([f"{s:>{col_len}}" for s in column_names]))
        print("Values:   ", " ".join([f"{s:>{col_len}}" for s in unscaled]))
        print("Weights:  ", " ".join([f"{s:>{col_len}}" for s in alpha]))
        print("Impact:   ", " ".join([f"{s:>{col_len}}" for s in impact]))
        print(f"Intercept: {self.alpha[0]:>{col_len}.{decimals}f}")
        print(f"Loss:      {self.score():>{col_len}.{decimals}f}")
        print(f"Subset:    {subset.mean():>{col_len}.{decimals}f}")
        print(f"Epsilon:   {self.epsilon:>{col_len}.{decimals}f}")
        if self.logit:
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
