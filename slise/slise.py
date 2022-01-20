"""
    This script contains the main slise functions, and classes
"""

from __future__ import annotations
from typing import Union, Tuple, Callable, List
from warnings import warn
from matplotlib.pyplot import Figure
import numpy as np
from scipy.special import expit as sigmoid
from slise.data import (
    DataScaling,
    add_constant_columns,
    add_intercept_column,
    remove_constant_columns,
    normalise_robust,
    scale_same,
    unscale_model,
)
from slise.optimisation import graduated_optimisation, loss_sharp
from slise.initialisation import initialise_candidates
from slise.utils import SliseWarning, mat_mul_inter, limited_logit
from slise.plot import (
    print_slise,
    plot_2d,
    fill_column_names,
    fill_prediction_str,
    plot_dist,
    plot_image,
    plot_dist_single,
)


def regression(
    X: np.ndarray,
    Y: np.ndarray,
    epsilon: float,
    lambda1: float = 0,
    lambda2: float = 0,
    weight: Optional[np.ndarray] = None,
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

    In robust regression we fit regression models that can handle data that
    contains outliers. SLISE accomplishes this by fitting a model such that
    the largest possible subset of the data items have an error less than a
    given value. All items with an error larger than that are considered
    potential outliers and do not affect the resulting model.

    It is highly recommended that you normalise the data, either before using SLISE or by setting normalise = TRUE.
    This is a wrapper that is equivalent to `SliseRegression(epsilon, **kwargs).fit(X, Y)`

    Args:
        X (np.ndarray): the data matrix
        Y (np.ndarray): the response vector
        epsilon (float): the error tolerance
        lambda1 (float, optional): the L1 regularistaion strength. Defaults to 0.
        lambda2 (float, optional): the L2 regularisation strength. Defaults to 0.
        weight (Optional[np.ndarray], optional): weight vector for the data items. Defaults to None.
        intercept (bool, optional): add an intercept term. Defaults to True.
        normalise (bool, optional): should X aclasses not be scaled). Defaults to False.
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
    ).fit(X, Y, weight)


def explain(
    X: np.ndarray,
    Y: np.ndarray,
    epsilon: float,
    x: Union[np.ndarray, int],
    y: Union[float, None] = None,
    lambda1: float = 0,
    lambda2: float = 0,
    weight: Optional[np.ndarray] = None,
    normalise: bool = False,
    logit: bool = False,
    initialisation: Callable[
        np.ndarray, np.ndarray, float, ..., Tuple[np.ndarray, float]
    ] = initialise_candidates,
    beta_max: float = 20,
    max_approx: float = 1.15,
    max_iterations: int = 300,
    debug: bool = False,
) -> SliseExplainer:
    """Use SLISE for explaining outcomes from black box models.

    SLISE can also be used to provide local model-agnostic explanations for
    outcomes from black box models. To do this we replace the ground truth
    response vector with the predictions from the complex model. Furthermore, we
    force the model to fit a selected item (making the explanation local). This
    gives us a local approximation of the complex model with a simpler linear
    model. In contrast to other methods SLISE creates explanations using real
    data (not some discretised and randomly sampled data) so we can be sure that
    all inputs are valid (i.e. in the correct data manifold, and follows the
    constraints used to generate the data, e.g., the laws of physics).

    It is highly recommended that you normalise the data, either before using SLISE or by setting normalise = TRUE.
    This is a wrapper that is equivalent to `SliseExplainer(X, Y, epsilon, **kwargs).explain(x, y)`

    Args:
        X (np.ndarray): the data matrix
        Y (np.ndarray): the vector of predictions
        epsilon (float): the error tolerance
        x (Union[np.ndarray, int]): the data item to explain, or an index to get the item from self.X
        y (Union[float, None], optional): the outcome to explain. If x is an index then this should be None (y is taken from self.Y). Defaults to None.
        lambda1 (float, optional): the L1 regularistaion strength. Defaults to 0.
        lambda2 (float, optional): the L2 regularistaion strength. Defaults to 0.
        weight (Optional[np.ndarray], optional): weight vector for the data items. Defaults to None.
        normalise (bool, optional): should X and Y be normalised (note that epsilon will not be scaled). Defaults to False.
        logit (bool, optional): do a logit transformation on the Y vector, this is recommended only if Y consists of probabilities. Defaults to False.
        initialisation (Callable[ np.ndarray, np.ndarray, float, ..., Tuple[np.ndarray, float] ], optional): function that takes (X, Y, epslion) and gives an initial values for alpha and beta. Defaults to initialise_candidates.
        beta_max (float, optional): the final sigmoid steepness. Defaults to 20.
        max_approx (float, optional): approximation ratio when selecting the next beta. Defaults to 1.15.
        max_iterations (int, optional): maximum number of OWL-QN iterations. Defaults to 300.
        debug (bool, optional): print debug statements each graduated optimisation step. Defaults to False.

    Returns:
        SliseExplainer: object containing the explanation
    """
    return SliseExplainer(
        X,
        Y,
        epsilon,
        lambda1,
        lambda2,
        normalise,
        logit,
        initialisation,
        beta_max,
        max_approx,
        max_iterations,
        debug,
    ).explain(x, y, weight)


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
        """Use SLISE for robust regression.

        In robust regression we fit regression models that can handle data that
        contains outliers. SLISE accomplishes this by fitting a model such that
        the largest possible subset of the data items have an error less than a
        given value. All items with an error larger than that are considered
        potential outliers and do not affect the resulting model.

        This constructor prepares the parameters, call `fit` to fit a robust regression to a dataset.
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
        assert epsilon > 0.0, "epsilon must be positive"
        assert lambda1 >= 0.0, "lambda1 must not be negative"
        assert lambda2 >= 0.0, "lambda2 must not be negative"
        assert beta_max > 0.0, "beta_max must be positive"
        assert max_approx > 1.0, "max_approx must be larger than 1.0"
        assert max_iterations > 0, "max_iterations must be positive"
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
        self.weight = None

    def fit(
        self, X: np.ndarray, Y: np.ndarray, weight: Optional[np.ndarray] = None,
    ) -> SliseRegression:
        """Robustly fit a linear regression to a dataset

        Args:
            X (np.ndarray): the data matrix
            Y (np.ndarray): the response vector
            weight (Optional[np.ndarray], optional): weight vector for the data items. Defaults to None.

        Returns:
            SliseRegression: self, containing the regression result
        """
        X = np.array(X)
        Y = np.array(Y)
        if len(X.shape) == 1:
            X.shape = X.shape + (1,)
        assert X.shape[0] == Y.shape[0], "X and Y must have the same number of items"
        self.X = X
        self.Y = Y
        if weight is None:
            self.weight = None
        else:
            self.weight = np.array(weight)
            assert len(self.weight) == len(
                self.Y
            ), "weight and Y must have the same number of items"
            assert np.all(self.weight >= 0.0), "negative weights are not allowed"
        # Preprocessing
        if self.normalise:
            X, x_cols = remove_constant_columns(X)
            if self.X.shape[1] == X.shape[1]:
                x_cols = None
            X, x_center, x_scale = normalise_robust(X)
            Y, y_center, y_scale = normalise_robust(Y)
            self.scale = DataScaling(x_center, x_scale, y_center, y_scale, x_cols)
        if self.intercept:
            X = add_intercept_column(X)
        # Initialisation
        alpha, beta = self.initialisation(X, Y, self.epsilon_orig)
        # Optimisation
        alpha = graduated_optimisation(
            alpha,
            X,
            Y,
            epsilon=self.epsilon_orig,
            beta=beta,
            lambda1=self.lambda1,
            lambda2=self.lambda2,
            weight=self.weight,
            beta_max=self.beta_max,
            max_approx=self.max_approx,
            max_iterations=self.max_iterations,
            debug=self.debug,
        )
        self.alpha = alpha
        if self.normalise:
            alpha2 = self.scale.unscale_model(alpha)
            if not self.intercept:
                if np.abs(alpha2[0]) > 1e-8:
                    warn(
                        "Intercept introduced due to scaling, consider setting intercept=True (or normalise=False)",
                        SliseWarning,
                    )
                    self.intercept = True
                    self.alpha = np.concatenate(([0], alpha))
                else:
                    alpha2 = alpha2[1:]
            self.coefficients = alpha2
            self.epsilon = self.epsilon_orig * y_scale
        else:
            self.coefficients = alpha
        return self

    def get_params(self, normalised: bool = False) -> np.ndarray:
        """Get the coefficients of the linear model

        Args:
            normalised (bool, optional): if the data is normalised within SLISE, return a linear model ftting the normalised data. Defaults to False.

        Returns:
            np.ndarray: the coefficients of the linear model
        """
        return self.alpha if normalised else self.coefficients

    @property
    def normalised(self):
        if self.normalise:
            return add_constant_columns(self.alpha, self.scale.columns, self.intercept)
        else:
            return None

    def predict(self, X: Union[np.ndarray, None] = None) -> np.ndarray:
        """Use the fitted model to predict new responses

        Args:
            X (Union[np.ndarray, None], optional): data matrix to predict, or None for using the fitted dataset. Defaults to None.

        Returns:
            np.ndarray: the predicted response
        """
        if X is None:
            return mat_mul_inter(self.X, self.coefficients)
        else:
            return mat_mul_inter(X, self.coefficients)

    def score(
        self, X: Union[np.ndarray, None] = None, Y: Union[np.ndarray, None] = None
    ) -> float:
        """Calculate the loss. Lower is better and it should usually be negative (unless the regularisation is very (/too?) strong).

        Args:
            X (Union[np.ndarray, None], optional): data matrix, or None for using the fitted dataset. Defaults to None.
            Y (Union[np.ndarray, None], optional): response vector, or None for using the fitted dataset. Defaults to None.

        Returns:
            float: the loss
        """
        if X is None or Y is None:
            X = self.X
            Y = self.Y
        if self.normalise:
            X = self.scale.scale_x(X)
            Y = self.scale.scale_y(Y)
        return loss_sharp(
            self.alpha, X, Y, self.epsilon_orig, self.lambda1, self.lambda2, self.weight
        )

    loss = score

    def subset(
        self, X: Union[np.ndarray, None] = None, Y: Union[np.ndarray, None] = None
    ) -> np.ndarray:
        """Get the subset (of non-outliers) used for the robust regression model

        Args:
            X (Union[np.ndarray, None], optional): data matrix, or None for using the fitted dataset. Defaults to None.
            Y (Union[np.ndarray, None], optional): response vector, or None for using the fitted dataset. Defaults to None.

        Returns:
            np.ndarray: the selected subset as a boolean mask
        """
        if X is None or Y is None:
            X = self.X
            Y = self.Y
        Y2 = mat_mul_inter(X, self.coefficients)
        return (Y2 - Y) ** 2 < self.epsilon ** 2

    def print(
        self,
        variables: Union[List[str], None] = None,
        decimals: int = 3,
        num_var: int = 10,
    ):
        """Print the current robust regression result

        Args:
            variables ( Union[List[str], None], optional): names of the variables/columns in X. Defaults to None.
            num_var (int, optional): exclude zero weights if there are too many variables. Defaults to 10.
            decimals (int, optional): the precision to use for printing. Defaults to 3.
        """
        print_slise(
            self.coefficients,
            self.intercept,
            self.subset(),
            self.score(),
            self.epsilon,
            variables,
            "SLISE Regression",
            decimals,
            num_var,
            alpha=self.normalised,
        )

    def plot_2d(
        self,
        title: str = "SLISE Regression",
        label_x: str = "x",
        label_y: str = "y",
        decimals: int = 3,
        fig: Union[Figure, None] = None,
    ) -> SliseRegression:
        """Plot the regression in a 2D scatter plot with a line for the regression model

        Args:
            title (str, optional): plot title. Defaults to "SLISE Regression".
            label_x (str, optional): x-axis label. Defaults to "x".
            label_y (str, optional): y-axis label. Defaults to "y".
            decimals (int, optional): number of decimals when writing numbers. Defaults to 3.
            fig (Union[Figure, None], optional): Pyplot figure to plot on, if None then a new plot is created and shown. Defaults to None.

        Raises:
            SliseException: if the data has too many dimensions
        """
        plot_2d(
            self.X,
            self.Y,
            self.coefficients,
            self.epsilon,
            None,
            None,
            False,
            title,
            label_x,
            label_y,
            decimals,
            fig,
        )

    def plot_dist(
        self,
        title: str = "SLISE Regression",
        variables: list = None,
        decimals: int = 3,
        fig: Union[Figure, None] = None,
    ) -> SliseExplainer:
        """Plot the regression with density distributions for the dataset and a barplot for the model.

        Args:
            title (str, optional): title of the plot. Defaults to "SLISE Explanation".
            variables (list, optional): names for the variables. Defaults to None.
            decimals (int, optional): the number of decimals to write. Defaults to 3.
            fig (Union[Figure, None], optional): Pyplot figure to plot on, if None then a new plot is created and shown. Defaults to None.
        """
        plot_dist(
            self.X,
            self.Y,
            self.coefficients,
            self.subset(),
            self.normalised,
            None,
            None,
            None,
            None,
            title,
            variables,
            decimals,
            fig,
        )

    def plot_subset(
        self,
        title: str = "Response Distribution",
        decimals: int = 0,
        fig: Union[Figure, None] = None,
    ):
        """Plot a density distributions for response and the response of the subset

        Args:
            title (str, optional): title of the plot. Defaults to "Response Distribution".
            decimals (int, optional): number of decimals when writing the subset size. Defaults to 0.
            fig (Union[Figure, None], optional): Pyplot figure to plot on, if None then a new plot is created and shown. Defaults to None.
        """
        plot_dist_single(self.Y, self.subset(), None, title, decimals, fig)


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
        normalise: bool = False,
        initialisation: Callable[
            np.ndarray, np.ndarray, float, ..., Tuple[np.ndarray, float]
        ] = initialise_candidates,
        beta_max: float = 20,
        max_approx: float = 1.15,
        max_iterations: int = 300,
        debug: bool = False,
    ):
        """Use SLISE for explaining outcomes from black box models.

        SLISE can also be used to provide local model-agnostic explanations for
        outcomes from black box models. To do this we replace the ground truth
        response vector with the predictions from the complex model.
        Furthermore, we force the model to fit a selected item (making the
        explanation local). This gives us a local approximation of the complex
        model with a simpler linear model. In contrast to other methods SLISE
        creates explanations using real data (not some discretised and randomly
        sampled data) so we can be sure that all inputs are valid (i.e. in the
        correct data manifold, and follows the constraints used to generate the
        data, e.g., the laws of physics).

        This prepares the dataset used for the explanations, call `explain` on this object to explain outcomes.
        It is highly recommended that you normalise the data, either before using SLISE or by setting normalise = TRUE.

        Args:
            X (np.ndarray): the data matrix
            Y (np.ndarray): the vector of predictions
            epsilon (float): the error tolerance
            lambda1 (float, optional): the L1 regularistaion strength. Defaults to 0.
            lambda2 (float, optional): the L2 regularistaion strength. Defaults to 0.
            logit (bool, optional): do a logit transformation on the Y vector, this is recommended opnly if Y consists of probabilities. Defaults to False.
            normalise (bool, optional): should X and Y be normalised (note that epsilon will not be scaled). Defaults to False.
            initialisation (Callable[ np.ndarray, np.ndarray, float, ..., Tuple[np.ndarray, float] ], optional): function that takes (X, Y, epslion) and gives an initial values for alpha and beta. Defaults to initialise_candidates.
            beta_max (float, optional): the final sigmoid steepness. Defaults to 20.
            max_approx (float, optional): approximation ratio when selecting the next beta. Defaults to 1.15.
            max_iterations (int, optional): maximum number of OWL-QN iterations. Defaults to 300.
            debug (bool, optional): print debug statements each graduated optimisation step. Defaults to False.
        """
        assert epsilon > 0.0, "epsilon must be positive"
        assert lambda1 >= 0.0, "lambda1 must not be negative"
        assert lambda2 >= 0.0, "lambda2 must not be negative"
        assert beta_max > 0.0, "beta_max must be positive"
        assert max_approx > 1.0, "max_approx must be larger than 1.0"
        assert max_iterations > 0, "max_iterations must be positive"
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
        X = np.array(X)
        Y = np.array(Y)
        if len(X.shape) == 1:
            X.shape = X.shape + (1,)
        assert X.shape[0] == Y.shape[0], "X and Y must have the same number of items"
        self.X = X
        self.Y = Y
        self.x = None
        self.y = None
        self.weight = None
        self.alpha = None
        self.coefficients = None
        # Preprocess data
        if logit:
            Y = limited_logit(Y)
        if self.normalise:
            X2, x_cols = remove_constant_columns(X)
            if X.shape[1] == X2.shape[1]:
                x_cols = None
            X, x_center, x_scale = normalise_robust(X2)
            Y, y_center, y_scale = normalise_robust(Y)
            self.scale = DataScaling(x_center, x_scale, y_center, y_scale, x_cols)
            self.epsilon = epsilon * y_scale
        else:
            self.scale = None
            self.epsilon = epsilon
        self.X2 = X
        self.Y2 = Y

    def explain(
        self,
        x: Union[np.ndarray, int],
        y: Union[float, None] = None,
        weight: Optional[np.ndarray] = None,
    ) -> SliseExplainer:
        """Explain an outcome from a black box model

        Args:
            x (Union[np.ndarray, int]): the data item to explain, or an index to get the item from self.X
            y (Union[float, None], optional): the outcome to explain. If x is an index then this should be None (y is taken from self.Y). Defaults to None.
            weight (Optional[np.ndarray], optional): weight vector for the data items. Defaults to None.

        Returns:
            SliseExplainer: self, with values set to the explanation
        """
        if weight is None:
            self.weight = None
        else:
            self.weight = np.array(weight)
            assert len(self.weight) == len(
                self.Y
            ), "weight and Y must have the same number of items"
            assert np.all(self.weight >= 0.0), "negative weights are not allowed"
        if y is None:
            assert isinstance(x, int) and (
                0 <= x < self.Y.shape[0]
            ), "if y is None then x must be an integer index [0, len(Y)["
            self.y = self.Y[x]
            self.x = self.X[x, :]
            y = self.Y2[x]
            x = self.X2[x, :]
        else:
            x = np.atleast_1d(np.array(x))
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
            beta=beta,
            lambda1=self.lambda1,
            lambda2=self.lambda2,
            weight=self.weight,
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

    def get_params(self, normalised: bool = False) -> np.ndarray:
        """Get the explanation as the coefficients of a linear model (approximating the black box model)

        Args:
            normalised (bool, optional): if the data is normalised within SLISE, return a linear model fitting the normalised data. Defaults to False.

        Returns:
            np.ndarray: the coefficients of the linear model (the first scalar in the vector is the intercept)
        """
        return self.alpha if normalised else self.coefficients

    @property
    def normalised(self):
        if self.normalise:
            return add_constant_columns(self.alpha, self.scale.columns, True)
        else:
            return None

    def predict(self, X: Union[np.ndarray, None] = None) -> np.ndarray:
        """Use the approximating linear model to predict new outcomes

        Args:
            X (Union[np.ndarray, None], optional): data matrix to predict, or None for using the fitted dataset. Defaults to None.

        Returns:
            np.ndarray: prediction vector
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
        """Calculate the loss. Lower is better and it should usually be negative (unless the regularisation is very (/too?) strong).

        Args:
            X (Union[np.ndarray, None], optional): data matrix, or None for using the fitted dataset. Defaults to None.
            Y (Union[np.ndarray, None], optional): response vector, or None for using the fitted dataset. Defaults to None.

        Returns:
            float: the loss
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
            self.alpha[1:],
            X,
            Y,
            self.epsilon_orig,
            self.lambda1,
            self.lambda2,
            self.weight,
        )

    loss = score

    def subset(
        self, X: Union[np.ndarray, None] = None, Y: Union[np.ndarray, None] = None
    ) -> np.ndarray:
        """Get the subset / neighbourhood used for the approximation (explanation)

        Args:
            X (Union[np.ndarray, None], optional): data matrix, or None for using the fitted dataset. Defaults to None.
            Y (Union[np.ndarray, None], optional): response vector, or None for using the fitted dataset. Defaults to None.

        Returns:
            np.ndarray: the subset as a boolean mask
        """
        if X is None or Y is None:
            X = self.X
            Y = self.Y
        if self.logit:
            Y = limited_logit(Y)
        res = mat_mul_inter(X, self.coefficients) - Y
        return res ** 2 < self.epsilon ** 2

    def get_impact(
        self, normalised: bool = False, x: Union[None, np.ndarray] = None
    ) -> np.ndarray:
        """Get the "impact" of different variables on the outcome.
            The impact is the (normalised) model times the (normalised) item.

        Args:
            normalised (bool, optional): return the normalised impact (if normalisation is used). Defaults to False.
            x (Union[None, np.ndarray], optional): the item to calculate the impact for (uses the explained item if None). Defaults to None.

        Returns:
            np.ndarray: the impact vector
        """
        if x is None:
            x = self.x
        if normalised and self.normalise:
            return add_constant_columns(
                add_intercept_column(self.scale.scale_x(x)) * self.alpha,
                self.scale.columns,
                True,
            )
        else:
            return add_intercept_column(x) * self.coefficients

    def print(
        self,
        variables: Union[List[str], None] = None,
        classes: Union[List[str], None] = None,
        num_var: int = 10,
        decimals: int = 3,
    ):
        """Print the current explanation

        Args:
            variables (Union[List[str], None], optional): the names of the (columns/) variables. Defaults to None.
            classes (Union[List[str], None], optional): the names of the classes, if explaining a classifier. Defaults to None.
            num_var (int, optional): exclude zero weights if there are too many variables. Defaults to 10.
            decimals (int, optional): the precision to use for printing. Defaults to 3.
        """
        print_slise(
            self.coefficients,
            True,
            self.subset(),
            self.score(),
            self.epsilon,
            variables,
            "SLISE Explanation",
            decimals,
            num_var,
            unscaled=self.x,
            unscaled_y=self.y,
            impact=self.get_impact(False),
            scaled=None if self.scale is None else self.scale.scale_x(self.x, False),
            alpha=self.normalised,
            scaled_impact=None if self.scale is None else self.get_impact(True),
            classes=classes,
            unscaled_preds=self.Y,
            logit=self.logit,
        )

    def plot_2d(
        self,
        title: str = "SLISE Explanation",
        label_x: str = "x",
        label_y: str = "y",
        decimals: int = 3,
        fig: Union[Figure, None] = None,
    ) -> SliseRegression:
        """Plot the explanation in a 2D scatter plot (where the explained item is marked) with a line for the approximating model.

        Args:
            title (str, optional): plot title. Defaults to "SLISE Explanation".
            label_x (str, optional): x-axis label. Defaults to "x".
            label_y (str, optional): y-axis label. Defaults to "y".
            decimals (int, optional): number of decimals when writing numbers. Defaults to 3.
            fig (Union[Figure, None], optional): Pyplot figure to plot on, if None then a new plot is created and shown. Defaults to None.

        Raises:
            SliseException: if the data has too many dimensions
        """
        plot_2d(
            self.X,
            self.Y,
            self.coefficients,
            self.epsilon,
            self.x,
            self.y,
            self.logit,
            title,
            label_x,
            label_y,
            decimals,
            fig,
        )

    def plot_image(
        self,
        width: int,
        height: int,
        saturated: bool = True,
        title: str = "SLISE Explanation",
        classes: Union[List, str, None] = None,
        decimals: int = 3,
        fig: Union[Figure, None] = None,
    ) -> SliseExplainer:
        """Plot the current explanation for a black and white image (e.g. MNIST)

        Args:
            width (int): the width of the image
            height (int): the height of the image
            saturated (bool, optional): should the explanation be more saturated. Defaults to True.
            title (str, optional): title of the plot. Defaults to "SLISE Explanation".
            classes (Union[List, str, None], optional): list of class names (first the negative, then the positive), or a single (positive) class name. Defaults to None.
            decimals (int, optional): the number of decimals to write. Defaults to 3.
            fig (Union[Figure, None], optional): Pyplot figure to plot on, if None then a new plot is created and shown. Defaults to None.
        """
        plot_image(
            self.x,
            self.y,
            self.Y,
            self.coefficients,
            width,
            height,
            saturated,
            title,
            classes,
            decimals,
            fig,
        )

    def plot_dist(
        self,
        title: str = "SLISE Explanation",
        variables: list = None,
        decimals: int = 3,
        fig: Union[Figure, None] = None,
    ) -> SliseExplainer:
        """Plot the current explanation with density distributions for the dataset and a barplot for the model.

        The barbplot contains both the approximating linear model (where the
        weights can be loosely interpreted as the importance of the different
        variables and their sign) and the "impact" which is the (scaled) model
        time the (scaled) item values (which demonstrates how the explained
        item interacts with the approximating linear model, since a negative
        weight times a negative value actually supports a positive prediction).

        Args:
            title (str, optional): title of the plot. Defaults to "SLISE Explanation".
            variables (list, optional): names for the variables. Defaults to None.
            decimals (int, optional): the number of decimals to write. Defaults to 3.
            fig (Union[Figure, None], optional): Pyplot figure to plot on, if None then a new plot is created and shown. Defaults to None.
        """
        plot_dist(
            self.X,
            self.Y,
            self.coefficients,
            self.subset(),
            self.normalised,
            self.x,
            self.y,
            self.get_impact(False),
            self.get_impact(True) if self.normalise else None,
            title,
            variables,
            decimals,
            fig,
        )

    def plot_subset(
        self,
        title: str = "Prediction Distribution",
        decimals: int = 0,
        fig: Union[Figure, None] = None,
    ):
        """Plot a density distributions for predictions and the predictions of the subset

        Args:
            title (str, optional): title of the plot. Defaults to "Prediction Distribution".
            decimals (int, optional): number of decimals when writing the subset size. Defaults to 0.
            fig (Union[Figure, None], optional): Pyplot figure to plot on, if None then a new plot is created and shown. Defaults to None.
        """
        plot_dist_single(self.Y, self.subset(), self.y, title, decimals, fig)
