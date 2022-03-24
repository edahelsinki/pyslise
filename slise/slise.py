"""
    This script contains the main slise functions, and classes
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple, Union
from warnings import warn

import numpy as np
from matplotlib.pyplot import Figure
from scipy.special import expit as sigmoid

from slise.data import (
    DataScaling,
    add_constant_columns,
    add_intercept_column,
    normalise_robust,
    remove_constant_columns,
)
from slise.initialisation import initialise_candidates, initialise_fixed
from slise.optimisation import (
    check_threading_layer,
    graduated_optimisation,
    loss_sharp,
    set_threads,
)
from slise.plot import plot_2d, plot_dist, plot_dist_single, plot_image, print_slise
from slise.utils import SliseWarning, limited_logit, mat_mul_inter


def regression(
    X: np.ndarray,
    Y: np.ndarray,
    epsilon: float,
    lambda1: float = 0,
    lambda2: float = 0,
    weight: Optional[np.ndarray] = None,
    intercept: bool = True,
    normalise: bool = False,
    init: Union[None, np.ndarray, Tuple[np.ndarray, float]] = None,
    initialisation: Callable[
        [np.ndarray, np.ndarray, float, Optional[np.ndarray]], Tuple[np.ndarray, float]
    ] = initialise_candidates,
    beta_max: float = 20,
    max_approx: float = 1.15,
    max_iterations: int = 300,
    debug: bool = False,
    num_threads: int = -1,
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
        X (np.ndarray): Data matrix.
        Y (np.ndarray): Response vector.
        epsilon (float): Error tolerance.
        lambda1 (float, optional): L1 regularisation strength. Defaults to 0.
        lambda2 (float, optional): L2 regularisation strength. Defaults to 0.
        weight (Optional[np.ndarray], optional): Weight vector for the data items. Defaults to None.
        intercept (bool, optional): Add an intercept term. Defaults to True.
        normalise (bool, optional): Should X aclasses not be scaled). Defaults to False.
        init (Union[None, np.ndarray, Tuple[np.ndarray, float]], optional): Use this alpha (and beta) value instead of the initialisation function. Defaults to None.
        initialisation (Callable[ [np.ndarray, np.ndarray, float, Optional[np.ndarray]], Tuple[np.ndarray, float] ], optional): Function that takes `(X, Y, epsilon, weight)` and gives an initial values for alpha and beta. Defaults to initialise_candidates.
        beta_max (float, optional): The stopping sigmoid steepness. Defaults to 20.
        max_approx (float, optional): Approximation ratio when selecting the next beta. Defaults to 1.15.
        max_iterations (int, optional): Maximum number of OWL-QN iterations. Defaults to 300.
        debug (bool, optional): Print debug statements each graduated optimisation step. Defaults to False.
        num_threads (int, optional): The number of threads to use for the optimisation. Defaults to -1.

    Returns:
        SliseRegression: Object containing the regression result.
    """
    return SliseRegression(
        epsilon=epsilon,
        lambda1=lambda1,
        lambda2=lambda2,
        intercept=intercept,
        normalise=normalise,
        initialisation=initialisation,
        beta_max=beta_max,
        max_approx=max_approx,
        max_iterations=max_iterations,
        debug=debug,
        num_threads=num_threads,
    ).fit(X=X, Y=Y, weight=weight, init=init)


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
    init: Union[None, np.ndarray, Tuple[np.ndarray, float]] = None,
    initialisation: Callable[
        [np.ndarray, np.ndarray, float, Optional[np.ndarray]], Tuple[np.ndarray, float]
    ] = initialise_candidates,
    beta_max: float = 20,
    max_approx: float = 1.15,
    max_iterations: int = 300,
    debug: bool = False,
    num_threads: int = -1,
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
        X (np.ndarray): Data matrix.
        Y (np.ndarray): Vector of predictions.
        epsilon (float): Error tolerance.
        x (Union[np.ndarray, int]): The data item to explain, or an index to get the item from self.X
        y (Union[float, None], optional): The outcome to explain. If x is an index then this should be None (y is taken from self.Y). Defaults to None.
        lambda1 (float, optional): L1 regularisation strength. Defaults to 0.
        lambda2 (float, optional): L2 regularisation strength. Defaults to 0.
        weight (Optional[np.ndarray], optional): Weight vector for the data items. Defaults to None.
        normalise (bool, optional): Should X and Y be normalised (note that epsilon will not be scaled). Defaults to False.
        logit (bool, optional): Do a logit transformation on the Y vector, this is recommended only if Y consists of probabilities. Defaults to False.
        init (Union[None, np.ndarray, Tuple[np.ndarray, float]], optional): Use this alpha (and beta) value instead of the initialisation function. Defaults to None.
        initialisation (Callable[ [np.ndarray, np.ndarray, float, Optional[np.ndarray]], Tuple[np.ndarray, float] ], optional): Function that takes `(X, Y, epsilon, weight)` and gives an initial values for alpha and beta. Defaults to initialise_candidates.
        beta_max (float, optional): The final sigmoid steepness. Defaults to 20.
        max_approx (float, optional): Approximation ratio when selecting the next beta. Defaults to 1.15.
        max_iterations (int, optional): Maximum number of OWL-QN iterations. Defaults to 300.
        debug (bool, optional): Print debug statements each graduated optimisation step. Defaults to False.
        num_threads (int, optional): The number of threads to use for the optimisation. Defaults to -1.

    Returns:
        SliseExplainer: Object containing the explanation.
    """
    return SliseExplainer(
        X=X,
        Y=Y,
        epsilon=epsilon,
        lambda1=lambda1,
        lambda2=lambda2,
        normalise=normalise,
        logit=logit,
        initialisation=initialisation,
        beta_max=beta_max,
        max_approx=max_approx,
        max_iterations=max_iterations,
        debug=debug,
        num_threads=num_threads,
    ).explain(x=x, y=y, weight=weight, init=init)


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
            [np.ndarray, np.ndarray, float, Optional[np.ndarray]],
            Tuple[np.ndarray, float],
        ] = initialise_candidates,
        beta_max: float = 20,
        max_approx: float = 1.15,
        max_iterations: int = 300,
        debug: bool = False,
        num_threads: int = -1,
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
            epsilon (float): Error tolerance.
            lambda1 (float, optional): L1 regularisation strength. Defaults to 0.
            lambda2 (float, optional): L2 regularisation strength. Defaults to 0.
            intercept (bool, optional): Add an intercept term. Defaults to True.
            normalise (bool, optional): Should X and Y be normalised (note that epsilon will not be scaled). Defaults to False.
            initialisation (Callable[ [np.ndarray, np.ndarray, float, Optional[np.ndarray]], Tuple[np.ndarray, float] ], optional): Function that takes `(X, Y, epsilon, weight)` and gives an initial values for alpha and beta. Defaults to initialise_candidates.
            beta_max (float, optional): The stopping sigmoid steepness. Defaults to 20.
            max_approx (float, optional): Approximation ratio when selecting the next beta. Defaults to 1.15.
            max_iterations (int, optional): Maximum number of OWL-QN iterations. Defaults to 300.
            debug (bool, optional): Print debug statements each graduated optimisation step. Defaults to False.
            num_threads (int, optional): The number of threads to use for the optimisation. Defaults to -1.
        """
        assert epsilon > 0.0, "`epsilon` must be positive!"
        assert lambda1 >= 0.0, "`lambda1` must not be negative!"
        assert lambda2 >= 0.0, "`lambda2` must not be negative!"
        assert beta_max > 0.0, "`beta_max` must be positive!"
        assert max_approx > 1.0, "`max_approx` must be larger than 1.0!"
        assert max_iterations > 0, "`max_iterations` must be positive!"
        self.epsilon = epsilon
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.init_fn = initialisation
        self.beta_max = beta_max
        self.max_approx = max_approx
        self.max_iterations = max_iterations
        self.debug = debug
        self._intercept = intercept
        self._normalise = normalise
        self._scale = None
        self._X = None
        self._Y = None
        self._weight = None
        self._alpha = None
        self._coefficients = None
        self.num_threads = num_threads
        check_threading_layer()

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        weight: Optional[np.ndarray] = None,
        init: Union[None, np.ndarray, Tuple[np.ndarray, float]] = None,
    ) -> SliseRegression:
        """Robustly fit a linear regression to a dataset

        Args:
            X (np.ndarray): Data matrix.
            Y (np.ndarray): Response vector.
            weight (Optional[np.ndarray], optional): Weight vector for the data items. Defaults to None.
            init (Union[None, np.ndarray, Tuple[np.ndarray, float]], optional): Use this alpha (and beta) value instead of the initialisation function. Defaults to None.

        Returns:
            SliseRegression: `self` (containing the regression result).
        """
        X = np.array(X)
        Y = np.array(Y)
        if len(X.shape) == 1:
            X.shape = X.shape + (1,)
        assert X.shape[0] == Y.shape[0], "X and Y must have the same number of items!"
        self._X = X
        self._Y = Y
        if weight is None:
            self._weight = None
        else:
            self._weight = np.array(weight)
            assert len(self._weight) == len(
                self._Y
            ), "Y and weight must have the same number of items!"
            assert np.all(self._weight >= 0.0), "Weights must not be negative!"
        # Preprocessing
        if self._normalise:
            X, x_cols = remove_constant_columns(X)
            if self._X.shape[1] == X.shape[1]:
                x_cols = None
            X, x_center, x_scale = normalise_robust(X)
            Y, y_center, y_scale = normalise_robust(Y)
            self._scale = DataScaling(x_center, x_scale, y_center, y_scale, x_cols)
        if self._intercept:
            X = add_intercept_column(X)
        # Initialisation
        threads = set_threads(self.num_threads)
        if init is None:
            alpha, beta = self.init_fn(X, Y, self.epsilon, self._weight)
        else:
            alpha, beta = initialise_fixed(init, X, Y, self.epsilon, self._weight)
        # Optimisation
        alpha = graduated_optimisation(
            alpha=alpha,
            X=X,
            Y=Y,
            epsilon=self.epsilon,
            beta=beta,
            lambda1=self.lambda1,
            lambda2=self.lambda2,
            weight=self._weight,
            beta_max=self.beta_max,
            max_approx=self.max_approx,
            max_iterations=self.max_iterations,
            debug=self.debug,
        )
        set_threads(threads)
        self._alpha = alpha
        if self._normalise:
            alpha2 = self._scale.unscale_model(alpha)
            if not self._intercept:
                if np.abs(alpha2[0]) > 1e-8:
                    warn(
                        "Intercept introduced due to scaling, consider setting intercept=True (or normalise=False)",
                        SliseWarning,
                    )
                    self._intercept = True
                    self._alpha = np.concatenate(([0], alpha))
                else:
                    alpha2 = alpha2[1:]
            self._coefficients = alpha2
        else:
            self._coefficients = alpha
        return self

    def get_params(self, normalised: bool = False) -> np.ndarray:
        """Get the coefficients of the linear model.

        Args:
            normalised (bool, optional): If the data is normalised within SLISE, return a linear model ftting the normalised data. Defaults to False.

        Returns:
            np.ndarray: The coefficients of the linear model.
        """
        warn("Use `coefficients` instead of `get_params()`.", SliseWarning)
        return self._alpha if normalised else self._coefficients

    @property
    def coefficients(self) -> np.ndarray:
        """Get the coefficients of the linear model.

        Returns:
            np.ndarray: The coefficients of the linear model (the first scalar in the vector is the intercept).
        """
        if self._coefficients is None:
            warn("Fit the model before retrieving coefficients", SliseWarning)
        return self._coefficients

    def normalised(self, all_columns: bool = True) -> Optional[np.ndarray]:
        """Get coefficients for normalised data (if the data is normalised within SLISE).

        Args:
            all_columns (bool, optional): Add coefficients for constant columns. Defaults to True.

        Returns:
            Optional[np.ndarray]: The normalised coefficients or None.
        """
        if self._alpha is None:
            warn("Fit the model before retrieving coefficients", SliseWarning)
        if self._normalise:
            if all_columns:
                return add_constant_columns(self._alpha, self._scale.columns, True)
            else:
                return self._alpha
        else:
            return None

    @property
    def scaled_epsilon(self) -> float:
        """Espilon fitting unnormalised data (if the data is normalised).

        Returns:
            float: Scaled epsilon.
        """
        if self._normalise:
            return self.epsilon * self._scale.y_scale
        else:
            return self.epsilon

    def predict(self, X: Union[np.ndarray, None] = None) -> np.ndarray:
        """Use the fitted model to predict new responses.

        Args:
            X (Union[np.ndarray, None], optional): Data matrix to predict, or None for using the fitted dataset. Defaults to None.

        Returns:
            np.ndarray: Predicted response.
        """
        if X is None:
            return mat_mul_inter(self._X, self.coefficients)
        else:
            return mat_mul_inter(X, self.coefficients)

    def score(
        self, X: Union[np.ndarray, None] = None, Y: Union[np.ndarray, None] = None
    ) -> float:
        """Calculate the loss. Lower is better and it should usually be negative (unless the regularisation is very (too?) strong).

        Args:
            X (Union[np.ndarray, None], optional): Data matrix, or None for using the fitted dataset. Defaults to None.
            Y (Union[np.ndarray, None], optional): Response vector, or None for using the fitted dataset. Defaults to None.

        Returns:
            float: The loss.
        """
        if self._alpha is None:
            warn("Fit the model before calculating the score", SliseWarning)
        if X is None or Y is None:
            X = self._X
            Y = self._Y
        if self._normalise:
            X = self._scale.scale_x(X)
            Y = self._scale.scale_y(Y)
        return loss_sharp(
            self._alpha, X, Y, self.epsilon, self.lambda1, self.lambda2, self._weight
        )

    loss = score
    value = score

    def subset(
        self, X: Union[np.ndarray, None] = None, Y: Union[np.ndarray, None] = None
    ) -> np.ndarray:
        """Get the subset (of non-outliers) used for the robust regression model.

        Args:
            X (Union[np.ndarray, None], optional): Data matrix, or None for using the fitted dataset. Defaults to None.
            Y (Union[np.ndarray, None], optional): Response vector, or None for using the fitted dataset. Defaults to None.

        Returns:
            np.ndarray: The selected subset as a boolean mask.
        """
        if X is None or Y is None:
            X = self._X
            Y = self._Y
        Y2 = mat_mul_inter(X, self.coefficients)
        return (Y2 - Y) ** 2 < self.scaled_epsilon ** 2

    def print(
        self,
        variables: Union[List[str], None] = None,
        decimals: int = 3,
        num_var: int = 10,
    ):
        """Print the current robust regression result.

        Args:
            variables (Union[List[str], None], optional): Names of the variables/columns in X. Defaults to None.
            num_var (int, optional): Exclude zero weights if there are too many variables. Defaults to 10.
            decimals (int, optional): Precision to use for printing. Defaults to 3.
        """
        print_slise(
            self.coefficients,
            self._intercept,
            self.subset(),
            self.score(),
            self.scaled_epsilon,
            variables,
            "SLISE Regression",
            decimals,
            num_var,
            alpha=self.normalised(),
        )

    def plot_2d(
        self,
        title: str = "SLISE Regression",
        label_x: str = "x",
        label_y: str = "y",
        decimals: int = 3,
        fig: Union[Figure, None] = None,
    ) -> SliseRegression:
        """Plot the regression in a 2D scatter plot with a line for the regression model.

        Args:
            title (str, optional): Title of the plot. Defaults to "SLISE Regression".
            label_x (str, optional): X-axis label. Defaults to "x".
            label_y (str, optional): Y-axis label. Defaults to "y".
            decimals (int, optional): Number of decimals when writing numbers. Defaults to 3.
            fig (Union[Figure, None], optional): Pyplot figure to plot on, if None then a new plot is created and shown. Defaults to None.

        Raises:
            SliseException: If the data has too many dimensions.
        """
        plot_2d(
            self._X,
            self._Y,
            self.coefficients,
            self.scaled_epsilon,
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
            title (str, optional): Title of the plot. Defaults to "SLISE Explanation".
            variables (list, optional): Names for the variables. Defaults to None.
            decimals (int, optional): Number of decimals to write. Defaults to 3.
            fig (Union[Figure, None], optional): Pyplot figure to plot on, if None then a new plot is created and shown. Defaults to None.
        """
        plot_dist(
            self._X,
            self._Y,
            self.coefficients,
            self.subset(),
            self.normalised(),
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
            title (str, optional): Title of the plot. Defaults to "Response Distribution".
            decimals (int, optional): Number of decimals when writing the subset size. Defaults to 0.
            fig (Union[Figure, None], optional): Pyplot figure to plot on, if None then a new plot is created and shown. Defaults to None.
        """
        plot_dist_single(self._Y, self.subset(), None, title, decimals, fig)


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
            [np.ndarray, np.ndarray, float, Optional[np.ndarray]],
            Tuple[np.ndarray, float],
        ] = initialise_candidates,
        beta_max: float = 20,
        max_approx: float = 1.15,
        max_iterations: int = 300,
        debug: bool = False,
        num_threads: int = -1,
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
            X (np.ndarray): Data matrix.
            Y (np.ndarray): Vector of predictions.
            epsilon (float): Error tolerance.
            lambda1 (float, optional): L1 regularisation strength. Defaults to 0.
            lambda2 (float, optional): L2 regularisation strength. Defaults to 0.
            logit (bool, optional): Do a logit transformation on the Y vector, this is recommended opnly if Y consists of probabilities. Defaults to False.
            normalise (bool, optional): Should X and Y be normalised (note that epsilon will not be scaled). Defaults to False.
            initialisation (Callable[ [np.ndarray, np.ndarray, float, Optional[np.ndarray]], Tuple[np.ndarray, float] ], optional): Function that takes `(X, Y, epsilon, weight)` and gives an initial values for alpha and beta. Defaults to initialise_candidates.
            beta_max (float, optional): The final sigmoid steepness. Defaults to 20.
            max_approx (float, optional): Approximation ratio when selecting the next beta. Defaults to 1.15.
            max_iterations (int, optional): Maximum number of OWL-QN iterations. Defaults to 300.
            debug (bool, optional): Print debug statements each graduated optimisation step. Defaults to False.
            num_threads (int, optional): The number of threads to use for the optimisation. Defaults to -1.
        """
        assert epsilon > 0.0, "`epsilon` must be positive!"
        assert lambda1 >= 0.0, "`lambda1` must not be negative!"
        assert lambda2 >= 0.0, "`lambda2` must not be negative!"
        assert beta_max > 0.0, "`beta_max` must be positive!"
        assert max_approx > 1.0, "`max_approx` must be larger than 1.0!"
        assert max_iterations > 0, "`max_iterations` must be positive!"
        self.epsilon = epsilon
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.init_fn = initialisation
        self.beta_max = beta_max
        self.max_approx = max_approx
        self.max_iterations = max_iterations
        self.debug = debug
        X = np.array(X)
        Y = np.array(Y)
        if len(X.shape) == 1:
            X.shape = X.shape + (1,)
        assert X.shape[0] == Y.shape[0], "X and Y must have the same number of items"
        self._logit = logit
        self._normalise = normalise
        self._X = X
        self._Y = Y
        self._x = None
        self._y = None
        self._weight = None
        self._alpha = None
        self._coefficients = None
        # Preprocess data
        if logit:
            Y = limited_logit(Y)
        if normalise:
            X2, x_cols = remove_constant_columns(X)
            if X.shape[1] == X2.shape[1]:
                x_cols = None
            X, x_center, x_scale = normalise_robust(X2)
            if logit:
                (y_center, y_scale) = (0, 1)
            else:
                Y, y_center, y_scale = normalise_robust(Y)
            self._scale = DataScaling(x_center, x_scale, y_center, y_scale, x_cols)
        else:
            self._scale = None
        self._X2 = X
        self._Y2 = Y
        self.num_threads = num_threads
        check_threading_layer()

    def explain(
        self,
        x: Union[np.ndarray, int],
        y: Union[float, None] = None,
        weight: Optional[np.ndarray] = None,
        init: Union[None, np.ndarray, Tuple[np.ndarray, float]] = None,
    ) -> SliseExplainer:
        """Explain an outcome from a black box model

        Args:
            x (Union[np.ndarray, int]): Data item to explain, or an index to get the item from self.X
            y (Union[float, None], optional): Prediction to explain. If x is an index then this should be None (y is taken from self.Y). Defaults to None.
            weight (Optional[np.ndarray], optional): Weight vector for the data items. Defaults to None.
            init (Union[None, np.ndarray, Tuple[np.ndarray, float]], optional): Use this alpha (and beta) value instead of the initialisation function. Defaults to None.

        Returns:
            SliseExplainer: `self` (containing the explanation).
        """
        if weight is None:
            self._weight = None
        else:
            self._weight = np.array(weight)
            assert len(self._weight) == len(
                self._Y
            ), "Y and weight must have the same number of items!"
            assert np.all(self._weight >= 0.0), "Weights must not be negative!"
        if y is None:
            assert isinstance(x, int) and (
                0 <= x < self._Y.shape[0]
            ), "If y is None then x must be an integer index [0, len(Y)["
            self._y = self._Y[x]
            self._x = self._X[x, :]
            y = self._Y2[x]
            x = self._X2[x, :]
        else:
            x = np.atleast_1d(np.array(x))
            self._x = x
            self._y = y
            if self._logit:
                y = limited_logit(y)
            if self._normalise:
                x = self._scale.scale_x(x)
                y = self._scale.scale_y(y)
        X = self._X2 - x[None, :]
        Y = self._Y2 - y
        threads = set_threads(self.num_threads)
        if init is None:
            alpha, beta = self.init_fn(X, Y, self.epsilon, self._weight)
        else:
            alpha, beta = initialise_fixed(init, X, Y, self.epsilon, self._weight)
        alpha = graduated_optimisation(
            alpha=alpha,
            X=X,
            Y=Y,
            epsilon=self.epsilon,
            beta=beta,
            lambda1=self.lambda1,
            lambda2=self.lambda2,
            weight=self._weight,
            beta_max=self.beta_max,
            max_approx=self.max_approx,
            max_iterations=self.max_iterations,
            debug=self.debug,
        )
        set_threads(threads)
        alpha = np.concatenate(
            (y - np.sum(alpha * x, dtype=x.dtype, keepdims=True), alpha)
        )
        self._alpha = alpha
        if self._normalise:
            y = self._y
            if self._logit:
                y = limited_logit(y)
            alpha2 = self._scale.unscale_model(alpha)
            alpha2[0] = y - np.sum(self._x * alpha2[1:])
            self._coefficients = alpha2
        else:
            self._coefficients = alpha
        return self

    def get_params(self, normalised: bool = False) -> np.ndarray:
        """Get the explanation as the coefficients of a linear model (approximating the black box model).

        Args:
            normalised (bool, optional): If the data is normalised within SLISE, return a linear model fitting the normalised data. Defaults to False.

        Returns:
            np.ndarray: The coefficients of the linear model (the first scalar in the vector is the intercept).
        """
        warn("Use `coefficients` instead of `get_params().", SliseWarning)
        return self._alpha if normalised else self._coefficients

    @property
    def coefficients(self) -> np.ndarray:
        """Get the explanation as the coefficients of a linear model (approximating the black box model).

        Returns:
            np.ndarray: The coefficients of the linear model (the first scalar in the vector is the intercept).
        """
        if self._coefficients is None:
            warn("Fit an explanation before retrieving coefficients", SliseWarning)
        return self._coefficients

    def normalised(self, all_columns: bool = True) -> Optional[np.ndarray]:
        """Get coefficients for normalised data (if the data is normalised within SLISE).

        Args:
            all_columns (bool, optional): Add coefficients for constant columns. Defaults to True.

        Returns:
            Optional[np.ndarray]: The normalised coefficients or None.
        """
        if self._alpha is None:
            warn("Fit an explanation before retrieving coefficients", SliseWarning)
        if self._normalise:
            if all_columns:
                return add_constant_columns(self._alpha, self._scale.columns, True)
            else:
                return self._alpha
        else:
            return None

    @property
    def scaled_epsilon(self) -> float:
        """Espilon fitting unnormalised data (if the data is normalised).

        Returns:
            float: Scaled epsilon.
        """
        if self._normalise:
            return self.epsilon * self._scale.y_scale
        else:
            return self.epsilon

    def predict(self, X: Union[np.ndarray, None] = None) -> np.ndarray:
        """Use the approximating linear model to predict new outcomes.

        Args:
            X (Union[np.ndarray, None], optional): Sata matrix to predict, or None for using the fitted dataset. Defaults to None.

        Returns:
            np.ndarray: Prediction vector.
        """
        if X is None:
            Y = mat_mul_inter(self._X, self.coefficients)
        else:
            Y = mat_mul_inter(X, self.coefficients)
        if self._logit:
            Y = sigmoid(Y)
        return Y

    def score(
        self, X: Union[np.ndarray, None] = None, Y: Union[np.ndarray, None] = None
    ) -> float:
        """Calculate the loss. Lower is better and it should usually be negative (unless the regularisation is very (/too?) strong).

        Args:
            X (Union[np.ndarray, None], optional): Data matrix, or None for using the fitted dataset. Defaults to None.
            Y (Union[np.ndarray, None], optional): Response vector, or None for using the fitted dataset. Defaults to None.

        Returns:
            float: The loss.
        """
        if self._alpha is None:
            warn("Fit an explanation before calculating the score", SliseWarning)
        x = self._x
        y = self._y
        if self._logit:
            y = limited_logit(y)
        if self._normalise:
            x = self._scale.scale_x(x)
            y = self._scale.scale_y(y)
        if X is None or Y is None:
            X = self._X2
            Y = self._Y2
        else:
            if self._logit:
                Y = limited_logit(Y)
            if self._normalise:
                X = self._scale.scale_x(X)
                Y = self._scale.scale_y(Y)
        X = X - x[None, :]
        Y = Y - y
        return loss_sharp(
            self._alpha[1:],
            X,
            Y,
            self.epsilon,
            self.lambda1,
            self.lambda2,
            self._weight,
        )

    loss = score
    value = score

    def subset(
        self, X: Union[np.ndarray, None] = None, Y: Union[np.ndarray, None] = None
    ) -> np.ndarray:
        """Get the subset / neighbourhood used for the approximation (explanation).

        Args:
            X (Union[np.ndarray, None], optional): Data matrix, or None for using the fitted dataset. Defaults to None.
            Y (Union[np.ndarray, None], optional): Response vector, or None for using the fitted dataset. Defaults to None.

        Returns:
            np.ndarray: The subset as a boolean mask.
        """
        if X is None or Y is None:
            X = self._X
            Y = self._Y
        if self._logit:
            Y = limited_logit(Y)
        res = mat_mul_inter(X, self.coefficients) - Y
        return res ** 2 < self.scaled_epsilon ** 2

    def get_impact(
        self, normalised: bool = False, x: Union[None, np.ndarray] = None
    ) -> np.ndarray:
        """Get the "impact" of different variables on the outcome.
            The impact is the (normalised) model times the (normalised) item.

        Args:
            normalised (bool, optional): Return the normalised impact (if normalisation is used). Defaults to False.
            x (Union[None, np.ndarray], optional): The item to calculate the impact for (uses the explained item if None). Defaults to None.

        Returns:
            np.ndarray: The impact vector.
        """
        if x is None:
            x = self._x
        if normalised and self._normalise:
            x = add_constant_columns(self._scale.scale_x(x), self._scale.columns, False)
            return add_intercept_column(x) * self.coefficients
        else:
            return add_intercept_column(x) * self.coefficients

    def print(
        self,
        variables: Union[List[str], None] = None,
        classes: Union[List[str], None] = None,
        num_var: int = 10,
        decimals: int = 3,
    ):
        """Print the current explanation.

        Args:
            variables (Union[List[str], None], optional): Names of the (columns/) variables. Defaults to None.
            classes (Union[List[str], None], optional): Names of the classes, if explaining a classifier. Defaults to None.
            num_var (int, optional): Exclude zero weights if there are too many variables. Defaults to 10.
            decimals (int, optional): Precision to use for printing. Defaults to 3.
        """
        print_slise(
            self.coefficients,
            True,
            self.subset(),
            self.score(),
            self.scaled_epsilon,
            variables,
            "SLISE Explanation",
            decimals,
            num_var,
            unscaled=self._x,
            unscaled_y=self._y,
            impact=self.get_impact(False),
            scaled=None if self._scale is None else self._scale.scale_x(self._x, False),
            alpha=self.normalised(),
            scaled_impact=None if self._scale is None else self.get_impact(True),
            classes=classes,
            unscaled_preds=self._Y,
            logit=self._logit,
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
            title (str, optional): Title of the plot. Defaults to "SLISE Explanation".
            label_x (str, optional): x-axis label. Defaults to "x".
            label_y (str, optional): Y-axis label. Defaults to "y".
            decimals (int, optional): Number of decimals when writing numbers. Defaults to 3.
            fig (Union[Figure, None], optional): Pyplot figure to plot on, if None then a new plot is created and shown. Defaults to None.

        Raises:
            SliseException: If the data has too many dimensions.
        """
        plot_2d(
            self._X,
            self._Y,
            self.coefficients,
            self.scaled_epsilon,
            self._x,
            self._y,
            self._logit,
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
        """Plot the current explanation for a black and white image (e.g. MNIST).

        Args:
            width (int): Width of the image.
            height (int): Height of the image.
            saturated (bool, optional): Should the explanation be more saturated. Defaults to True.
            title (str, optional): Title of the plot. Defaults to "SLISE Explanation".
            classes (Union[List, str, None], optional): List of class names (first the negative, then the positive), or a single (positive) class name. Defaults to None.
            decimals (int, optional): Number of decimals to write. Defaults to 3.
            fig (Union[Figure, None], optional): Pyplot figure to plot on, if None then a new plot is created and shown. Defaults to None.
        """
        plot_image(
            self._x,
            self._y,
            self._Y,
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
            title (str, optional): Title of the plot. Defaults to "SLISE Explanation".
            variables (list, optional): Names for the variables. Defaults to None.
            decimals (int, optional): Number of decimals to write. Defaults to 3.
            fig (Union[Figure, None], optional): Pyplot figure to plot on, if None then a new plot is created and shown. Defaults to None.
        """
        plot_dist(
            self._X,
            self._Y,
            self.coefficients,
            self.subset(),
            self.normalised(),
            self._x,
            self._y,
            self.get_impact(False),
            self.get_impact(True) if self._normalise else None,
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
            title (str, optional): Title of the plot. Defaults to "Prediction Distribution".
            decimals (int, optional): Number of decimals when writing the subset size. Defaults to 0.
            fig (Union[Figure, None], optional): Pyplot figure to plot on, if None then a new plot is created and shown. Defaults to None.
        """
        plot_dist_single(self._Y, self.subset(), self._y, title, decimals, fig)
