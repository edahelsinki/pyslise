"""
    This script contains functions for plotting the results
"""

from typing import List, Union, Tuple
from warnings import warn
import numpy as np
from scipy.stats import gaussian_kde
from scipy.special import expit as sigmoid
from matplotlib import pyplot as plt
from matplotlib.pyplot import Axes, Figure
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.patches import Patch
from slise.utils import SliseException, SliseWarning, mat_mul_inter, limited_logit


# SLISE colors, for unified identity
SLISE_ORANGE = "#fda411"
SLISE_PURPLE = "#9887cb"
SLISE_DARKORANGE = "#e66101"
SLISE_DARKPURPLE = "#5e3c99"
SLISE_COLORMAP = LinearSegmentedColormap.from_list(
    "SLISE", [SLISE_DARKORANGE, SLISE_ORANGE, "#ffffff", SLISE_PURPLE, SLISE_DARKPURPLE]
)


def fill_column_names(
    names: Union[List[str], None] = None, amount: int = -1, intercept: bool = False
) -> list:
    """Make sure the list of column names is of the correct size

    Keyword Arguments:
        names {list} -- prefilled list of column names (default: {None})
        amount {int} -- the number of columns, including intercept (default: {-1})
        intercept {bool} -- is the first column an intercept column (default: {False})

    Returns:
        list -- list of column names
    """
    if amount < 1:
        return names
    if names is None:
        if intercept:
            return ["Intercept"] + ["Variable %d" % i for i in range(amount)]
        else:
            return ["Variable %d" % i for i in range(amount)]
    if len(names) > amount:
        warn("Too many column names given", SliseWarning)
        names = names[:amount]
    if len(names) < amount:
        warn("Too few column names given", SliseWarning)
        names = names + ["Variable %d" % i for i in range(len(names), amount)]
    if intercept:
        return ["Intercept"] + names
    else:
        return names


def fill_prediction_str(
    y: float, class_names: Union[List[str], None] = None, decimals: int = 3
) -> str:
    """Fill a string with the prediction meassage for explanations

    Arguments:
        y {float} -- the prediction

    Keyword Arguments:
        class_names {list} -- list of class names, if classification (default: {None})
        decimals {int} -- the decimal precision (default: {3})

    Returns:
        str -- the formatted message
    """
    if class_names is not None:
        if len(class_names) > 1:
            if y >= 0.0 and y <= 1.0:
                if y > 0.5:
                    return f"Predicted: {y*100:.{decimals}f}% {class_names[1]}"
                else:
                    return f"Predicted: {(1-y)*100:.{decimals}f}% {class_names[0]}"
            else:
                if y > 0:
                    return f"Predicted: {y:.{decimals}f} {class_names[1]}"
                else:
                    return f"Predicted: {-y:.{decimals}f} {class_names[0]}"
        else:
            if y >= 0.0 and y <= 1.0:
                return f"Predicted: {y*100:.{decimals}f}% {class_names[0]}"
            else:
                return f"Predicted: {y:.{decimals}f} {class_names[0]}"
    else:
        return f"Predicted: {y:.{decimals}f}"


def extended_limits(
    x: np.ndarray, extension: float = 0.05, steps: int = 2
) -> np.ndarray:
    min = np.min(x)
    max = np.max(x)
    diff = max - min
    if steps <= 2:
        return np.array([min - diff * extension, max + diff * extension])
    else:
        return np.linspace(min - diff * extension, max + diff * extension, steps)


def plot_2d(
    X: np.ndarray,
    Y: np.ndarray,
    alpha: np.ndarray,
    epsilon: float,
    x: Union[np.ndarray, None] = None,
    y: Union[float, None] = None,
    logit: bool = False,
    title: str = "SLISE for Robust Regression",
    label_x: str = "x",
    label_y: str = "y",
    decimals: int = 3,
    fig: Union[Axes, None] = None,
):
    """Plot the regression/explanation in a 2D scatter plot with a line for the regression model (and the explained item marked)

    Args:
        X (np.ndarray): data matrix
        Y (np.ndarray): response vector
        alpha (np.ndarray): regression model
        epsilon (float): error tolerance
        x (Union[np.ndarray, None], optional): explained item. Defaults to None.
        y (Union[float, None], optional): explained outcome. Defaults to None.
        logit (bool, optional): should Y be logit-transformed. Defaults to False.
        title (str, optional): plot title. Defaults to "SLISE for Robust Regression".
        label_x (str, optional): x-axis label. Defaults to "x".
        label_y (str, optional): y-axis label. Defaults to "y".
        decimals (int, optional): number of decimals when writing numbers. Defaults to 3.
        fig (Union[Axes, None], optional): Pyplot axes to plot on, if None then a new plot is created and shown. Defaults to None.

    Raises:
        SliseException: if the data has too many dimensions
    """
    if fig is None:
        fig = plt
    if X.size != Y.size:
        raise SliseException(f"Can only plot 1D data, |Y| = {Y.size} != {X.size} = |X|")
    x_limits = extended_limits(X, 0.03, 20 if logit else 2)
    y_limits = mat_mul_inter(x_limits[:, None], alpha)
    if logit:
        fig.fill_between(
            x_limits,
            sigmoid(y_limits + epsilon),
            sigmoid(y_limits - epsilon),
            color=SLISE_PURPLE + "33",
            label="Subset",
        )
        y_limits = sigmoid(y_limits)
    else:
        fig.fill_between(
            x_limits,
            y_limits + epsilon,
            y_limits - epsilon,
            color=SLISE_PURPLE + "33",
            label="Subset",
        )
    fig.plot(X.ravel(), Y, "o", color="black", label="Dataset")
    fig.plot(x_limits, y_limits, "-", color=SLISE_PURPLE, label="Model")
    if x is not None and y is not None:
        fig.plot(x, y, "o", color=SLISE_ORANGE, label="Explained Item")
    formula = ""
    if isinstance(alpha, float) or len(alpha) == 1:
        formula = f"{float(alpha):.{decimals}f} * {label_x}"
    elif np.abs(alpha[0]) > 1e-8:
        sign = "-" if alpha[1] < 0.0 else "+"
        formula = f"{alpha[0]:.{decimals}f} {sign} {abs(alpha[1]):.{decimals}f} $\\cdot$ {label_x}"
    else:
        formula = f"{alpha[1]:.{decimals}f} * {label_x}"
    if logit:
        formula = f"$\\sigma$({formula})"
    fig.legend()
    if plt == fig:
        fig.xlabel(label_x)
        fig.ylabel(label_y)
        fig.title(f"{title}: {label_y} = {formula}")
        plt.tight_layout()
        plt.show()
    else:
        fig.set_xlabel(label_x)
        fig.set_ylabel(label_y)
        fig.set_title(f"{title}: {label_y} = {formula}")


def get_explanation_order(
    alpha: np.ndarray, intercept: bool = True, min: int = 5
) -> (np.ndarray, np.ndarray):
    """
        Get the order in which to show the values in the plots
    """
    if intercept:
        order = np.argsort(alpha[1:]) + 1
        if len(order) > min:
            order = order[np.nonzero(alpha[order])]
        order = np.concatenate((order, np.zeros(1, order.dtype)))
    else:
        order = np.argsort(alpha)
        if len(order) > min:
            order = order[np.nonzero(alpha[order])]
    return np.flip(order)


def plot_dist(
    X: np.ndarray,
    Y: np.ndarray,
    alpha: np.ndarray,
    subset: np.ndarray,
    x: Union[np.ndarray, None] = None,
    y: Union[float, None] = None,
    impact: Union[np.ndarray, None] = None,
    title: str = "SLISE Explanation",
    column_names: list = None,
    fig: Union[Figure, None] = None,
):
    """Plot the SLISE result with density distributions for the dataset and barplot for the model

    Args:
        X (np.ndarray): data matrix
        Y (np.ndarray): response vector
        alpha (np.ndarray): linear model
        subset (np.ndarray): selected subset
        x (Union[np.ndarray, None], optional): the explained item (if it is an explanation). Defaults to None.
        y (Union[float, None], optional): the explained outcome (if it is an explanation). Defaults to None.
        impact (Union[np.ndarray, None], optional): impact vector (scaled x*alpha), if available. Defaults to None.
        title (str, optional): title of the plot. Defaults to "SLISE Explanation".
        column_names (list, optional): names for the variables. Defaults to None.
        fig (Union[Axes, None], optional): Pyplot axes to plot on, if None then a new plot is created and shown. Defaults to None.
    """
    # Values and order
    order = get_explanation_order(alpha, True)
    column_names = fill_column_names(column_names, X.shape[1], True)
    if len(alpha) == X.shape[1]:
        alpha = np.concatenate((np.zeros(1, alpha.dtype), alpha))
        column_names[0] = ""
    bins = max(10, min(50, len(Y) // 20))
    alpha = alpha[order]
    if impact is not None:
        impact = impact[order] / np.max(np.abs(impact)) * np.max(np.abs(alpha))
    column_names = [column_names[i] for i in order]
    # Figures:
    plot = False
    if isinstance(fig, Figure):
        axs = fig.subplots(len(order), 2)
    else:
        plot = True
        fig, axs = plt.subplots(len(order), 2)
    fig.suptitle(title)
    # Density plots
    subset_size = subset.mean()

    def fill_density(ax, X, x, n):
        kde1 = gaussian_kde(X, 0.25)
        kde2 = gaussian_kde(X[subset], 0.25)
        lim = extended_limits(X, 0.1, 100)
        ax.plot(lim, kde1(lim), color="black", label="Dataset")
        ax.plot(lim, kde2(lim) * subset_size, color=SLISE_PURPLE, label="Subset")
        if x is not None:
            ax.relim()
            ax.vlines(x, *ax.get_ylim(), color=SLISE_ORANGE, label="Explained Item")
        ax.set_yticks([])
        ax.set_ylabel(
            n, rotation=0, horizontalalignment="right", verticalalignment="center"
        )

    fill_density(axs[0, 0], Y, y, "Response")
    axs[0, 0].legend()
    axs[0, 0].set_title("Dataset Distribution")
    for i, k, n in zip(range(1, len(order)), order[1:] - 1, column_names[1:]):
        fill_density(axs[i, 0], X[:, k], x[k] if x is not None else None, n)
    # Bar plots
    gs = axs[0, 1].get_gridspec()
    for ax in axs[:, 1]:
        ax.remove()
    axbig = fig.add_subplot(gs[:, 1])
    if x is None or y is None:
        axbig.set_title("Linear Model")
    else:
        axbig.set_title("Explanation")
    ticks = np.arange(len(column_names))
    axbig.set_yticks(ticks)
    axbig.set_yticklabels(column_names)
    axbig.set_ylim(bottom=ticks[0] - 0.45, top=ticks[-1] + 0.45)
    axbig.invert_yaxis()
    if impact is None:
        column_color = [SLISE_ORANGE if v < 0 else SLISE_PURPLE for v in alpha]
        axbig.barh(ticks, alpha, color=column_color)
    else:
        axbig.barh(
            ticks - 0.2, alpha, height=0.35, color=SLISE_PURPLE, label="Linear Model"
        )
        axbig.barh(ticks + 0.2, impact, height=0.35, color=SLISE_ORANGE, label="Impact")
        axbig.legend()
    axbig.yaxis.tick_right()
    # meta
    fig.tight_layout()
    if plot:
        plt.show()


def plot_explanation_image(
    x: np.ndarray,
    y: float,
    alpha: np.ndarray,
    width: int,
    height: int,
    # scaler: DataScaler,
    class_names: list = None,
    decimals: int = 2,
):
    """Plot the current explanation for a black and white image (MNIST like)

    Arguments:
        x {np.ndarray} -- the explained image
        y {float} -- the explained prediction
        alpha {np.ndarray} -- the explanation
        width {int} -- the width of the image
        height {int} -- the height of the image
        scaler {DataScaler} -- scaler used to unscale the data

    Keyword Arguments:
        class_names {str or list} -- the names of the class (str) / classes (list), if explaining a classifier (default: {None})
        decimals {int} -- the precision to use for printing (default: {2})
    """
    alpha = scaler.unscale_model(alpha)[1:]
    alpha.shape = (width, height)
    alpha = alpha.T
    x = scaler.unscale(x)[0]
    x.shape = (width, height)
    x = x.T
    alpha = sigmoid(alpha * (4 / np.max(np.abs(alpha))))
    y = scaler.unscale(None, y)[1]
    if isinstance(class_names, str):
        class_names = ("not " + class_names, class_names)
    plt.imshow(
        alpha,
        interpolation="none",
        cmap=SLISE_COLORMAP,
        norm=Normalize(vmin=-0.1, vmax=1.1),
    )
    plt.contour(
        range(height),
        range(width),
        x,
        levels=[np.median(x) * 0.5 + np.mean(x) * 0.5],
        colors="black",
    )
    plt.xticks([])
    plt.yticks([])
    plt.title(
        "SLISE Explanation   |   " + fill_prediction_str(y, class_names, decimals)
    )
    if class_names is not None:
        plt.legend(
            (Patch(facecolor=SLISE_ORANGE), Patch(facecolor=SLISE_PURPLE)),
            class_names[:2],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.01),
            ncol=2,
        )
    plt.tight_layout()
    plt.show()
