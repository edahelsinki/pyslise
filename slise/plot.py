"""
    This script contains functions for plotting the results
"""

from collections import OrderedDict
from typing import List, Union, Tuple
from warnings import warn
import numpy as np
from scipy.stats import gaussian_kde
from scipy.special import expit as sigmoid
from matplotlib import pyplot as plt
from matplotlib.pyplot import Figure
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
BW_COLORMAP = LinearSegmentedColormap.from_list("BW", ["black", "white"])


def fill_column_names(
    names: Union[List[str], None] = None, amount: int = -1, intercept: bool = False
) -> List[str]:
    """Make sure the list of column names is of the correct size

    Args:
        names (Union[List[str], None], optional): prefilled list of column names. Defaults to None.
        amount (int, optional): the number of columns. Defaults to -1.
        intercept (bool, optional): should an intercept column be added. Defaults to False.

    Returns:
        List[str]: list of column names
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
    y: float,
    Y: Union[np.ndarray, None] = None,
    classes: Union[List[str], str, None] = None,
    decimals: int = 3,
) -> str:
    """Create a string describing the prediction

    Args:
        y (float): the prediction
        Y (Union[np.ndarray, None]): vector of predictions (used to guess if the predictions are probabilities). Defaults to None.
        classes (Union[List[str], str, None], optional): list of class names (starting with the negative class), or singular class name. Defaults to None.
        decimals (int, optional): how many decimals hsould be written. Defaults to 3.

    Returns:
        str: description of prediction
    """
    if classes is not None:
        prob = Y is not None and (0 <= Y.min() < 0.5) and (0.5 < Y.max() <= 1)
        if isinstance(classes, str):
            if prob:
                return f"Predicted: {y*100:.{decimals}f}% {classes[0]}"
            else:
                return f"Predicted: {y:.{decimals}f} {classes}"
        else:
            if prob:
                if y > 0.5:
                    return f"Predicted: {y*100:.{decimals}f}% {classes[1]}"
                else:
                    return f"Predicted: {(1-y)*100:.{decimals}f}% {classes[0]}"
            else:
                if y > 0:
                    return f"Predicted: {y:.{decimals}f} {classes[1]}"
                else:
                    return f"Predicted: {-y:.{decimals}f} {classes[0]}"
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


def get_explanation_order(
    alpha: np.ndarray, intercept: bool = True, min: int = 5, th=1e-6
) -> (np.ndarray, np.ndarray):
    """
        Get the order in which to show the values in the plots
    """
    if intercept:
        order = np.argsort(alpha[1:]) + 1
        if len(order) > min:
            order = order[np.nonzero(alpha[order])]
            if len(order) > min:
                order = order[np.abs(alpha[order]) > np.max(np.abs(alpha)) * th]
        order = np.concatenate((order, np.zeros(1, order.dtype)))
    else:
        order = np.argsort(alpha)
        if len(order) > min:
            order = order[np.nonzero(alpha[order])]
            if len(order) > min:
                order = order[np.abs(alpha[order]) > np.max(np.abs(alpha)) * th]
    return np.flip(order)


def print_slise(
    coefficients: np.ndarray,
    intercept: bool,
    subset: np.ndarray,
    loss: float,
    epsilon: float,
    variables: Union[List[str], None] = None,
    title: str = "SLISE",
    decimals: int = 3,
    num_var: int = 10,
    unscaled: Union[None, np.ndarray] = None,
    unscaled_y: Union[None, float] = None,
    impact: Union[None, np.ndarray] = None,
    scaled: Union[None, np.ndarray] = None,
    alpha: Union[None, np.ndarray] = None,
    scaled_impact: Union[None, np.ndarray] = None,
    classes: Union[List[str], None] = None,
    unscaled_preds: Union[None, np.ndarray] = None,
    logit: bool = False,
):
    """Print the results from SLISE

    Args:
        coefficients (np.ndarray): the linear model coefficients
        intercept (bool): is the first coefficient an intercept
        subset (np.ndarray): subset mask
        loss (float): SLISE loss
        epsilon (float): (unscaled) error tolerance
        variables (Union[List[str], None], optional): variable names. Defaults to None.
        title (str, optional): title to print first. Defaults to "SLISE".
        decimals (int, optional): number of decimals to print. Defaults to 3.
        num_var (int, optional): exclude zero weights if there are too many variables. Defaults to 10.
        unscaled (Union[None, np.ndarray], optional): unscaled x (explained item). Defaults to None.
        unscaled_y (Union[None, float], optional): unscaled y (explained outcome). Defaults to None.
        impact (Union[None, np.ndarray], optional): unscaled impact (coefficients * x). Defaults to None.
        scaled (Union[None, np.ndarray], optional): scaled x (explained item). Defaults to None.
        alpha (Union[None, np.ndarray], optional): scaled model. Defaults to None.
        scaled_impact (Union[None, np.ndarray], optional): scaled impact (alpha * scaled_x). Defaults to None.
        classes (Union[List[str], None], optional): class names (if applicable). Defaults to None.
        unscaled_preds (Union[None, np.ndarray], optional): unscaled resonse (Y-vector). Defaults to None.
        logit (bool, optional): a logit transformation has been applied. Defaults to False.
    """
    rows = OrderedDict()
    rows["Variable Names:    "] = fill_column_names(
        variables, len(coefficients) - intercept, intercept
    )
    if unscaled is not None:
        rows["Explained Item:"] = [""] + ["%%.%df" % decimals % a for a in unscaled]
        rows["Model Weights:"] = ["%%.%df" % decimals % a for a in coefficients]
    else:
        rows["Coefficients:"] = ["%%.%df" % decimals % a for a in coefficients]
    if impact is not None:
        rows["Prediction Impact:"] = ["%%.%df" % decimals % a for a in impact]
    if scaled is not None:
        rows["Normalised Item:"] = [""] + ["%%.%df" % decimals % a for a in scaled]
    if alpha is not None:
        rows["Normalised Weights:"] = ["%%.%df" % decimals % a for a in alpha]
    if scaled_impact is not None:
        rows["Normalised Impact:"] = ["%%.%df" % decimals % a for a in scaled_impact]
    col_len = [
        max(8, *vs) + 1
        for vs in zip(*(tuple(len(v) for v in vs) for vs in rows.values()))
    ]
    if len(coefficients) > num_var:
        col_len = [l if c != 0 else 0 for l, c in zip(col_len, coefficients)]
    lab_len = max(len(l) for l in rows)
    if title:
        print(title)
    if unscaled_y is not None:
        print(fill_prediction_str(unscaled_y, unscaled_preds, classes, decimals))
    for k in rows:
        print(
            f"{k:<{lab_len}}",
            " ".join([f"{s:>{c}}" for s, c in zip(rows[k], col_len) if c > 0]),
        )
    loss = f"{loss:.{decimals}f}"
    epsilon = f"{epsilon:.{decimals}f}"
    subsize = f"{subset.mean():.{decimals}f}"
    col_len = max(len(loss), len(epsilon), len(subsize), 8)
    print(f"Loss:          {loss   :>{col_len}}")
    print(f"Subset:        {subsize:>{col_len}}")
    print(f"Epsilon:       {epsilon:>{col_len}}")
    if logit and unscaled_preds is not None:
        if isinstance(classes, list) and len(classes) == 2:
            print(
                f"Class Balance: {(unscaled_preds[subset] > 0.5).mean() * 100:>.{decimals}f}% {classes[0]} | {(unscaled_preds[subset] < 0.5).mean() * 100:>.{decimals}f}% {classes[1]}"
            )
        else:
            print(
                f"Class Balance: {(unscaled_preds[subset] > 0.5).mean() * 100:>.{decimals}f}% | {(unscaled_preds[subset] < 0.5).mean() * 100:>.{decimals}f}%"
            )


def plot_2d(
    X: np.ndarray,
    Y: np.ndarray,
    model: np.ndarray,
    epsilon: float,
    x: Union[np.ndarray, None] = None,
    y: Union[float, None] = None,
    logit: bool = False,
    title: str = "SLISE for Robust Regression",
    label_x: str = "x",
    label_y: str = "y",
    decimals: int = 3,
    fig: Union[Figure, None] = None,
):
    """Plot the regression/explanation in a 2D scatter plot with a line for the regression model (and the explained item marked)

    Args:
        X (np.ndarray): data matrix
        Y (np.ndarray): response vector
        model (np.ndarray): regression model
        epsilon (float): error tolerance
        x (Union[np.ndarray, None], optional): explained item. Defaults to None.
        y (Union[float, None], optional): explained outcome. Defaults to None.
        logit (bool, optional): should Y be logit-transformed. Defaults to False.
        title (str, optional): plot title. Defaults to "SLISE for Robust Regression".
        label_x (str, optional): x-axis label. Defaults to "x".
        label_y (str, optional): y-axis label. Defaults to "y".
        decimals (int, optional): number of decimals when writing numbers. Defaults to 3.
        fig (Union[Figure, None], optional): Pyplot figure to plot on, if None then a new plot is created and shown. Defaults to None.

    Raises:
        SliseException: if the data has too many dimensions
    """
    if fig is None:
        plot = True
        fig, ax = plt.subplots()
    else:
        ax = fig.subplots()
        plot = False
    if X.size != Y.size:
        raise SliseException(f"Can only plot 1D data, |Y| = {Y.size} != {X.size} = |X|")
    x_limits = extended_limits(X, 0.03, 20 if logit else 2)
    y_limits = mat_mul_inter(x_limits[:, None], model)
    if logit:
        ax.fill_between(
            x_limits,
            sigmoid(y_limits + epsilon),
            sigmoid(y_limits - epsilon),
            color=SLISE_PURPLE + "33",
            label="Subset",
        )
        y_limits = sigmoid(y_limits)
    else:
        ax.fill_between(
            x_limits,
            y_limits + epsilon,
            y_limits - epsilon,
            color=SLISE_PURPLE + "33",
            label="Subset",
        )
    ax.plot(X.ravel(), Y, "o", color="black", label="Dataset")
    if x is not None and y is not None:
        ax.plot(x_limits, y_limits, "-", color=SLISE_PURPLE, label="Model")
        ax.plot(x, y, "o", color=SLISE_ORANGE, label="Explained Item")
    else:
        ax.plot(x_limits, y_limits, "-", color=SLISE_ORANGE, label="Model")
    formula = ""
    if isinstance(model, float) or len(model) == 1:
        formula = f"{float(model):.{decimals}f} * {label_x}"
    elif np.abs(model[0]) > 1e-8:
        sign = "-" if model[1] < 0.0 else "+"
        formula = f"{model[0]:.{decimals}f} {sign} {abs(model[1]):.{decimals}f} $\\cdot$ {label_x}"
    else:
        formula = f"{model[1]:.{decimals}f} * {label_x}"
    if logit:
        formula = f"$\\sigma$({formula})"
    ax.legend()
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.set_title(f"{title}: {label_y} = {formula}")
    fig.tight_layout()
    if plot:
        plt.show()


def plot_dist(
    X: np.ndarray,
    Y: np.ndarray,
    model: np.ndarray,
    subset: np.ndarray,
    alpha: Union[np.ndarray, None] = None,
    x: Union[np.ndarray, None] = None,
    y: Union[float, None] = None,
    impact: Union[np.ndarray, None] = None,
    norm_impact: Union[np.ndarray, None] = None,
    title: str = "SLISE Explanation",
    variables: list = None,
    decimals: int = 3,
    fig: Union[Figure, None] = None,
):
    """Plot the SLISE result with density distributions for the dataset and barplot for the model

    Args:
        X (np.ndarray): data matrix
        Y (np.ndarray): response vector
        model (np.ndarray): linear model
        subset (np.ndarray): selected subset
        alpha (Union[np.ndarray, None]): scaled model. Defaults to None.
        x (Union[np.ndarray, None], optional): the explained item (if it is an explanation). Defaults to None.
        y (Union[float, None], optional): the explained outcome (if it is an explanation). Defaults to None.
        impact (Union[np.ndarray, None], optional): impact vector (unscaled x*alpha), if available. Defaults to None.
        norm_impact (Union[np.ndarray, None], optional): impact vector (scaled x*alpha), if available. Defaults to None.
        title (str, optional): title of the plot. Defaults to "SLISE Explanation".
        variables (list, optional): names for the (columns/) variables. Defaults to None.
        decimals (int, optional): number of decimals when writing numbers. Defaults to 3.
        fig (Union[Figure, None], optional): Pyplot figure to plot on, if None then a new plot is created and shown. Defaults to None.
    """
    # Values and order
    variables = fill_column_names(variables, X.shape[1], True)
    if alpha is None:
        noalpha = True
        alpha = model
    else:
        noalpha = False
    if len(model) == X.shape[1]:
        model = np.concatenate((np.zeros(1, model.dtype), model))
        alpha = np.concatenate((np.zeros(1, model.dtype), alpha))
        variables[0] = ""
    order = get_explanation_order(np.abs(alpha), True)
    model = model[order]
    alpha = alpha[order]
    if impact is not None:
        impact = impact[order]
    if norm_impact is not None:
        norm_impact = norm_impact[order]
    variables = [variables[i] for i in order]
    subsize = subset.mean()

    # Figures:
    if isinstance(fig, Figure):
        plot = False
        axs = fig.subplots(len(order), 2, squeeze=False)
    else:
        plot = True
        fig, axs = plt.subplots(len(order), 2, squeeze=False)
    fig.suptitle(title)

    # Density plots

    def fill_density(ax, X, x, n):
        if np.var(X) == 0:
            X = np.random.normal(X[0], 1e-8, len(X))
        kde1 = gaussian_kde(X, 0.2)
        kde2 = gaussian_kde(X[subset], 0.2)
        lim = extended_limits(X, 0.1, 100)
        ax.plot(lim, kde1(lim), color="black", label="Dataset")
        ax.plot(
            lim,
            kde2(lim) * subsize,
            color=SLISE_PURPLE,
            label=f"Subset: {subsize * 100:.0f}%",
        )
        if x is not None:
            ax.relim()
            ax.vlines(x, *ax.get_ylim(), color=SLISE_ORANGE, label="Explained Item")
        ax.set_yticks([])
        ax.set_ylabel(
            n, rotation=0, horizontalalignment="right", verticalalignment="center"
        )

    if x is None and y is None:
        fill_density(axs[0, 0], Y, y, "Response")
    else:
        fill_density(axs[0, 0], Y, y, "Prediction")
    axs[0, 0].legend()
    axs[0, 0].set_title("Dataset Distribution")
    for i, k, n in zip(range(1, len(order)), order[1:] - 1, variables[1:]):
        fill_density(axs[i, 0], X[:, k], x[k] if x is not None else None, n)

    # Bar plots
    def text(x, y, v):
        if v != 0:
            axbig.text(
                x,
                y,
                f"{v:.{decimals}f}",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.75),
            )

    gs = axs[0, 1].get_gridspec()
    for ax in axs[:, 1]:
        ax.remove()
    axbig = fig.add_subplot(gs[:, 1])
    if x is None or y is None:
        axbig.set_title("Linear Model")
    else:
        axbig.set_title("Explanation")
    ticks = np.arange(len(variables))
    axbig.set_yticks(ticks)
    axbig.set_yticklabels(variables)
    axbig.set_ylim(bottom=ticks[0] - 0.45, top=ticks[-1] + 0.45)
    axbig.invert_yaxis()
    if impact is None and noalpha:
        column_color = [SLISE_ORANGE if v < 0 else SLISE_PURPLE for v in alpha]
        axbig.barh(ticks, alpha, color=column_color)
        for y, v in zip(ticks, model):
            text(0, y, v)
    elif impact is None and not noalpha:
        axbig.barh(
            ticks - 0.2,
            model / np.max(np.abs(model)),
            height=0.35,
            color=SLISE_PURPLE,
            label="Coefficients",
        )
        axbig.barh(
            ticks + 0.2,
            alpha / np.max(np.abs(alpha)),
            height=0.35,
            color=SLISE_ORANGE,
            label="Normalised",
        )
        for y, a, m in zip(ticks, alpha, model):
            text(0, y, m)
            text(0, y, a)
        axbig.set_xticks([])
        axbig.legend()
    elif norm_impact is None:
        axbig.barh(
            ticks[1:] - 0.2,
            model[1:] / np.max(np.abs(model)),
            height=0.35,
            color=SLISE_PURPLE,
            label="Linear Model",
        )
        axbig.barh(
            ticks[0], model[0] / np.max(np.abs(model)), height=0.35, color=SLISE_PURPLE,
        )
        axbig.barh(
            ticks[1:] + 0.2,
            impact[1:] / np.max(np.abs(impact[1:])),
            height=0.35,
            color=SLISE_ORANGE,
            label="Prediction Impact",
        )
        for y, a, m in zip(ticks, impact, model):
            if y == ticks[0]:
                text(0, y, m)
                continue
            text(0, y - 0.2, m)
            text(0, y + 0.2, a)
        axbig.set_xticks([])
        axbig.legend()
    else:
        axbig.barh(
            ticks[1:] - 0.33,
            model[1:] / np.max(np.abs(model)),
            height=0.2,
            color=SLISE_PURPLE,
            label="Linear Model",
        )
        axbig.barh(
            ticks[0] - 0.11,
            model[0] / np.max(np.abs(model)),
            height=0.2,
            color=SLISE_PURPLE,
        )
        axbig.barh(
            ticks[1:] - 0.11,
            alpha[1:] / np.max(np.abs(alpha)),
            height=0.2,
            color=SLISE_DARKPURPLE,
            label="Normalised Model",
        )
        axbig.barh(
            ticks[0] + 0.11,
            alpha[0] / np.max(np.abs(alpha)),
            height=0.2,
            color=SLISE_DARKPURPLE,
        )
        axbig.barh(
            ticks[1:] + 0.11,
            impact[1:] / np.max(np.abs(impact[1:])),
            height=0.2,
            color=SLISE_ORANGE,
            label="Prediction Impact",
        )
        axbig.barh(
            ticks[1:] + 0.33,
            norm_impact[1:] / np.max(np.abs(norm_impact[1:])),
            height=0.2,
            color=SLISE_DARKORANGE,
            label="Normalised Impact",
        )
        for y, i1, i2, m1, m2 in zip(ticks, impact, norm_impact, model, alpha):
            if y == ticks[0]:
                text(0, y - 0.11, m1)
                text(0, y + 0.11, m2)
                continue
            text(0, y - 0.33, m1)
            text(0, y - 0.11, m2)
            text(0, y + 0.11, i1)
            text(0, y + 0.33, i2)
        axbig.set_xticks([])
        axbig.legend()
    axbig.yaxis.tick_right()

    # Meta:
    fig.tight_layout()
    if plot:
        plt.show()


def plot_image(
    x: np.ndarray,
    y: float,
    Y: np.ndarray,
    model: np.ndarray,
    width: int,
    height: int,
    saturated: bool = True,
    title: str = "SLISE Explanation",
    classes: Union[List, str, None] = None,
    decimals: int = 3,
    fig: Union[Figure, None] = None,
):
    """Plot an explanation for a black and white image (e.g. MNIST)

    Args:
        x (np.ndarray): the explained item
        y (float): the explained outcome
        Y (np.ndarray): dataset response vector (used for guessing prediction formatting)
        model (np.ndarray): the approximating model
        width (int): the width of the image
        height (int): the height of the image
        saturated (bool, optional): should the explanation be more saturated. Defaults to True.
        title (str, optional): title of the plot. Defaults to "SLISE Explanation".
        classes (Union[List, str, None], optional): list of class names (first the negative, then the positive), or a single (positive) class name. Defaults to None.
        decimals (int, optional): the number of decimals to write. Defaults to 3.
        fig (Union[Figure, None], optional): Pyplot figure to plot on, if None then a new plot is created and shown. Defaults to None.
    """
    intercept = model[0]
    model = model[1:]
    model.shape = (width, height)
    x.shape = (width, height)
    if saturated:
        model = sigmoid(model * (4 / np.max(np.abs(model))))
    if fig is None:
        fig, [ax1, ax2] = plt.subplots(1, 2)
        plot = True
    else:
        [ax1, ax2] = fig.subplots(1, 2)
        plot = False
    fig.suptitle(title)
    # Image
    ax1.imshow(x, cmap=BW_COLORMAP)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title("Explained Item")
    ax1.set_xlabel(fill_prediction_str(y, Y, classes, decimals))
    # Explanation Image
    ax2.imshow(
        model,
        interpolation="none",
        cmap=SLISE_COLORMAP,
        norm=Normalize(vmin=-0.1, vmax=1.1),
    )
    ax2.contour(range(height), range(width), x, levels=1, colors="#00000033")
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title("Explanation")
    if classes is None:
        classes = ["Negative", "Positive"]
    elif isinstance(classes, str):
        classes = ["Not " + classes, classes]
    ax2.legend(
        (Patch(facecolor=SLISE_ORANGE), Patch(facecolor=SLISE_PURPLE)),
        classes[:2],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=2,
    )
    fig.tight_layout()
    if plot:
        plt.show()


def plot_dist_single(
    data: np.ndarray,
    subset: np.ndarray,
    item: Union[float, None] = None,
    title: str = "Response Distribution",
    decimals: int = 0,
    fig: Union[Figure, None] = None,
):
    """Plot a density distributions for a single vector of the dataset

    Args:
        data (np.ndarray): vector
        subset (np.ndarray): selected subset
        item (Union[np.ndarray, None], optional): the explained item (if it is an explanation). Defaults to None.
        title (str, optional): title of the plot. Defaults to "Response Distribution".
        decimals (int, optional): number of decimals when writing the subset size. Defaults to 0.
        fig (Union[Figure, None], optional): Pyplot figure to plot on, if None then a new plot is created and shown. Defaults to None.
    """
    subsize = subset.mean()
    if isinstance(fig, Figure):
        ax = fig.subplots(1, 1)
        plot = False
    else:
        fig, ax = plt.subplots(1, 1)
        plot = True
    ax.set_title(title)
    kde1 = gaussian_kde(data, 0.2)
    kde2 = gaussian_kde(data[subset], 0.2)
    lim = extended_limits(data, 0.1, 100)
    ax.plot(lim, kde1(lim), color="black", label="Dataset")
    ax.plot(
        lim,
        kde2(lim) * subsize,
        color=SLISE_PURPLE,
        label=f"Subset: {subsize * 100:.{decimals}f}%",
    )
    if item is not None:
        ax.relim()
        ax.vlines(item, *ax.get_ylim(), color=SLISE_ORANGE, label="Explained Item")
    ax.set_yticks([])
    ax.legend()
    if plot:
        plt.show()
