"""
    This script contains functions for plotting the results
"""

from warnings import warn
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import expit as sigmoid
from slise.data import DataScaler, local_into, local_model, mat_mul_with_intercept
from slise.utils import SliseWarning


# SLISE colors, for unified identity
SLISE_ORANGE = "#fda411"
SLISE_PURPLE = "#9887cb"
SLISE_DARKORANGE = "#e66101"
SLISE_DARKPURPLE = "#5e3c99"


def fill_column_names(names: list = None, amount: int = -1, intercept: bool = False) -> list:
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
            return ["Intercept"] + ["Col %d"%i for i in range(amount-1)]
        else:
            return ["Col %d"%i for i in range(amount)]
    elif len(names) == amount:
        if intercept:
            warn("No room to add the name for the intercept column", SliseWarning)
        return names
    elif len(names) == amount - 1 and intercept:
        return ["Intercept"] + names
    elif len(names) > amount:
        warn("Too many column names given", SliseWarning)
        return names[:amount]
    else:
        warn("Too few column names given", SliseWarning)
        if intercept:
            names = ["Intercept"] + names
        return names + ["Col %d"%i for i in range(len(names), amount)]

def fill_prediction_str(y: float, class_names: list = None, decimals: int = 3) -> str:
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
                    return f"Predicted: {y*100:.{decimals}f} % {class_names[1]}"
                else:
                    return f"Predicted: {(1-y)*100:.{decimals}f} % {class_names[0]}"
            else:
                if y > 0:
                    return f"Predicted: {y:.{decimals}f} {class_names[1]}"
                else:
                    return f"Predicted: {-y:.{decimals}f} {class_names[0]}"
        else:
            if y >= 0.0 and y <= 1.0:
                return f"Predicted: {y*100:.{decimals}f} % {class_names[0]}"
            else:
                return f"Predicted: {y:.{decimals}f} {class_names[0]}"
    else:
        return f"Predicted: {y:.{decimals}f}"


def plot_regression_2D(X: np.ndarray, Y: np.ndarray, alpha: np.ndarray, epsilon: float,
        scaler: DataScaler, decimals: int = 2):
    """Plot 1D data in a 2D scatter plot, with a line for the regression model

    Arguments:
        X {np.ndarray} -- the dataset (data matrix)
        Y {np.ndarray} -- the dataset (prediciton vector)
        alpha {np.ndarray} -- the regression model
        epsilon {float} -- the error tolerance
        scaler {DataScaler} -- scaler used to unscale the data

    Keyword Arguments:
        decimals {int} -- the number of decimals for the axes (default: {2})

    Raises:
        Exception: if the data is not 1D (intercept allowed)
    """
    if scaler.intercept:
        X = X[:, 1].ravel()
    else:
        X = X.ravel()
    if len(X) != len(Y):
        raise Exception(f"Can only plot 1D data (len(Y) != len(X): {len(Y)} != {len(X)})")
    XL = np.array((X.min(), X.max()))
    ext = (XL[1] - XL[0]) * 0.02
    XL = XL + [-ext, ext]
    YL = mat_mul_with_intercept(XL, alpha)
    XL = XL.ravel()
    plt.fill_between(XL, YL+epsilon, YL-epsilon, color=SLISE_PURPLE+"33")
    plt.plot(XL, YL, "-", color=SLISE_PURPLE)
    plt.plot(X, Y, 'o', color=SLISE_ORANGE)
    ticks = plt.xticks()[0]
    plt.xticks(ticks, [f"{v:.{decimals}f}" for v in scaler.scaler_x.unscale(ticks)])
    ticks = plt.yticks()[0]
    plt.yticks(ticks, [f"{v:.{decimals}f}" for v in scaler.scaler_y.unscale(ticks)])
    plt.show()

def get_explanation_order(alpha: np.ndarray, mask: np.ndarray) -> (np.ndarray, np.ndarray):
    """
        Get the order in which to show the values in the plots
    """
    order = np.argsort(alpha)
    first = np.where(order == 0)[0][0]
    order = np.concatenate((order[:first], order[(1 + first):], [0]))
    order_outer = np.concatenate(([0], np.atleast_1d(mask + 1)))[order]
    return order, order_outer

def plot_explanation_tabular(x: np.ndarray, y: float, alpha: np.ndarray, scaler: DataScaler,
        column_names: list = None, class_names: list = None, decimals: int = 3):
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

    Arguments:
        x {np.ndarray} -- the explained item
        y {float} -- the explained prediction
        alpha {np.ndarray} -- the explanation
        scaler {DataScaler} -- scaler used to unscale the data

    Keyword Arguments:
        column_names {list} -- the names of the features/variables (default: {None})
        class_names {str or list} -- the names of the class (str) / classes (list), if explaining a classifier (default: {None})
        decimals {int} -- the precision to use for printing (default: {3})
    """
    # Values
    scaled_x = np.concatenate(([1.0], x))
    impact = scaled_x * alpha
    impact /= np.sum(np.abs(impact))
    scaled_x[0] = 0.0
    unscaled_x = np.concatenate(([0.0], scaler.scaler_x.unscale(x)))
    column_names = fill_column_names(column_names, len(unscaled_x), True)
    if isinstance(class_names, str):
        class_names = ("not "+class_names, class_names)
    # Sorting
    order, order_outer = get_explanation_order(alpha, scaler.scaler_x.mask)
    scaled_x = scaled_x[order]
    alpha = alpha[order]
    impact = impact[order]
    unscaled_x = unscaled_x[order_outer]
    column_names = [column_names[i] for i in order_outer]
    # Plot title
    plt.figure()
    plt.suptitle("SLISE Explanation   |   " + fill_prediction_str(scaler.scaler_y.unscale(y), class_names, decimals))
    # Plot value
    plt.subplot(1, 3, 1)
    val_col_nam = [f"{n}\n{x:.{decimals}f}" for n, x in zip(column_names, unscaled_x)]
    val_col_nam[-1] = ""
    plt.barh(val_col_nam, scaled_x, color="grey")
    plt.title("\n\nExplained Item")
    if not np.allclose(unscaled_x, scaled_x):
        plt.xticks([0.0], ["Scaled Mean"])
    # Plot weights
    plt.subplot(1, 3, 2)
    wei_col_nam = [f"{n}\n{x:.{decimals}f}" for n, x in zip(column_names, alpha)]
    wei_col_col = [SLISE_ORANGE if v < 0 else SLISE_PURPLE for v in alpha]
    plt.barh(wei_col_nam, alpha, color=wei_col_col)
    plt.title("\n\nLocal Linear Model")
    if class_names is not None and len(class_names) > 1:
        amax = alpha.abs().max() * 0.9
        plt.xticks([-amax, amax], class_names[:2])
    else:
        plt.xticks([])
    # Plot impact
    plt.subplot(1, 3, 3)
    imp_col_nam = [f"{n}\n{x:.{decimals}f}" for n, x in zip(column_names, impact)]
    imp_col_col = [SLISE_ORANGE if v < 0 else SLISE_PURPLE for v in impact]
    plt.barh(imp_col_nam, impact, color=imp_col_col)
    plt.title("\n\nActual Impact")
    if class_names is not None and len(class_names) > 1:
        imax = impact.abs().max() * 0.9
        plt.xticks([-imax, imax], class_names[:2])
    else:
        plt.xticks([])
    # Plot meta
    plt.tight_layout()
    plt.show()

def inset_pos(a: float, b: float, p: float = 0.2) -> (float, float):
    """
        Inset an interval with a percentage
    """
    if a > b:
        return inset_pos(b, a, p)
    d = b - a
    d = d * p * 0.5
    return a + d, b - d


def plot_explanation_dist(x: np.ndarray, y: float, X: np.ndarray, Y: np.ndarray, alpha: np.ndarray, subset: np.ndarray,
        scaler: DataScaler, column_names: list = None, class_names: list = None, decimals: int = 3):
    """Plot the current explanation (for tabular data), with density plots of the dataset and subset

    Arguments:
        x {np.ndarray} -- the explained item
        y {float} -- the explained prediction
        X {np.ndarray} -- the dataset (data matrix)
        Y {np.ndarray} -- the dataset (prediciton vector)
        alpha {np.ndarray} -- the explanation
        subset {np.ndarray} -- the subset of approximated items (as a boolean mask)
        scaler {DataScaler} -- scaler used to unscale the data

    Keyword Arguments:
        column_names {list} -- the names of the features/variables (default: {None})
        class_names {str or list} -- the names of the class (str) / classes (list), if explaining a classifier (default: {None})
        decimals {int} -- the precision to use for printing (default: {3})
    """
    # Values and order
    Xu, Yu = scaler.unscale(X, Y)
    xu, yu = scaler.unscale(x, y)
    impact = np.concatenate(([1.0], x)) * alpha
    impact /= np.sum(np.abs(impact))
    order, order_outer = get_explanation_order(alpha, scaler.scaler_x.mask)
    column_names = fill_column_names(column_names, len(xu) + 1, True)
    bins = max(10, min(50, len(Y)//20))
    if isinstance(class_names, str):
        class_names = ("not "+class_names, class_names)
    alpha = alpha[order]
    impact = impact[order]
    column_names = [column_names[i] for i in order_outer]
    # subplots
    rows = max(3, len(order))
    fig, axs = plt.subplots(rows, 2)
    fig.suptitle("SLISE Explanation   |   " + fill_prediction_str(yu, class_names, decimals))
    gs = axs[0, 0].get_gridspec()
    for ax in axs[1:, 1]:
        ax.remove()
    aih = (rows-1)//2
    axi = fig.add_subplot(gs[-aih:, 1])
    axa = fig.add_subplot(gs[(-2*aih):-aih, 1])
    # Y hist
    axs[0, 0].hist(Yu, bins=bins, density=False, histtype="step", color="black", label="Dataset")
    axs[0, 0].hist(Yu[subset], bins=bins, density=False, histtype="step", color=SLISE_PURPLE, label="Subset")
    axs[0, 0].relim()
    axs[0, 0].vlines(yu, *axs[0, 0].get_ylim(), color=SLISE_ORANGE, label="Explained Item")
    axs[0, 0].set_yticks([])
    if class_names is not None and len(class_names) > 1:
        pos = inset_pos(*axs[0, 0].get_xlim(), 0.2)
        if pos[0]*pos[1] < 0:
            axs[0, 0].set_xticks((pos[0], 0, pos[1]))
            axs[0, 0].set_xticklabels((class_names[0], "0", class_names[1]))
        else:
            axs[0, 0].set_xticks(pos)
            axs[0, 0].set_xticklabels(class_names[:2])
    axs[0, 0].set_title("\n\nPrediction")
    h, l = axs[0, 0].get_legend_handles_labels()
    axs[0, 1].legend(h, l, loc="center", bbox_to_anchor=(0, 0.5))
    axs[0, 1].axis("off")
    # X hists
    for i, k, n in zip(reversed(range(1, len(order))), order_outer[:-1]-1, column_names[:-1]):
        ax = axs[i, 0]
        ax.hist(Xu[:, k], bins=bins, density=False, histtype="step", color="black")
        ax.hist(Xu[subset, k], bins=bins, density=False, histtype="step", color=SLISE_PURPLE)
        ax.relim()
        ax.vlines(xu[k], *ax.get_ylim(), color=SLISE_ORANGE)
        ax.set_yticks([])
        ax.set_title(n)
    # alpha
    wei_col_nam = [f"{n} {x:{decimals + 3}.{decimals}f}" for n, x in zip(column_names, alpha)]
    wei_col_col = [SLISE_ORANGE if v < 0 else SLISE_PURPLE for v in alpha]
    axa.barh(wei_col_nam, alpha, color=wei_col_col)
    axa.set_title("Local Linear Model")
    if class_names is not None and len(class_names) > 1:
        pos = axa.get_xlim()
        amax = np.abs(alpha).max() * 0.4
        axa.set_xticks(inset_pos(pos[0]*0.6 - amax, pos[1]*0.6 + amax, 0.2))
        axa.set_xticklabels(class_names[:2])
    else:
        axa.set_xticks([])
    # impact
    imp_col_nam = [f"{n} {x:{decimals + 3}.{decimals}f}" for n, x in zip(column_names, impact)]
    imp_col_col = [SLISE_ORANGE if v < 0 else SLISE_PURPLE for v in impact]
    axi.barh(imp_col_nam, impact, color=imp_col_col)
    axi.set_title("Actual Impact")
    if class_names is not None and len(class_names) > 1:
        imax = np.abs(impact).max() * 0.4
        pos = axi.get_xlim()
        axi.set_xticks(inset_pos(pos[0]*0.6 - imax, pos[1]*0.6 + imax, 0.2))
        axi.set_xticklabels(class_names[:2])
    else:
        axi.set_xticks([])
    # meta
    plt.tight_layout()
    plt.show()
