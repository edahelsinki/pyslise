# These tests check if the plotting functions run

import numpy as np
from matplotlib import pyplot as plt
from slise import regression, explain
from slise.plot import get_explanation_order

from .utils import data_create2


def test_plot2d():
    print("Testing 2D plots")
    X, Y, mod = data_create2(40, 1)
    reg = regression(X, Y, 0.1, lambda1=1e-4, lambda2=1e-4, intercept=False)
    reg.plot_2d(fig=plt.figure())
    reg = regression(
        X, Y, 0.1, lambda1=1e-4, lambda2=1e-4, intercept=True, normalise=True
    )
    reg.plot_2d(fig=plt.figure())
    exp = explain(X, Y, 0.1, 5, lambda1=1e-4, lambda2=1e-4)
    exp.plot_2d(fig=plt.figure())
    Y -= Y.min() - 0.01
    Y /= Y.max() + 0.01
    exp = explain(X, Y, 1.0, 5, lambda1=1e-4, lambda2=1e-4, logit=True)
    exp.plot_2d(fig=plt.figure())
    # plt.show()
    plt.close("all")


def test_print():
    X, Y, mod = data_create2(40, 5)
    reg = regression(X, Y, 0.1, lambda1=1e-4, lambda2=1e-4, intercept=True)
    reg.print()
    reg.print(variables=[str(i) for i in range(5)], decimals=2, num_var=4)
    reg = regression(X, Y, 0.1, lambda1=1e-4, lambda2=1e-4, intercept=False)
    reg.print()
    reg.print(variables=[str(i) for i in range(5)], decimals=2, num_var=4)
    exp = explain(X, Y, 0.1, 5, lambda1=1e-4, lambda2=1e-4)
    exp.print()
    exp.print(classes=["asd", "bds"], variables=[str(i) for i in range(5)], num_var=4)
    exp.print(classes="bds", decimals=2, num_var=4)
    Y = Y - Y.min() + 0.01
    Y /= Y.max() + 0.01
    exp = explain(X, Y, 1.0, 5, lambda1=1e-4, lambda2=1e-4, normalise=True, logit=True)
    exp.print()
    exp.print(classes=["asd", "bds"], variables=[str(i) for i in range(5)], num_var=4)
    exp.print(classes="bds", decimals=2, num_var=4)


def test_dist():
    try:
        X, Y, mod = data_create2(40, 5)
        reg = regression(X, Y, 0.1, lambda1=1e-4, lambda2=1e-4, intercept=False)
        reg.plot_dist(fig=plt.figure())
        reg.plot_dist(variables=[str(i) for i in range(5)], order=3, fig=plt.figure())
        reg.plot_dist(title="asd", order=range(1, 4), decimals=5, fig=plt.figure())
        reg = regression(
            X, Y, 0.1, lambda1=1e-4, lambda2=1e-4, intercept=True, normalise=True
        )
        reg.plot_dist(fig=plt.figure())
        reg.plot_dist(variables=[str(i) for i in range(5)], order=3, fig=plt.figure())
        reg.plot_dist(title="asd", order=range(1, 4), decimals=5, fig=plt.figure())
        exp = explain(X, Y, 0.1, 5, lambda1=1e-4, lambda2=1e-4)
        exp.plot_dist(fig=plt.figure())
        exp.plot_dist(variables=[str(i) for i in range(5)], order=3, fig=plt.figure())
        exp.plot_dist(title="asd", order=range(1, 4), decimals=5, fig=plt.figure())
        Y = Y - Y.min() - 0.01
        Y /= Y.max() + 0.01
        exp = explain(
            X, Y, 1.0, 5, lambda1=1e-4, lambda2=1e-4, normalise=True, logit=True
        )
        exp.plot_dist(fig=plt.figure())
        exp.plot_dist(variables=[str(i) for i in range(5)], order=3, fig=plt.figure())
        exp.plot_dist(title="asd", order=range(1, 4), decimals=5, fig=plt.figure())
        reg.plot_subset(fig=plt.figure())
        exp.plot_subset(fig=plt.figure())
    finally:
        plt.close("all")


def test_img():
    print("Testing image plots")
    X, Y, mod = data_create2(200, 16)
    X[:, 6] = X[:, 9] = X[:, 11] = 0
    exp = explain(X, Y, 0.1, 5, lambda1=1e-4, lambda2=1e-4)
    exp.plot_image(4, 4, fig=plt.figure())
    # plt.show()
    plt.close("all")


def test_order():
    def check(a, b, **kwargs):
        assert np.allclose(a[get_explanation_order(a, **kwargs)], b)

    a = np.arange(5)
    check(a, [0, 4, 3, 2, 1], intercept=True, min=4)
    check(a, [4, 3, 2, 1], intercept=False, min=4)
    check(a, [4, 3, 2], intercept=False, max=3)
    check(a, [0, 4, 3, 2], intercept=True, max=3)
    a = np.arange(-3, 4)
    check(a, [-3, 3, 2, 1, -1, -2], intercept=True, min=4)
    check(a, [3, 2, 1, -1, -2, -3], intercept=False, min=4)
    check(a, [-3, 3, 2, -2], intercept=True, max=3)
    check(a, [3, 2, -2, -3], intercept=False, max=3)
