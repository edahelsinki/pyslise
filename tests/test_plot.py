# These tests check if the plotting functions run

from matplotlib import pyplot as plt
from slise import regression, explain

from .utils import *


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


def test_dist():
    print("Testing dist plots")
    X, Y, mod = data_create2(40, 5)
    reg = regression(X, Y, 0.1, lambda1=1e-4, lambda2=1e-4, intercept=False)
    reg.plot_dist(fig=plt.figure())
    reg = regression(
        X, Y, 0.1, lambda1=1e-4, lambda2=1e-4, intercept=True, normalise=True
    )
    reg.plot_dist(fig=plt.figure())
    exp = explain(X, Y, 0.1, 5, lambda1=1e-4, lambda2=1e-4)
    exp.plot_dist(fig=plt.figure())
    Y = Y - Y.min() - 0.01
    Y /= Y.max() + 0.01
    exp = explain(X, Y, 1.0, 5, lambda1=1e-4, lambda2=1e-4, normalise=True, logit=True)
    exp.plot_dist(fig=plt.figure())
    reg.plot_subset(fig=plt.figure())
    exp.plot_subset(fig=plt.figure())
    # plt.show()
    plt.close("all")


def test_img():
    print("Testing image plots")
    X, Y, mod = data_create2(200, 16)
    X[:, 6] = X[:, 9] = X[:, 11] = 0
    exp = explain(X, Y, 0.1, 5, lambda1=1e-4, lambda2=1e-4)
    exp.plot_image(4, 4, fig=plt.figure())
    # plt.show()
    plt.close("all")
