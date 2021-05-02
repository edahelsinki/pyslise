# These tests check if the plotting functions run

from matplotlib import pyplot as plt
from slise import regression, explain

from .utils import *


def test_plot2d():
    print("Testing 2D plots")
    X, Y, mod = data_create2(40, 1)
    reg = regression(X, Y, 0.1, lambda1=0.01, lambda2=0.01, intercept=False)
    reg.plot_2d(axis=plt.figure().add_subplot())
    reg = regression(
        X, Y, 0.1, lambda1=0.01, lambda2=0.01, intercept=True, normalise=True
    )
    reg.plot_2d(axis=plt.figure().add_subplot())
    exp = explain(X, Y, 0.1, 5, lambda1=0.01, lambda2=0.01)
    exp.plot_2d(axis=plt.figure().add_subplot())
    Y -= Y.min() - 0.01
    Y /= Y.max() + 0.01
    exp = explain(X, Y, 1.0, 5, lambda1=0.01, lambda2=0.01, logit=True)
    exp.plot_2d(axis=plt.figure().add_subplot())
    # plt.show()
    plt.close("all")


def test_dist():
    print("Testing dist plots")
    X, Y, mod = data_create2(40, 5)
    reg = regression(X, Y, 0.1, lambda1=0.01, lambda2=0.01, intercept=False)
    reg.plot_dist(fig=plt.figure())
    reg = regression(
        X, Y, 0.1, lambda1=0.01, lambda2=0.01, intercept=True, normalise=True
    )
    reg.plot_dist(fig=plt.figure())
    exp = explain(X, Y, 0.1, 5, lambda1=0.01, lambda2=0.01)
    exp.plot_dist(fig=plt.figure())
    Y -= Y.min() - 0.01
    Y /= Y.max() + 0.01
    exp = explain(X, Y, 1.0, 5, lambda1=0.01, lambda2=0.01, logit=True)
    exp.plot_dist(fig=plt.figure())
    # plt.show()
    plt.close("all")
