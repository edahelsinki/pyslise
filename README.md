# SLISE - Sparse Linear Subset Explanations

Python implementation of the SLISE algorithm. The SLISE algorithm can be used for
both robust regression and to explain outcomes from black box models.
For more details see [the paper](https://rdcu.be/bVbda), alternatively for a more informal
overview see [the presentation](https://github.com/edahelsinki/slise/raw/master/vignettes/presentation.pdf),
or [the poster](https://github.com/edahelsinki/slise/raw/master/vignettes/poster.pdf).

> **Björklund A., Henelius A., Oikarinen E., Kallonen K., Puolamäki K.**  
> *Sparse Robust Regression for Explaining Classifiers.*  
> Discovery Science (DS 2019).  
> Lecture Notes in Computer Science, vol 11828, Springer.  
> https://doi.org/10.1007/978-3-030-33778-0_27

## The idea

In robust regression we fit regression models that can handle data that contains outliers (see the example below for why outliers are problematic for normal regression). SLISE accomplishes this by fitting a model such that the largest possible subset of the data items have an error less than a given value. All items with an error larger than that are considered potential outliers and do not affect the resulting model.

SLISE can also be used to provide *local model-agnostic explanations* for outcomes from black box models. To do this we replace the ground truth response vector with the predictions from the complex model. Furthermore, we force the model to fit a selected item (making the explanation local). This gives us a local approximation of the complex model with a simpler linear model (this is similar to, e.g., [LIME](https://github.com/marcotcr/lime) and [SHAP](https://github.com/slundberg/shap)). In contrast to other methods SLISE creates explanations using real data (not some discretised and randomly sampled data) so we can be sure that all inputs are valid (i.e. in the correct data manifold, and follows the constraints used to generate the data, e.g., the laws of physics).

## Installation

To install this package just run:
```sh
pip install https://github.com/edahelsinki/pyslise
```
Alternatively you can download the repo and run `python -m build` to build a wheel or `pip install .` to install locally.

### Other Languages

The (original) R implementation can be found [here](https://github.com/edahelsinki/slise).

## Example

SLISE is a robust regression algorithm, which means that it is able to handle outliers. This is in contrast to, e.g., normal least-squares regression, which gives skewed results in presence of outliers:  
![Example of Robust Regression](examples/ex1.png)

SLISE can also be used to explain outcomes from black box models by locally approximating the complex models with a simpler linear model (in a way that takes the dataset into account):  
![Example of Robust Regression](examples/ex2.png)


For more detailed examples and descriptions see the [examples](https://github.com/edahelsinki/pyslise/tree/master/examples) directory.

## Dependencies

This implementation is requires Python 3 and the following packages:

- matplotlib
- numba
- numpy
- PyLBFGS
- scipy
