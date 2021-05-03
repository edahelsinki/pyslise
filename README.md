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

### Other Languages

The (original) R implementation can be found [here](https://github.com/edahelsinki/slise).

## Installation

To install this package just run:
```sh
pip install https://github.com/edahelsinki/pyslise
```

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
