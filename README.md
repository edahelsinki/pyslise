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

SLISE, as a robust regression algorithm, is able to handle outliers, while normal least-squares regression gives skewed results:  
![Example of Robust Regression](examples/ex1.png)

**`TODO`**

For more detailed examples and descriptions see the [examples](tree/master/examples) directory.

## Dependencies

- Python 3
- numpy
- scipy
- PyLBFGS
- mumba
- matplotlib
