"""
    __ SLISE - Sparse Linear Subset Explanations __

    The SLISE algorithm can be used for both robust regression and to explain outcomes from black box models.


    In robust regression we fit regression models that can handle data that
    contains outliers. SLISE accomplishes this by fitting a model such that
    the largest possible subset of the data items have an error less than a
    given value. All items with an error larger than that are considered
    potential outliers and do not affect the resulting model.

    SLISE can also be used to provide local model-agnostic explanations for
    outcomes from black box models. To do this we replace the ground truth
    response vector with the predictions from the complex model. Furthermore, we
    force the model to fit a selected item (making the explanation local). This
    gives us a local approximation of the complex model with a simpler linear
    model. In contrast to other methods SLISE creates explanations using real
    data (not some discretised and randomly sampled data) so we can be sure that
    all inputs are valid (i.e. in the correct data manifold, and follows the
    constraints used to generate the data, e.g., the laws of physics).


    More in-depth details about the algorithm can be found in the papers:

    Björklund A., Henelius A., Oikarinen E., Kallonen K., Puolamäki K.
    Sparse Robust Regression for Explaining Classifiers.
    Discovery Science (DS 2019).
    Lecture Notes in Computer Science, vol 11828, Springer.
    https://doi.org/10.1007/978-3-030-33778-0_27

    Björklund A., Henelius A., Oikarinen E., Kallonen K., Puolamäki K.
    Robust regression via error tolerance.
    Data Mining and Knowledge Discovery (2022).
    https://doi.org/10.1007/s10618-022-00819-2

"""

from slise.slise import (
    SliseRegression,
    regression,
    SliseExplainer,
    explain,
    SliseWarning,
)
from slise.utils import limited_logit as logit
from slise.data import normalise_robust
