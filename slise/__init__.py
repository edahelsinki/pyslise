"""
    The SLISE algorithm can be used for both robust regression and to explain outcomes from black box models.
"""

from slise.slise import SliseRegression, regression, SliseExplainer, explain
from slise.utils import limited_logit as logit
from slise.data import normalise_robust
