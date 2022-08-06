"""Exports a set of functions and classes for evolutionary algorithms.

TheoTF's tools package extends DEAP's homonymous package, providing
additional variations of common operations in evolutionary algorithms,
like mutation, initialization, evaluation.
This package is composed of several sub-modules:
    - initialization: exports tools for individual initialization.
    - crossover: exports tools for crossover of individuals.
    - mutation: exports tools for mutation of individuals.
    - evaluation: exports tools for a model-based fitness evaluation of
    individuals
"""

from theotf.core.tools.initialization import *
from theotf.core.tools.crossover import *
from theotf.core.tools.mutation import *
from theotf.core.tools.evaluation import *