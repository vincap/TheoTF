"""Exports a set of utility functions used by other theotf components.

The utilities module defines several utility functions primarily used
by other theotf components, but potentially useful for users as well.
"""

from tensorflow import keras
from theotf import ptypes
import copy

def parseParameterString(pstring) :
    """Splits a parameter string into a tuple layerName, attributeName.

    Expects a string formatted as layerName.attributeName and returns a
    tuple containing layerName and attributeName as separate strings.
    For example:
        parseParameterString('Dense.units') # returns ('Dense', 'units')

    Args:
        pstring (str): a string formatted as layerName.attributeName,
        for example 'Dense.units'.
        
    Returns:
        A (layerName, attributeName) tuple of strings.
        
    Raises:
        RuntimeError: if pstring is not properly formatted.
    """
    
    separatorIdx = pstring.find('.')
    
    if separatorIdx < 0 :
        raise RuntimeError('Wrong parameter format. Must be layerName.attributeName.')
        
    layerName = pstring[:separatorIdx]
    attributeName = pstring[separatorIdx+1:]
    
    return (layerName, attributeName)

def individualToModel(individual, paramsGrid, modelConfig) :
    """Decodes an individual into a keras Sequential model.

    Decodes an individual into a keras Sequential model starting from
    the provided configuration (obtained via model.get_config()) and
    using the specified parameters grid to translate the individual into
    model parameters.
    The individual and the parameters grid must be compatible: they
    must contain the same number of values and the i-th value of the
    individual must agree with the type specified by the i-th parameter
    in the grid.
    Parameters grid and model must also be compatible, meaning that
    the provided model configuration must contain all the specified
    parameters. Furthermore, the type associated with a given parameter
    in the grid must be compatible with the corresponding parameter in
    the model (for example, if a the activation attribute expects a
    callable object, the 'layer.activation' parameter must provide
    callable objects). All parameters defined in the model but not in
    the parameters grid are left untouched and returned as specified in
    the provided configuration.

    Args:
        individual: sequence of values for the requested parameters.
        
        paramsGrid (dict): dictionary which keys specify the parameters
        within the model and which values provide values to associate
        with the corresponding parameter. Values must be instances of
        the classes provided in the ptypes module. For example:

        ```
        paramsGrid = {
                'Dense.units':ptypes.DRange(32, 128),
                'Dense.activations':ptypes.Categorical(
                                        (relu, tanh, sigmoid)),
                'Dropout.rate':ptypes.CRange(0.4, 0.6)
        }
        ```
                          
        modelConfig (dict): reference architecture of the model, as
        returned by model.get_config().
        
    Returns:
        A tensorflow.keras.Sequential object.
    """
    
    model = keras.Sequential.from_config(modelConfig)

    for indVal, pname, pval in zip(individual, paramsGrid.keys(), paramsGrid.values()) :
        layer, attr = parseParameterString(pname)

        if isinstance(pval, ptypes.Categorical) :
            setattr(model.get_layer(name=layer), attr, pval[indVal])
        elif isinstance(pval, ptypes.DRange) or isinstance(pval, ptypes.CRange) :
            setattr(model.get_layer(name=layer), attr, indVal)

    return model

def variationAwareSigmas(paramValues, rangeScale=0.3) :
    """Returns a standard deviation for each parameter.
    
    Computes a standard deviation value for each parameter in the
    provided Iterable. Standard deviation depends on the variation
    range of a parameter, which for Categoricals is the number of
    possible values, while for Ranges is the width (upper-lower).
    Returned stds are typically used to mutate individuals using 
    gaussian mutation.

    Args:
        paramValues (Iterable): a sequence of parameter values, which
        must be instances of classes defined in the ptypes module.
        
        rangeScale (float): a floating-point number in [.0, 1.0]
        defining how to scale the variation range of parameters (default
        is 0.3).
        
    Returns:
        A list of standard deviations, one for each input parameter.
    """
    
    if rangeScale < 0.0 :
        rangeScale = 0.0
    elif rangeScale > 1.0 :
        rangeScale = 1.0
        
    sigmas = []
        
    for pval in paramValues :
        if isinstance(pval, ptypes.Categorical) :
            s = rangeScale*len(pval)
        else :
            s = rangeScale*(pval.upper()-pval.lower())
            
        sigmas.append(s)
        
    return sigmas

def differentialIndividualCorrection(individual, paramValues) :
    """Corrects a differential individual according to parameter types.
    
    In differential evolution, individuals are real-valued vectors. To
    build a TensorFlow model out of a differential individual, its real
    values must first be converted according to the corresponding
    parameter type. This function corrects the values of the given
    individual according to the type of the corresponding parameter, as
    specified in the provided iterable.
    Note that this function makes no assumption about value ranges. So
    the values of the provided individual must already fall in the
    correct range. This function does not cast values to their expected
    range.

    Args:
        individual: a real-valued individual.
        
        paramValues (Iterable): a sequence of parameter values, which
        must be instances of classes defined in the ptypes module.
        
    Returns:
        The corrected individual. Input individual is left unchanged.
    """
    
    correctedInd = copy.deepcopy(individual)
    
    for idx, pval in enumerate(paramValues) :
        if isinstance(pval, ptypes.DRange) or isinstance(pval, ptypes.Categorical) :
            correctedInd[idx] = round(correctedInd[idx])
            
    return correctedInd

def EvaluationCallback(ind, fitness) :
    print(f'Individual {ind} evaluated; fitness={fitness}')