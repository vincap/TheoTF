from theotf import ptypes
import random

def typeawareIndividualInit(IndividualClass, paramValues) :
    """Randomly initializes an individual according to the provided
    values.
    
    Randomly initializes a fresh-created individual as an instance of
    the provided class. Returned individual has as many genes as there
    are values in the provided Iterable. Each gene is initialized with
    a random value from the corresponding element in paramValues.

    Args:
        IndividualClass: class used to create an individual.
    
        paramValues (Iterable): a sequence of parameter values, which
        must be instances of classes defined in the ptypes module.
        
    Returns:
        A randomly-initialized instance of IndividualClass.
    """
    
    anIndividual = IndividualClass()
    
    for pval in paramValues :
        if isinstance(pval, ptypes.DRange) :
            v = random.randint(pval.lower(), pval.upper())
        elif isinstance(pval, ptypes.CRange) :
            v = random.uniform(pval.lower(), pval.upper())
        elif isinstance(pval, ptypes.Categorical) :
            v = random.randint(0, len(pval)-1)
            
        anIndividual.append(v)
        
    return anIndividual

def differentialIndividualInit(IndividualClass, paramValues) :
    """Randomly initializes a real-valued individual according to the
    provided values.
    
    Randomly initializes a fresh-created individual as an instance of
    the provided class. Returned individual has as many genes as there
    are values in the provided Iterable. Each gene is initialized with
    a real number randomly extracted within the range of the
    corresponding parameter.

    Args:
        IndividualClass: class used to create an individual.
    
        paramValues (Iterable): a sequence of parameter values, which
        must be instances of classes defined in the ptypes module.
        
    Returns:
        A randomly-initialized real-valued instance of IndividualClass.
    """
    
    anIndividual = IndividualClass()
    
    for pval in paramValues :
        if isinstance(pval, ptypes.CRange) or isinstance(pval, ptypes.DRange) :
            v = random.uniform(pval.lower(), pval.upper())
        elif isinstance(pval, ptypes.Categorical) :
            v = random.uniform(0, len(pval)-1)
            
        anIndividual.append(v)
        
    return anIndividual