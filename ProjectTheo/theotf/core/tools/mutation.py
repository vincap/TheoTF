from theotf import ptypes
from deap import tools as dtools
import copy
import random

def additiveGaussianMutation(individual, paramValues, mu, sigmas, geneMutProb) :
    """In-place additive gaussian mutation.
    
    Applies in-place additive gaussian mutation on the provided
    individual. With a probability equal to geneMutProb, the i-th gene
    is mutated by summing a random number sampled from a normal
    distribution with mean=mu and sigma=sigmas[i] to its current value.
    Parameter values are used to guarantee that the individual is still
    valid after mutation. This function is based on the
    deap.tools.mutGaussian() function. NOTE: This mutation is a random
    process and the individual might not actually mutate. In this case,
    output individual will be exactly equal to the input individual.

    Args:
        individual: an individual (typically a sequence of genes).
    
        paramValues (Iterable): a sequence of parameter values, which
        must be instances of classes defined in the ptypes module.
            
        mu (float): center (mean) for the gaussian distributions.
        
        sigmas: sequence of standard deviations for the gaussian
        distributions.
        
        geneMutProb (float): independent gene mutation probability.
        
    Returns:
        A tuple of one individual (for compatibility with DEAP).
    """
    
    # in-place mutation of the individual
    dtools.mutGaussian(individual, mu, sigmas, geneMutProb)
    
    # Controls to make sure that individual values are still valid after mutation
    # Values might go off-range and integer values might need to be rounded.
    for idx, pval in enumerate(paramValues) :
        
        if isinstance(pval, ptypes.Categorical) :
            individual[idx] = round(individual[idx])
            lo = 0
            hi = len(pval)-1
        elif isinstance(pval, ptypes.DRange) :
            individual[idx] = round(individual[idx])
            lo = pval.lower()
            hi = pval.upper()
        elif isinstance(pval, ptypes.CRange) :
            lo = pval.lower()
            hi = pval.upper()
            
        if individual[idx] < lo :
            individual[idx] = lo
        elif individual[idx] > hi :
            individual[idx] = hi
            
    return (individual,)

def differentialMutationRand1(target, pop, scale, paramValues) :
    """Differential mutation in the standard rand/1 variant.
    
    Performs a standard rand/1 differential mutation as described in
    (Reiner and Storn, 1997). Three random individuals (r1, r2, r3) are
    sampled without replacement (so that they are all different) from
    the given population, making sure that each one of them is different
    from the given target individual. Considering individuals as real
    vectors, a mutant individual is then generated as:
    
        mutant = r1 + scale * (r2 - r3)
        
    Values in mutant are limited to their corresponding value ranges as
    specified in paramValues.
    Note that in order to apply this mutation procedure, the given
    population must contain at least 4 individuals. A ValueError
    exception is raised if population size is less than 4.

    Args:
        target: a target individual.
        
        pop (list): a sequence containing at least 4 individuals.
        
        scale (float): float value between 0.0 and 2.0, determining the strength
        of the mutation.
        
        paramValues (Iterable): a sequence of parameter values, which
        must be instances of classes defined in the ptypes module.
        
    Returns:
        The mutated individual.
    """
    
    if len(pop) < 4 :
        raise ValueError('Population size cannot be smaller than 4.')
        
    randSample = random.sample(pop, 3)
    
    # make sure that all 4 individuals (target + 3 random individuals) are different.
    while target in randSample :
        randSample = random.sample(pop, 3)
        
    mutant = copy.deepcopy(randSample[0])
    
    # implements vector sum.
    for idx, pval in enumerate(paramValues) :
        mutant[idx] += (randSample[1][idx] - randSample[2][idx]) * scale
        
        # make sure that value after mutation still falls into its parameter's range.
        if isinstance(pval, ptypes.Categorical) :
            lo = 0.0
            hi = len(pval)-1.0
        elif isinstance(pval, ptypes.DRange) or isinstance(pval, ptypes.CRange) :
            lo = pval.lower()
            hi = pval.upper()
            
        if mutant[idx] < lo :
            mutant[idx] = lo
        elif mutant[idx] > hi :
            mutant[idx] = hi
        
    return mutant