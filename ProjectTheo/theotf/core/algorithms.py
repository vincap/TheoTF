"""Functions defining evolutionary algorithms skeletons.

Functions contained in the algorithms module provide a skeleton for
common evolutionary algorithms. They implement the template method
design pattern and leverage the idea of DEAP's Toolbox to define
at runtime which evolutionary operators to use.
"""

from deap import base
from deap import tools
import numpy
import random

def standardEvolution(toolbox, cxProb, mutProb, populationSize, maxGenerations, hofSize, verbose=True) :
    """Standard evolutionary algorithm.
    
    This function implements a basic evolutionary algorithm. It creates
    a population of the given size, and generation after generation, for 
    a given maximum number of generations, tries to improve the quality
    of the individuals in order to maximize their fitness. Individuals
    undergo operations of crossover, mutation, selection and evaluation:
    the user can define how these operations actually behave through the
    provided Toolbox instance.

    Args:
        toolbox (deap.base.Toolbox): used to specify which functions to
        use for population initialization, crossover, mutation,
        selection and evaluation. The function expects the
        following names to be defined in the toolbox:
            -initPop(n) for population initialization.
            -mate(ind1, ind2) for in-place crossover of two individuals.
            -mutate(ind) for in-place individual mutation.
            -select(individuals, k) to select which individuals to mate.
            The function expects select to return a set of individuals
            to mate in successive pairs, for example, the first selected
            individual will mate with the second, the third with the
            fourth and so on.
            -evaluate(ind) for individual evaluation.
            
        cxProb (float): probability of crossover.
        
        mutProb (float): probability of mutation.
        
        populationSize (int): how many individuals to consider. Note
        that the bigger populationSize is, the longer the execution time
        will be.
        
        maxGenerations (int): for how many generations the evolution
        must run. Note that the bigger maxGenerations is, the longer the
        execution time will be.
        
        hofSize (int): size of the Hall of Fame, i.e. how many top 
        performant individuals should be kept and returned.
        
        verbose (bool): whether to log or not evolution progress 
        (default is True).
        
    Returns:
        A tuple containing a population (list) of individuals after the
        last generation, an hall of fame of the best individuals ever
        generated and a logbook containing evolution statistics.
    """
    
    logbook = tools.Logbook()
    logbook.header = "gen", "avg", "std", "min", "max"
    
    stats = tools.Statistics(key=lambda ind : ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    hof = tools.HallOfFame(hofSize)

    pop = toolbox.initPop(n=populationSize)
    
    if verbose :
        print('Evaluating generation 0...')
    
    for ind in pop :
        ind.fitness.values = toolbox.evaluate(ind)

    statsRec = stats.compile(pop)
    logbook.record(gen=0, **statsRec)

    hof.update(pop)

    for currGen in range(maxGenerations) :
        
        if verbose :
            print(f'Generation {currGen+1} of {maxGenerations}:', end=' ')

        offspring = toolbox.select(individuals=pop, k=populationSize)
        offspring = list(map(toolbox.clone, offspring))
        
        if verbose :
            print('applying crossover...', end=' ')

        for child1, child2 in zip(offspring[::2], offspring[1::2]) :
            if random.random() < cxProb :
                toolbox.mate(ind1=child1, ind2=child2) # in-place crossover
                del child1.fitness.values
                del child2.fitness.values
                
        if verbose :
            print('applying mutation...', end=' ')

        for mutant in offspring :
            if random.random() < mutProb :
                beforeMutation = [v for v in mutant]
                toolbox.mutate(mutant)

                # checks if the individual has actually mutated
                if beforeMutation != mutant :
                    del mutant.fitness.values

        if verbose :
            print('evaluating individuals...')
            
        for ind in offspring :
            if not ind.fitness.valid :
                ind.fitness.values = toolbox.evaluate(ind)
                pop.append(ind)

        # from the whole population (original individuals and new offsprings)
        # select the best individuals while keeping population size constant.
        survivors = tools.selBest(pop, populationSize)
        pop = [toolbox.clone(ind) for ind in survivors]

        statsRec = stats.compile(pop)
        logbook.record(gen=currGen+1, **statsRec)
        hof.update(pop)
        
    return pop, hof, logbook

def differentialEvolution(toolbox, populationSize, maxGenerations, hofSize, verbose=True) :
    """Differential evolution algorithm.
    
    This function implements a differential evolution algorithm.
    It creates a population of the given size, and generation
    after generation, for a given maximum number of generations,
    tries to improve the quality of the individuals in order to
    maximize their fitness. Individuals are considered as real-valued
    sequences and undergo operations of mutation and crossover as
    described in (Reiner and Storn, 1997). According to the common
    classification of DE/x/y/z for differential evolution variants,
    where x, y and z concern mutation and crossover, the user can
    specify the desired variant through a Toolbox instance.

    Args:
        toolbox (deap.base.Toolbox): used to specify which functions to
        use for population initialization, crossover, mutation, and
        evaluation. The function expects the following names to be
        defined in the toolbox:
            -initPop(n) for population initialization.
            -mate(target, mutant) for crossover of two individuals that
            returns a trial individual.
            -mutate(target, population) to generate and return a mutant
            individual of the given target from the given population.
            -evaluate(ind) for individual evaluation. Individuals are
            provided as real-valued sequences and any eventual decoding
            must be implemented by the evaluation function.
        
        populationSize (int): how many individuals to consider. Note
        that the bigger populationSize is, the longer the execution time
        will be.
        
        maxGenerations (int): for how many generations the evolution
        must run. Note that the bigger maxGenerations is, the longer the
        execution time will be.
        
        hofSize (int): size of the Hall of Fame, i.e. how many top 
        performant individuals should be kept and returned.
        
        verbose (bool): whether to log or not evolution progress 
        (default is True).
        
    Returns:
        A tuple containing a population (list) of individuals after the
        last generation, an hall of fame of the best individuals ever
        generated and a logbook containing evolution statistics.
    """
    
    logbook = tools.Logbook()
    logbook.header = "gen", "avg", "std", "min", "max"
    
    stats = tools.Statistics(key=lambda ind : ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    hof = tools.HallOfFame(hofSize)
    
    pop = toolbox.initPop(n=populationSize)
    
    if verbose :
        print('Evaluating generation 0...')
    
    for ind in pop :
        ind.fitness.values = toolbox.evaluate(ind)
        
    statsRec = stats.compile(pop)
    logbook.record(gen=0, **statsRec)

    hof.update(pop)
    
    for currGen in range(maxGenerations) :
        
        if verbose :
            print(f'Generation {currGen+1}...', end=' ')
            
        for idx, target in enumerate(pop) :
            
            if verbose :
                print('Applying mutation...', end=' ')
                
            mutant = toolbox.mutate(target, pop)
            
            if verbose :
                print('Applying crossover...', end=' ')
            
            trial = toolbox.mate(target, mutant)
            
            if verbose :
                print('Evaluating trial...')
            
            trial.fitness.values = toolbox.evaluate(trial)
            
            if trial.fitness >= target.fitness :
                pop[idx] = trial
                
        statsRec = stats.compile(pop)
        logbook.record(gen=currGen+1, **statsRec)
        hof.update(pop)
                
    return pop, hof, logbook