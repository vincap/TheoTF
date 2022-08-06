"""Main interface for TheoTF optimization.

The optimizers module exports classes that enable end users to access
TheoTF functionalities without worrying about evolutionary algorithms.
"""

from theotf import ptypes
from theotf.core import tools as ttools
from theotf.core import algorithms
from theotf.core import utilities
from deap import creator
from deap import base
from deap import tools as dtools

# basic Fitness and Individual types.
creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
creator.create('Individual', list, fitness=creator.FitnessMin)

class OptimizationResult :
    """Contains results produced by an evolutionary optimizer.
    
    OptimizationResult is nothing more than a container for results
    produced by an evolutionary optimizer. It contains:
    - A dict containing the best hyperparameters configuration found
    during optimization.
    - A keras.Sequential model with the best hyperparameters
    configuration.
    - The validation loss achieved by such model.
    - An history logbook reporting loss evolution during the
    optimization process.
    """
    
    def __init__(self, bestIndividual, logBook, paramsGrid, modelConfig) :
        self.__bestModel = utilities.individualToModel(bestIndividual, paramsGrid, modelConfig)
        self.__bestLoss = bestIndividual.fitness.values
        self.__history = logBook
        self.__bestParams = {}
        
        for pname, pval, indval in zip(paramsGrid.keys(), paramsGrid.values(), bestIndividual) :
            if isinstance(pval, ptypes.DRange) or isinstance(pval, ptypes.CRange) :
                val = indval
            elif isinstance(pval, ptypes.Categorical) :
                val = pval[indval]
                
            self.__bestParams[pname] = val
            
    @property
    def bestModel(self) :
        return self.__bestModel
    
    @property
    def bestLoss(self) :
        return self.__bestLoss
    
    @property
    def history(self) :
        return self.__history

    @property
    def bestParams(self) :
        return self.__bestParams
    
class StandardEvoOpt :
    """An evolutionary optimizer based on a standard evolutionary
    algorithm.
    
    StandardEvoOpt defines a simple model optimization interface that
    enables users to optimize their model through a basic evolution
    strategy without worrying about operators and evolution at all.
    Evolution is actually carried out by the standardEvolution function
    (in theotf.core.algorithms).
    """
    
    # static class attributes.
    __mutationCenter = 0.0
    __geneMutationProb = 0.8
    __tournamentSize = 5
    __cxProb = 0.7
    __cxSwapProb = 0.5
    __mutProb = 0.1
    __hofSize = 3
    
    def __init__(self, refModel, paramsGrid, populationSize=10, maxGenerations=20) :
        """Builds an evolutionary optimizer for the given model.
        
        Builds an evolutionary optimizer, ready to optimize the given
        parameters for the given model.
        
        Args:
            refModel (tf.keras.Sequential): model to optimize.
            
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
            
            populationSize (int): defines how many distinct
            configurations should be maintained during optimization.
            Note that the bigger populationSize is, the longer the
            optimization will take (default is 10).
            
            maxGenerations (int): defines for how many generations the
            optimization should be executed. Note that the bigger
            maxGenerations is, the longer the optimization will take 
            (default is 20).
        """
        
        self.__modelConfig = refModel.get_config()
        self.__paramsGrid = dict(paramsGrid)
        self.__populationSize = populationSize
        self.__maxGenerations = maxGenerations
        self.__mutationSigmas = utilities.variationAwareSigmas(self.__paramsGrid.values())
        
        self.__toolbox = base.Toolbox()
        self.__toolbox.register('initInd', ttools.typeawareIndividualInit, IndividualClass=creator.Individual, paramValues=self.__paramsGrid.values())
        self.__toolbox.register('initPop', dtools.initRepeat, container=list, func=self.__toolbox.initInd)
        self.__toolbox.register('mate', dtools.cxUniform, indpb=StandardEvoOpt.__cxSwapProb)
        self.__toolbox.register('mutate', ttools.additiveGaussianMutation,
                                paramValues=self.__paramsGrid.values(),
                                mu=StandardEvoOpt.__mutationCenter,
                                sigmas=self.__mutationSigmas,
                                geneMutProb=StandardEvoOpt.__geneMutationProb)        
        self.__toolbox.register('select', dtools.selTournament, tournsize=StandardEvoOpt.__tournamentSize)

    def fit(self, train, validation, tfLoss, tfOptimizer='adam', batchSize=64, trainEpochs=20, tfCallbacks=None, verbose=True) :
        """Optimizes the model associated with this optimizer instance.
        
        Runs a standard evolutionary algorithm to optimize the
        parameters of the model associated with this optimizer at
        creation time.
        
        Args:
            train (tuple): a tuple containing training patterns in
            position 0 and training labels in position 1 (e.g. a tuple
            like (train_x, train_y)).
            
            validation (tuple): a tuple containing validation patterns
            in position 0 and validation labels in position 1 (e.g. a
            tuple like (val_x, val_y)).
            
            tfLoss (str or keras.losses.Loss): a keras Loss object
            specifying the loss function to use.

            tfOptimizer (str or keras.optimizers.Optimizer): a 
            tensorflow optimizer to use for model training (default is 
            'adam').

            batchSize (int): batch size to use for training (default is 
            64).

            trainingEpochs (int): maximum number of training iterations 
            (default is 20).

            tfCallbacks (list of Callable): tensorflow training 
            callbacks (default is None).
            
            verbose (bool): whether to log or not evolution progress.
            
        Returns :
            An OptimizerResult instance containing the best
            hyperparameters configuration found during evolution, a
            keras.Sequential model corresponding to this configuration,
            the validation loss achieved by this model and an history
            logbook reporting loss evolution during the optimization
            process.
        """
        
        evalCbk = None
        if verbose == True :
            evalCbk = [utilities.EvaluationCallback]
        
        self.__toolbox.register('evaluate', ttools.IndividualEvaluatorTF(evalCbk),
                                paramsGrid=self.__paramsGrid,
                                modelConfig=self.__modelConfig,
                                trainSet=train,
                                validationSet=validation,
                                tfLoss=tfLoss,
                                tfOptimizer=tfOptimizer,
                                batchSize=batchSize,
                                trainEpochs=trainEpochs,
                                tfCallbacks=tfCallbacks)
        
        pop, hof, logbook = algorithms.standardEvolution(toolbox=self.__toolbox,
                                                         cxProb=StandardEvoOpt.__cxProb,
                                                         mutProb=StandardEvoOpt.__mutProb,
                                                         populationSize=self.__populationSize,
                                                         maxGenerations=self.__maxGenerations,
                                                         hofSize=StandardEvoOpt.__hofSize,
                                                         verbose=verbose)
        
        self.__toolbox.unregister('evaluate')
        
        return OptimizationResult(hof[0], logbook, self.__paramsGrid, self.__modelConfig)
        # return (utilities.individualToModel(hof[0], self.__paramsGrid, self.__modelConfig), logbook)
        # return (hof[0], logbook)
    
class DifferentialEvoOpt :
    """An evolutionary optimizer based on a differential evolution
    algorithm.
    
    DifferentialEvoOpt defines a simple model optimization interface
    that enables users to optimize their model with a differential
    evolution approach, without worrying about operators and evolution
    at all. Evolution is actually carried out by the
    differentialEvolution function (in theotf.core.algorithms).
    """
    
    # static class attributes.
    __mutScale=1.0
    __cxProb = 0.5
    __hofSize = 3
    
    def __init__(self, refModel, paramsGrid, populationSize=10, maxGenerations=20) :
        """Builds a differential evolution optimizer for the given model.
        
        Builds a differential evolution optimizer, ready to optimize the
        given parameters for the given model.
        
        Args:
            refModel (tf.keras.Sequential): model to optimize.
            
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
            
            populationSize (int): defines how many distinct
            configurations should be maintained during optimization.
            Note that the bigger populationSize is, the longer the
            optimization will take (default is 10).
            
            maxGenerations (int): defines for how many generations the
            optimization should be executed. Note that the bigger
            maxGenerations is, the longer the optimization will take
             (default is 20).
        """
        
        self.__modelConfig = refModel.get_config()
        self.__paramsGrid = dict(paramsGrid)
        self.__populationSize = populationSize
        self.__maxGenerations = maxGenerations
        
        self.__toolbox = base.Toolbox()
        self.__toolbox.register('initInd', ttools.differentialIndividualInit, IndividualClass=creator.Individual, paramValues=self.__paramsGrid.values())
        self.__toolbox.register('initPop', dtools.initRepeat, container=list, func=self.__toolbox.initInd)
        self.__toolbox.register('mate', ttools.differentialCrossoverBin, cxProb=DifferentialEvoOpt.__cxProb)
        self.__toolbox.register('mutate', ttools.differentialMutationRand1, scale=DifferentialEvoOpt.__mutScale, paramValues=self.__paramsGrid.values())        

    def fit(self, train, validation, tfLoss, tfOptimizer='adam', batchSize=64, trainEpochs=20, tfCallbacks=None, verbose=True) :
        """Optimizes the model associated with this optimizer instance.
        
        Runs a differential evolution algorithm to optimize the
        parameters of the model associated with this optimizer at
        creation time.
        
        Args:
            train (tuple): a tuple containing training patterns in
            position 0 and training labels in position 1 (e.g. a tuple
            like (train_x, train_y)).
            
            validation (tuple): a tuple containing validation patterns
            in position 0 and validation labels in position 1 (e.g. a
            tuple like (val_x, val_y)).
            
            tfLoss (str or keras.losses.Loss): a keras Loss object
            specifying the loss function to use.

            tfOptimizer (str or keras.optimizers.Optimizer): a 
            tensorflow optimizer to use for model training (default is 
            'adam').

            batchSize (int): batch size to use for training (default is 
            64).

            trainingEpochs (int): maximum number of training iterations 
            (default is 20).

            tfCallbacks (list of Callable): tensorflow training 
            callbacks (default is None).
            
            verbose (bool): whether to log or not evolution progress.
            
        Returns :
            An OptimizerResult instance containing the best
            hyperparameters configuration found during evolution, a
            keras.Sequential model corresponding to this configuration,
            the validation loss achieved by this model and an history
            logbook reporting loss evolution during the optimization
            process.
        """
        
        evalCbk = None
        if verbose == True :
            evalCbk = [utilities.EvaluationCallback]
        
        self.__toolbox.register('evaluate', ttools.DifferentialEvaluatorTF(evalCbk),
                               paramsGrid=self.__paramsGrid,
                               modelConfig=self.__modelConfig,
                               trainSet=train,
                               validationSet=validation,
                               tfLoss=tfLoss,
                               tfOptimizer=tfOptimizer,
                               batchSize=batchSize,
                               trainEpochs=trainEpochs,
                               tfCallbacks=tfCallbacks)
        
        pop, hof, logbook = algorithms.differentialEvolution(toolbox=self.__toolbox,
                                                             populationSize=self.__populationSize,
                                                             maxGenerations=self.__maxGenerations,
                                                             hofSize=DifferentialEvoOpt.__hofSize,
                                                             verbose=verbose)
        
        self.__toolbox.unregister('evaluate')
        
        correctedBest = utilities.differentialIndividualCorrection(hof[0], self.__paramsGrid.values())
        return OptimizationResult(correctedBest, logbook, self.__paramsGrid, self.__modelConfig)
        # return (utilities.individualToModel(correctedBest, self.__paramsGrid, self.__modelConfig), logbook)
        # return (correctedBest, logbook)