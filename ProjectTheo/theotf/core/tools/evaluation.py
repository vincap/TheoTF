from theotf.core import utilities
from tensorflow import keras

def tfIndividualEvaluation(individual, paramsGrid, modelConfig, trainSet, validationSet, tfLoss, tfOptimizer='adam', batchSize=64, trainEpochs=20, tfCallbacks=None) :
    """Evaluates a given individual from the corresponding model.
    
    Computes a fitness value for the provided individual as the
    validation loss of its corresponding Tensorflow model. First, the
    individual is decoded into a keras.Sequential model, which is then
    trained using the provided training set and finally evaluated on the
    provided validation set. Validation loss defines the fitness of the
    individual, hence lower is better. Training is based on Tensorflow
    and configured with the provided parameters.

    Args:
        individual: an individual (typically a sequence of genes).
    
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
        returned by model.get_config()
        
        trainSet (tuple): a tuple containing training patterns in
        position 0 and training labels in position 1 (e.g. a tuple like
        (train_x, train_y)).
            
        validationSet (tuple): a tuple containing validation patterns in
        position 0 and validation labels in position 1 (e.g. a tuple
        like (val_x, val_y)).
        
        tfLoss (str or keras.losses.Loss): a keras Loss object
        specifying the loss function to use.
        
        tfOptimizer (str or keras.optimizers.Optimizer): a tensorflow
        optimizer to use for model training (default is 'adam').
        
        batchSize (int): batch size to use for training (default is 64).
        
        trainingEpochs (int): maximum number of training iterations 
        (default is 20).
        
        tfCallbacks (list of Callable): tensorflow training callbacks 
        (default is None).
        
    Returns:
        A tuple containing a single fitness for the given individual.
    """
    
    model = utilities.individualToModel(individual, paramsGrid, modelConfig)

    model.compile(optimizer=tfOptimizer, loss=tfLoss)
    
    model.fit(x=trainSet[0], y=trainSet[1],
              validation_data=validationSet,
              batch_size=batchSize,
              epochs=trainEpochs,
              verbose=0,
              callbacks=tfCallbacks)

    fitness = model.evaluate(x=validationSet[0], y=validationSet[1], verbose=0)

    keras.backend.clear_session()
    del model

    return (fitness,)

class IndividualEvaluatorTF :
    """tfIndividualEvaluation wrapper for standard individuals.
    
    Wraps the tfIndividualEvalutation function for individuals used in
    the standard evolutionary algorithm (heterogeneous individuals which
    values are constrained to their corresponding parametr types) to
    provide additional functionalities.
    Specifically this class makes the evaluation smarter by maintaining
    a local history of already evaluated individuals: if an individual
    has been already evaluated in the past, its fitness is immediately
    returned, without re-evaluating the individual. This class also
    allows to set callback functions that are automatically invoked
    after an individual is evaluated.
    """
    
    def __init__(self, evaluationCallbacks=None) :
        """Builds a new standard evaluator object.
        
        Builds a standard evaluator object with a clean evaluation 
        history and the specified callbacks.
        
        Args:
            evaluationCallbacks (list of callable): list of callback
            functions to invoke after individual evaluation. Each
            callback must accept an individual as first argument and its
            fitness as second argument (default is None).
        """
        
        self.__evaluationHistory = {}
        self.__evaluationCallbacks = evaluationCallbacks
        
    def __call__(self, individual, paramsGrid, modelConfig, trainSet, validationSet, tfLoss, tfOptimizer='adam', batchSize=64, trainEpochs=20, tfCallbacks=None) :
        """Evaluates a given individual from the corresponding model.
        Redirects to the tfIndividualEvaluation function.

        If the specified individual has not been already evaluated in
        the past, its fitness value is computed via the
        tfIndividualEvaluation function with the provided parameters.
        If the specified individual has already been evaluated in the
        past, its previously computed fitness value is returned
        immediately.

        Args:
            individual: an individual (typically a sequence of genes).

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
            returned by model.get_config()

            trainSet (tuple): a tuple containing training patterns in
            position 0 and training labels in position 1 (e.g. a tuple like
            (train_x, train_y)).

            validationSet (tuple): a tuple containing validation patterns in
            position 0 and validation labels in position 1 (e.g. a tuple
            like (val_x, val_y)).

            tfLoss (str or keras.losses.Loss): a keras Loss object
            specifying the loss function to use.

            tfOptimizer (str or keras.optimizers.Optimizer): a tensorflow
            optimizer to use for model training (default is 'adam').

            batchSize (int): batch size to use for training (default is 64).

            trainingEpochs (int): maximum number of training iterations 
            (default is 20).

            tfCallbacks (list of Callable): tensorflow training callbacks 
            (default is None).

        Returns:
            A tuple containing a single fitness for the given individual.
        """
        
        if str(individual) in self.__evaluationHistory :
            fitness = self.__evaluationHistory[str(individual)]
        else :
            fitness = tfIndividualEvaluation(individual,
                                             paramsGrid,
                                             modelConfig,
                                             trainSet,
                                             validationSet,
                                             tfLoss,
                                             tfOptimizer,
                                             batchSize,
                                             trainEpochs,
                                             tfCallbacks)
                 
            self.__evaluationHistory[str(individual)] = fitness
            
        if self.__evaluationCallbacks != None :
            for cbk in self.__evaluationCallbacks :
                cbk(individual, fitness)
            
        return fitness
    
class DifferentialEvaluatorTF :
    """tfIndividualEvaluation wrapper for differential individuals.
    
    Wraps the tfIndividualEvalutation function for individuals used in
    the differential evolution (real-valued individuals) to provide
    additional functionalities.
    Specifically, this class decodes a real-valued individual into a
    standard individual (according to its corresponding parameters) to
    be able to evaluate it using the tfIndividualEvaluation function. It
    furthermore relies on a local instance of IndividualEvaluatorTF to
    keep track of all the already evaluated individuals.
    """
    
    def __init__(self, evaluationCallbacks=None) :
        """Builds a new differential evaluator object.
        
        Builds a differential evaluator object with a clean evaluation 
        history and the specified callbacks.
        
        Args:
            evaluationCallbacks (list of callable): list of callback
            functions to invoke after individual evaluation. Each
            callback must accept an individual as first argument and its
            fitness as second argument (default is None).
        """
        
        self.__baseEvaluator = IndividualEvaluatorTF(evaluationCallbacks)
        
    def __call__(self, individual, paramsGrid, modelConfig, trainSet, validationSet, tfLoss, tfOptimizer='adam', batchSize=64, trainEpochs=20, tfCallbacks=None) :
        """Evaluates a given individual from the corresponding model.
        Redirects to the tfIndividualEvaluation function.

        First, the specified individual, which is expected to be a real-
        valued individual, is transformed into a standard individual
        according to the provided parameters. Then, if the corrected
        individual has not been already evaluated in the past, its
        fitness value is computed via the tfIndividualEvaluation
        function with the provided parameters. If the specified
        individual has already been evaluated in the past, its
        previously computed fitness value is returned immediately.

        Args:
            individual: an individual (typically a sequence of genes).

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
            returned by model.get_config()

            trainSet (tuple): a tuple containing training patterns in
            position 0 and training labels in position 1 (e.g. a tuple like
            (train_x, train_y)).

            validationSet (tuple): a tuple containing validation patterns in
            position 0 and validation labels in position 1 (e.g. a tuple
            like (val_x, val_y)).

            tfLoss (str or keras.losses.Loss): a keras Loss object
            specifying the loss function to use.

            tfOptimizer (str or keras.optimizers.Optimizer): a tensorflow
            optimizer to use for model training (default is 'adam').

            batchSize (int): batch size to use for training (default is 64).

            trainingEpochs (int): maximum number of training iterations 
            (default is 20).

            tfCallbacks (list of Callable): tensorflow training callbacks 
            (default is None).

        Returns:
            A tuple containing a single fitness for the given individual.
        """
        
        correctedIndividual = utilities.differentialIndividualCorrection(individual, paramsGrid.values())
        
        return self.__baseEvaluator(correctedIndividual,
                                    paramsGrid,
                                    modelConfig,
                                    trainSet,
                                    validationSet,
                                    tfLoss,
                                    tfOptimizer,
                                    batchSize,
                                    trainEpochs,
                                    tfCallbacks)