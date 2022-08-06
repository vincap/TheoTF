# **TheoTF: Tuning Hyperparameters using Evolutionary Optimization on TensorFlow.**

TheoTF is a Python package implementing evolutionary algorithms for hyperparameter tuning of neural networks built with Keras and Tensorflow. At the moment only models from the Sequential API are supported.

TheoTF has an extremely simple interface that allows users to optimize their models without caring about evolutionary algorithms at all. TheoTF provides two different strategies based on standard evolution and differential evolution. Both strategies are ready to be used out-of-the-box and can also be customized by the user thanks to the modularity of TheoTF and thanks to [DEAP](https://github.com/DEAP/deap), an evolutionary framework on which TheoTF is built.

## Installation.
You can download TheoTF from its GitHub repo. Main package can be found under the `ProjectTheo/theotf` directory.

## Requirements.
TheoTF has been developed and tested with Python 3.9. The following list reports the additional packages needed to use TheoTF and which version was used to develop and test TheoTF.
- DEAP (version 1.3.1)
- Numpy (version 1.23.1)
- Tensorflow (2.9.1)

## Usage Example.
To start using TheoTF, the `optimizers` and `ptypes` modules are all you need.
- `optimizers`: provides a ready-to-use interface for optimization strategies based on standard evolution or differential evolution.
- `ptypes`: provides custom data types used to specify which hyperparameters TheoTF should optimize.

The following example shows how to optimize a simple full-connected network with dropout to classify the MNIST digits dataset. A more detailed version can be found in the provided [example notebook](https://github.com/vincap/TheoTF/blob/main/ProjectTheo/TheoTF_Example.ipynb).

```python
from theotf import optimizers, ptypes
from tensorflow import keras

# Loading and normalizing data.
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

# 80/20 train-validation split.
x_val, y_val = x_train[48000:], y_train[48000:]
x_train, y_train = x_train[:48000], y_train[:48000]

earlyStoppingCBK = keras.callbacks.EarlyStopping(
    patience=3,
    mode='min',
    restore_best_weights=True
)

# Reference architecture for the model we will optimize.
refModel = keras.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(units=32, name='Dense1'),
    keras.layers.Dropout(rate=0.3, name='Dropout1'),
    keras.layers.Dense(units=32, name='Dense2'),
    keras.layers.Dropout(rate=0.3, name='Dropout2'),
    keras.layers.Dense(units=10, name='Dense3')
])

# Dictionary used to specify which hyperparameters to optimize
paramsGrid = {

    'Dense1.units': ptypes.DRange(32, 128),
    'Dense1.activation': ptypes.Categorical([keras.activations.relu, keras.activations.tanh, keras.activations.sigmoid]),
    'Dropout1.rate': ptypes.CRange(0.3, 0.6),
    'Dense2.units': ptypes.DRange(32, 256),
    'Dense2.activation': ptypes.Categorical([keras.activations.relu, keras.activations.tanh, keras.activations.sigmoid]),
    'Dropout2.rate': ptypes.CRange(0.3, 0.6)
}

# We create a standard evolutionary optimizer
# You can also use a differential evolutionary optimizer
stdEvoOpt = optimizers.StandardEvoOpt(refModel, paramsGrid, populationSize=8, maxGenerations=20)

# Launch optimization (this will take some time to complete)
stdResults = stdEvoOpt.fit(
    train=(x_train, y_train),
    validation=(x_val, y_val),
    tfLoss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    tfCallbacks=[earlyStoppingCBK]
)

# Retrieve the model with the best found configuration
# THIS MODEL MUST BE RETRAINED
optModel = stdResults.bestModel
```

## Acknowledgments
TheoTF was developed as an university exam project by Vincenzo Capone, University of Naples 'Parthenope'. The code is provided as is, with no guarantees except that bugs are almost surely present.