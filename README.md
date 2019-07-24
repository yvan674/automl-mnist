# AutoML Practice
The point of this project is to practice and get comfortable with the optimizers built by the Uni Freiburg AutoML team.
For this project, we will be optimizing a network built to classify the MNIST database.

## AutoML
AutoML is a group from University of Freiburg dedicated to researching methods to automate machine learning optimization and network creation.

## Hyperparameters
We will be optimizing the following hyperparameters:

| Hyperparameter                 | Range                               | Default | Step size | Type  | Comments                                 |
|--------------------------------|-------------------------------------|---------|-----------|-------|------------------------------------------|
| Number of Convolutional layers | [1, 7]                              | 1       | -         | Int   | Number of convolutional layers used      |
| Kernel Size                    | [3, 7]                              | 3       | -         | Int   |                                          |
| Activation Type                | {relu, leaky relu, linear, sigmoid} | ReLU    | -         | Cat   |                                          |
| Per-layer dropout              | [0, 0.99]                           | 0.5     | -         | Float |                                          |
| Fully Connected Layer Units    | [25, 4096]                          | 50     | -         | Int   |                                          |
| Fully Connected Dropout        | [0, 0.99]                           | 0.5     | -         | Float | Final dropout layer value                |
| Batch size                     | [32, 4000]                          | 32      | Log       | Int   |                                          |
| Learning rate                  | [1e-5, 1]                           | 1e-3    | Log       | Float |                                          |
| Optimizer                      | {adagrad, adam, rmsprop, sgd}       | SGD     | -         | Cat   |                                          |
| Adam epsilon                   | [5e-3, 1]                           | 1e-2    | 5e-3      | Float | Only active when Adam is used            |
| SGD/RMSprop momentum           | [0, 1]                              | 0       | 5e-3      | Float | Only active when SGD or RMSprop are used |

| Hyperparameter                 | Variable name
|--------------------------------|---------------------
| Number of Convolutional layers | num_conv_layers
| Kernel Size                    | kernel_size
| Activation Type                | activation
| Per-layer dropout              | per_layer_dropout
| Fully Connect Layer Units      | fc_units
| Fully Connected Dropout        | fc_dropout
| Batch size                     | batch_size
| Learning rate                  | lr
| Optimizer                      | optimizer
| Adam epsilon                   | eps
| SGD/RMSprop momentum           | momentum
## Optimization
We will be using the BOHB optimizer.