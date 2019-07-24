# AutoML Practice
The point of this project is to practice and get comfortable with the optimizers built by the Uni Freiburg AutoML team.
For this project, we will be optimizing a network built to classify the MNIST database.

## AutoML
AutoML is a group from University of Freiburg dedicated to researching methods to automate machine learning optimization and network creation.

## Hyperparameters
We will be optimizing the following hyperparameters:

| Hyperparameter              | Range                               | Default | Step size | Type  | Comments                                 |
|-----------------------------|-------------------------------------|---------|-----------|-------|------------------------------------------|
| Batch size                  | [32, 4000]                          | 32      | Log       | Int   |                                          |
| Number of layers            | [1, 10]                             | 1       | -         | Int   | Number of convolutional layers used      |
| Activation Type             | {ReLU, Leaky ReLu, Linear, Sigmoid} | ReLU    | -         | Cat   |                                          |
| Per-layer dropout           | [0, 0.99]                           | 0.5     | -         | Float |                                          |
| Fully Connected Layer Units | [64, 4096]                          | 128     | -         | Int   |                                          |
| Learning rate               | [1e-5, 1]                           | 1e-3    | Log       | Float |                                          |
| Final Dropout               | [0, 0.99]                           | 0.5     | -         | Float | Final dropout layer value                |
| Optimizer                   | {Adagrad, Adam, RMSprop, SGD}       | SGD     | -         | Cat   |                                          |
| Adam epsilon                | [5e-3, 1]                           | 1e-2    | 5e-3      | Float | Only active when Adam is used            |
| SGD/RMSprop momentum        | [0, 1]                              | 0       | 5e-3      | Float | Only active when SGD or RMSprop are used |

## Optimization
We will be using the BOHB optimizer.