English | [简体中文](README.zh-CN.md)

# A Multilayer Perceptron (MLP) from Scratch with NumPy (NumPy-Keras)
<p align="center">
  <img src="figures/numpy_keras.jpg" width="20%">
  <br>
  <b>NumPy-Keras</b>
</p>

**NumPy-Keras**, originally named **NumPyMultilayerPerceptron**, is a library for implementing a multilayer perceptron (MLP) using `numpy`. Its purpose is to provide a simple and easy-to-understand implementation, aimed at facilitating learning and teaching.

<p align="center">
  <img src="figures/MLP.jpg">
  <br>
  <b>Figure 1.</b> Multilayer Perceptron
</p>

> [!IMPORTANT]
> Major Update: A Completely New Implementation
- We have refactored the entire library to make it easier to understand and use.
    - The latest release offers an implementation that only requires the `Python` standard library and depends on `numpy`.
    - No other libraries need to be installed, even common ones like `scipy` or `scikit-learn`.
- We provide an interface that is closer to `Keras` for better usability.
- We have optimized the code for better performance, including improvements in numerical stability, coding style, and security.
- To enhance the learning experience, we have added extra features, including:
  - A progress bar (requires the `tqdm` library)
  - A training history plot (requires the `matplotlib` library)
  - Automatic differentiation (requires the `autograd` library)
  - If you are only interested in the specifics of the multilayer perceptron implementation, you can use just the `numpy` library, with lazy imports and exception handling to avoid errors if the necessary libraries are missing. <u> **Again, emphasize**: You only need the `numpy` library to run this framework.</u>

## Table of Contents
- [0. Quick Start](#sparkles-0-quick-start)
- [1. Introduction](#sparkles-1-introduction)
- [2. Dependencies](#sparkles-2-dependencies)
- [3. Testing on Other Datasets](#sparkles-3-testing-on-other-datasets)
  - [3.1 Toy Example: Random Dataset](#31-toy-example-random-dataset)
    - [3.1.1 Basic Model](#311-basic-model)
    - [3.1.2 SGD Optimizer](#312-sgd-optimizer)
    - [3.1.3 Adam Optimizer](#313-adam-optimizer)
    - [3.1.4 The Effect of Dropout](#314-the-effect-of-dropout)
    - [3.1.5 The Effect of BatchNormalization](#315-the-effect-of-batchnormalization)
    - [3.1.5 The Effect of Batch Size on BatchNormalization](#315-the-effect-of-batch-size-on-batchnormalization)
    - [3.1.6 Classification Problem](#316-classification-problem)
  - [3.2 Multi-Class Classification Problem](#32-multi-class-classification-problem)
    - [3.2.1 Load the Dataset](#321-load-the-dataset)
    - [3.2.2 Build the Model](#322-build-the-model)
    - [3.2.3 Compile the Model](#323-compile-the-model)
    - [3.2.4 Train the Model](#324-train-the-model)
    - [3.2.5 Visualize the Training History](#325-visualize-the-training-history)
- [4. Key Modules](#sparkles-4-key-modules)
    - [4.1 Activations](#41-activations)
    - [4.2 Layers](#42-layers)
    - [4.3 Optimizers](#43-optimizers)
    - [4.4 Callbacks](#44-callbacks)
- [5. Conclusion](#sparkles-5-conclusion)
- [6. Version Log](#sparkles-6-version-Log)

## :sparkles: 0. Quick Start
- Clone the repository.

```bash
git clone https://github.com/XavierSpycy/NumPy-Keras.git
cd NumPy-Keras
```

- Create a virtual environment.

```bash
conda create -n numpy_keras python=3.12 -y
```

- Activate the virtual environment.

```bash
conda activate numpy_keras
```

- Install dependencies.

```bash
pip install -r requirements.txt
```

To avoid installing extra dependencies for additional features, we have commented out the non-`numpy` dependencies. If you need these features, you can uncomment the lines and rerun `pip3 install -r requirements.txt`.

If you are using miniconda, you may also need to install dependencies for `Jupyter Notebook`.

```bash
pip3 install jupyter ipywidgets
```

- Finally, you can learn how to use the library through the Jupyter Notebooks in the notebooks folder.

## :sparkles: 1. Introduction
Multi-layer Perceptron (MLP) is one of the most basic neural network models. It consists of an input layer, one or more hidden layers, and an output layer. Each layer consists of multiple neurons, each with an activation function. An MLP is a feedforward neural network, and its output is calculated by forward propagation from the input layer to the output layer.

The current mainstream deep learning frameworks, such as `TensorFlow`, `PyTorch`, etc., provide efficient implementations, but the underlying implementations are complex and difficult to understand. Therefore, to better understand the principles of deep learning, we provide an implementation of a multi-layer perceptron framework using `NumPy`.

We mentioned `Keras` because our implementation was inspired by the `Keras` interface. From the interface perspective, `Keras` provides a high-level interface that allows users to easily build neural network models, making it very suitable for beginners because it is simple and easy to understand. For this reason, `TensorFlow` 2.0 and later versions also use `Keras` as their high-level interface.

When we open the [official website](https://www.tensorflow.org/) of `TensorFlow`, we can see the following code example:

```python
import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
```

Meanwhile, we can see the following code example in our framework:

```python
import numpy_keras as keras

X_train, y_train, X_test, y_test = mnist_load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu', kernel_initializer='he_normal'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=5, verbose=1)
print(f"Accuracy on the training set: {model.evaluate(X_train, y_train):.2%}")
print(f"Accuracy on the test set: {model.evaluate(X_test, y_test):.2%}")
# Outputs:
# Accuracy on the training set: 98.52%
# Accuracy on the test set: 98.52%
```

We can notice that our framework is very similar to the `Keras` interface, which allows beginners to refer to a more mature framework and build and train neural network models by themselves.

Detailed usage can be found in our [Jupyter Notebook](notebooks/beginner.ipynb).

In addition to the basic implementation, we also provide some additional features, such as:
- A progress bar during training to better monitor the training process (requires the `tqdm` library).
- A function to plot the training history when we train the model to better visualize the training process (requires the `matplotlib` library).
  - Use `keras.plot_history(history)` to plot the training history.

<p align="center">
  <img src="figures/history.png">
  <br>
  <b>Figure 2.</b> Training History
</p>

To conclude, our framework is a lightweight framework that only depends on the `numpy` library and provides a simple and easy-to-understand implementation. We hope that this framework can help users better understand the principles of deep learning and better use deep learning frameworks.

## :sparkles: 2. Dependencies
Due to the fact that we have not conducted extensive testing, we cannot guarantee that the library will work on specific versions of `Python` and `numpy`. However, we list our development environment for reference:
- Python 3.12.1
- numpy 1.26.4

Although we expect to achieve the most lightweight implementation, we still provide some additional features for better usability. For example, we provide a progress bar during training to better monitor the training process (requires the `tqdm` library); we also provide a function to plot the training history to better visualize the training process (requires the `matplotlib` library).

We have implemented these features using lazy loading, so you only need `numpy` to run this library when you don't need these features.

To avoid errors, we recommend using our environment or an environment similar to ours. 

Here are the versions of `tqdm`, `matplotlib`, and `autograd`:
- tqdm 4.66.1
- matplotlib 3.8.2
- autograd 1.7.0

If there are no major differences in the versions, we believe that this library should work on other versions as well.

## :sparkles: 3. Testing on Other Datasets
Perhaps you have a question: We only tested on the MNIST dataset, and our model performed very well in terms of accuracy. But is it possible that our model overfits on the MNIST dataset? How does it perform on other datasets?

We are going to test our model on a simple random dataset and a slightly more complex ten-classification problem to better understand how our model performs on other datasets.

### 3.1 Toy Example: Random Dataset
When introducing this module, we may involve some core features. If you have any questions about these concepts, we will explain them in detail in the following sections to help you better understand them.

We will test on a simple random dataset. You can view more details by visiting our [Jupyter Notebook](notebooks/toy-example.ipynb).

```python
import numpy as np
import matplotlib.pyplot as plt
```

To ensure the reproducibility of our experiments and the comparability between different models, we set a random seed.

```python
np.random.seed(3407)

y_1 = np.hstack([np.random.normal(1, 1, size=(100, 2)),  np.ones(shape=(100, 1))])
y_2 = np.hstack([np.random.normal(-1, 1, size=(40, 2)), -np.ones(shape=(40, 1))])
dataset = np.vstack([y_1, y_2])

X_train, y_train = dataset[:, 0:2], dataset[:, 2]
```

Following the construction of the dataset, we will visualize the dataset.

<p align="center">
  <img src="figures/dataset.png">
  <br>
  <b>Figure 3.</b> Random Dataset
</p>

Now, we can import our framework to witness the performance of our model on this dataset.

```python
import numpy_keras as keras
```

To better visualize the decision boundary, we provide a function to plot the decision boundary.

```python 
def plot_decision_boundary(model, X_train, y_train):
    xx, yy = np.meshgrid(np.arange(-2, 2, .02), np.arange(-2, 2, .02))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(15,7))
    plt.subplot(1, 2, 1)
    plt.pcolormesh(xx, yy, Z>0, cmap='cool')
    plt.scatter(X_train[:, 0], X_train[:, 1], c=[(['b', 'r'])[int(d>0)] for d in y_train], s=100)
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.grid()
    plt.title('Labels')
    plt.subplot(1, 2, 2)
    plt.pcolormesh(xx, yy, Z>0, cmap='cool')
    plt.scatter(X_train[:, 0], X_train[:, 1], c=[(['b', 'r'])[int(d>0)] for d in model.predict(X_train)], s=100)
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.grid()
    plt.title('Predictions')
```

#### 3.1.1 Basic Model
We will use a simple model to solve this problem.

```python
layers = [
    keras.layers.Input(2),
    keras.layers.Dense(3, activation="relu", kernel_initializer='he_normal'),
    keras.layers.Dense(1, activation='tanh')
]
```

Although we construct a dataset for a classification problem, in fact, any classification problem can be transformed into a regression problem. We will use the Mean Squared Error (MSE) as the loss function and use $R^2$ as the evaluation metric.

#### 3.1.2 SGD Optimizer
To begin with, we will use the Stochastic Gradient Descent (SGD) optimizer, which is the first optimizer that most beginners encounter and is widely used. To compare the performance under the same learning rate, we will not use the string form to pass the optimizer, but pass an instance of the optimizer class and set the learning rate to $1 \times 10^{-3}$ (usually the default value of the Adam optimizer).

```python
model = keras.Sequential(layers)
model.compile(loss='mse', optimizer=keras.optimizers.SGD(1e-3), metrics=['r2_score'])
history = model.fit(X_train, y_train, batch_size=2, epochs=500, verbose=1)
keras.plot_history(history)
```

<p align="center">
  <img src="figures/history_sgd.png">
  <br>
  <b>Figure 4.</b> Training History (with SGD Optimizer)
</p>

<p align="center">
  <img src="figures/db_sgd.png">
  <br>
  <b>Figure 5.</b> Decision Boundary (with SGD Optimizer)
</p>

#### 3.1.3 Adam Optimizer
Next, we will use a more advanced optimizer, the Adam optimizer, to train our model. The Adam optimizer adjusts the learning rate for each parameter by computing the first and second moments of the gradients. In simpler terms, the Adam optimizer is an adaptive learning rate optimizer. It allows us to converge faster and makes it easier to adjust the learning rate.

```python
model.compile(loss='mse', optimizer='adam', metrics=['r2_score'])
```

<p align="center">
  <img src="figures/history_adam.png">
  <br>
  <b>Figure 6.</b> Training History (with Adam Optimizer)
</p>

<p align="center">
  <img src="figures/db_adam.png">
  <br>
  <b>Figure 7.</b> Decision Boundary (with Adam Optimizer)
</p>

It is worth noting that the decision boundary has an interesting "corner".

#### 3.1.4 The Effect of Dropout
We will use a model containing a Dropout layer to solve this problem. Dropout is a regularization technique that randomly drops a portion of neurons during training to reduce overfitting.

```python
layers = [
    keras.layers.Input(2),
    keras.layers.Dense(3, activation="relu", kernel_initializer='he_normal'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1, activation='tanh')
]
```
<p align="center">
  <img src="figures/history_dropout.png">
  <br>
  <b>Figure 8.</b> Training History (with Dropout)
</p>

<p align="center">
  <img src="figures/db_dropout.png">
  <br>
  <b>Figure 9.</b> Decision Boundary (with Dropout)
</p>

#### 3.1.5 The Effect of BatchNormalization
We will use a model containing a BatchNormalization layer to solve this problem. BatchNormalization is a technique used in neural networks to standardize the activations of a given input layer in small batches, helping to normalize and accelerate the training process. In common cognition, we usually follow the order of `Linear -> BatchNormalization -> Activation -> Dropout`. Therefore, we will try this order here.

```python
layers = [
    keras.layers.Input(2),
    keras.layers.Dense(3, activation=None),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1, activation='tanh')
]
```

<p align="center">
  <img src="figures/history_bn.png">
  <br>
  <b>Figure 10.</b> Training History (with BatchNormalization)

<p align="center">
  <img src="figures/db_bn.png">
  <br>
  <b>Figure 11.</b> Decision Boundary (with BatchNormalization)
</p>

Unfortunately, the BatchNormalization layer causes the loss function to oscillate sharply, and from the decision boundary, the model's performance is very poor. This is because our batch size is too small for the BatchNormalization layer to work properly. This is also a drawback of the BatchNormalization layer, as it is very sensitive to the batch size.

#### 3.1.5 The Effect of Batch Size on BatchNormalization
In terms of the sensitivity of the BatchNormalization layer to the batch size, we will try different batch sizes to better understand the sensitivity of the BatchNormalization layer to the batch size.

```python
history = model.fit(X_train, y_train, batch_size=16, epochs=500, verbose=1)
```

<p align="center">
  <img src="figures/history_bn_16.png">
  <br>
  <b>Figure 12.</b> Training History (with BatchNormalization, batch size = 16)
</p>

<p align="center">
  <img src="figures/db_bn_16.png">
  <br>
  <b>Figure 13.</b> Decision Boundary (with BatchNormalization, batch size = 16)
</p>

As we can see, when the batch size is 16, the loss function stability of the BatchNormalization layer has improved, and the decision boundary looks more reasonable. However, due to the small size of our dataset, we cannot solve the problem of the BatchNormalization layer by increasing the batch size, nor can we explore the sensitivity of the BatchNormalization layer to the batch size in more depth. We hope this will provide a starting point for further exploration.

#### 3.1.6 Classification Problem
We have always considered this problem as a regression problem, but in fact, it is a classification problem. We will use a model containing a Softmax layer and use the cross-entropy loss function to solve this problem. By the way, we have always considered this problem as a regression problem, but in fact, it is a classification problem. We will use a model containing a Softmax layer and use the cross-entropy loss function to solve this problem. By the way, our `Sequential` class adds layers through the `add` method, which is very similar to the `Sequential` class in `Keras`.

```python
model = keras.Sequential()
model.add(keras.layers.Input(2))
model.add(keras.layers.Dense(3, activation='relu', kernel_initializer='he_normal'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(2, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

<p align="center">
  <img src="figures/history_classification.png">
  <br>
  <b>Figure 14.</b> Training History (Classification Problem)
</p>

<p align="center">
  <img src="figures/db_classification.png">
  <br>
  <b>Figure 15.</b> Decision Boundary (Classification Problem)
</p>

### 3.2 Multi-Class Classification Problem
In this section, we will test on a slightly more complex ten-classification problem.

Since we already have test data in our dataset, we will not need to split the dataset. However, in real-world scenarios, we usually split the dataset into a training set, a validation set, and a test set. Typically, the validation set and the test set are often confused, although strictly speaking, this is incorrect, especially in production environments, where the test set often does not have real labels and may also have a different distribution from the training data.

To better understand the performance of the model and the performance that the model can achieve, we will treat the test set as both the test set and the validation set. The former is used to simulate the test set as a separate set, while the latter is treated as a validation set split from the training set.

You can view more details by visiting our [Jupyter Notebook](notebooks/assignment.ipynb).

#### 3.2.1 Load the Dataset
```python
import numpy as np

np.random.seed(42)

X_train = np.load('data/train_data.npy')
y_train = np.load('data/train_label.npy').squeeze()
X_test = np.load('data/test_data.npy')
y_test = np.load('data/test_label.npy').squeeze()
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
```

In general, our dataset contains 50,000 training samples and 10,000 test samples, with a feature dimension of 128.

#### 3.2.2 Build the Model
Here, we use `numpy_keras` to build a multi-layer perceptron (MLP) model. We will use a model with 12 hidden layers. We will use the `ELU` activation function and use the `He` uniform initializer to initialize the weights. We will also use the `Dropout` layer to reduce overfitting.

```python
import numpy_keras as keras

model = keras.Sequential()
model.add(keras.layers.Input(shape=X_train.shape[1]))
model.add(keras.layers.Dense(120, activation='elu', kernel_initializer='he_uniform'))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Dense(112, activation='elu', kernel_initializer='he_uniform'))
model.add(keras.layers.Dropout(0.20))
model.add(keras.layers.Dense(96, activation='elu', kernel_initializer='he_uniform'))
model.add(keras.layers.Dropout(0.15))
model.add(keras.layers.Dense(64, activation='elu', kernel_initializer='he_uniform'))
model.add(keras.layers.Dropout(0.10))
model.add(keras.layers.Dense(32, activation='elu', kernel_initializer='he_uniform'))
model.add(keras.layers.Dense(24, activation='elu', kernel_initializer='he_uniform'))
model.add(keras.layers.Dense(16, activation='elu', kernel_initializer='he_uniform'))
model.add(keras.layers.Dense(10, activation='softmax'))
```

In this section, we also provide the model architecture we used. The selected architecture is based on personal experience with deep learning tasks in the past, **and by no means represents an optimal architecture in any practical sense**. We strongly recommend that you choose the appropriate model architecture based on your task requirements and dataset characteristics. Feel free to try different architectures and choose the best one based on experimental results.

<p align="center">
  <img src="figures/optimal.svg">
  <br>
  <b>Figure 16.</b> A Complex Model Architecture
</p>

#### 3.2.3 Compile the Model
We use the `Adam` optimizer to compile our model, and use `SparseCategoricalCrossentropy` as the loss function, as well as `Accuracy` as the evaluation metric. Additionally, we will use the `EarlyStopping` and `ReduceLROnPlateau` callback functions to improve model performance.

```python
early_stop = keras.callbacks.EarlyStopping('val_accuracy', mode='max', patience=5, restore_best_weights=True)
lr_scheduler = keras.callbacks.ReduceLROnPlateau('val_accuracy', mode='max', factor=0.5, patience=3, min_lr=1e-6)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

#### 3.2.4 Train the Model
##### Test Set as Test Set

```python
history = model.fit(X_train, y_train, epochs=60, batch_size=128, verbose=1, callbacks=[early_stop, lr_scheduler], validation_split=0.1)
```

- Performance
    - Accuracy (Training Set): 62.14%
    - Accuracy (Test Set): 55.35%
- Early Stop Epoch: 33

##### Test Set as Validation Set

```python
history = model.fit(X_train, y_train, epochs=60, batch_size=128, verbose=1, callbacks=[early_stop, lr_scheduler], validation_data=(X_test, y_test))
```

- Performance
    - Accuracy (Training Set): 64.98%
    - Accuracy (Test Set): 56.02%
- Early Stop Epoch: 51

#### 3.2.5 Visualize the Training History

```python
keras.plot_history(history)
```

<p align="center">
  <img src="figures/assn_output.png">
  <br>
  <b>Figure 17.</b> Loss Function and Accuracy Curve
</p>

## :sparkles: 4. Key Modules
### 4.1 [Activations](numpy_keras/activations)
Given the shortcomings of markdown in rendering mathematical formulas, we will only list the functions we have implemented here, rather than displaying the mathematical formulas. However, we highly recommend the [documentation](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity) of `PyTorch`. It provides excellent formulas and figures to help you better understand the activation functions. Our implementation refers to this documentation. We would also like to thank the `PyTorch` development team for their contributions to the open-source community.

- ELU
- Hardshrink
- Hardsigmoid
- Hardtanh
- LeakyReLU
- LogSigmoid
- ReLU
- ReLU6
- SELU
- CELU
- Sigmoid
- Softplus
- Softshrink
- Softsign
- Tanh
- Softmax

In addition to the above activation functions, we have implemented more activation functions in our `autograd` module, which is used for automatic differentiation. In our `autograd` module, we have implemented more activation functions in addition to the above activation functions:

- Hardswish
- GELU
- SiLU
- Mish
- Softsign
- Tanhshrink
- Threshold

For the use of automatic differentiation (`autograd`), you only need to make the following changes to the code:

Replace the following import statement:
```python
import numpy_keras as keras
```
with:
```python
import numpy_keras.autograd as keras
```

you can seamlessly switch to the functionality of `autograd`.

### 4.2 Layers
- **[Dense](numpy_keras/layers/dense.py)**
    -  Fully Connected Layer
    - **Definition**:  
        A dense layer, also known as a fully connected layer, is a layer in a neural network where all input nodes (or neurons) are connected to every output node. It is called "dense" or "fully connected" because all inputs and outputs are connected to each other.
    - **Mathematical Representation**:     
        $y = f(Wx + b)$
    
- **[BatchNormalization](numpy_keras/layers/batch_norm.py)**
    - **Definition**:     
        Batch normalization is a technique used in neural networks to standardize the activations of a given input layer in small batches, helping to normalize and accelerate the training process. 
    - **Mathematical Representation**:    
        For a given mini-batch, $B$, of size $m$, with activations $x$:

        $\mu_B = \frac{1}{m}\Sigma_{i=1}^m x_i$      
        $\sigma_B^2 = \frac{1}{m}\Sigma_{i=1}^m (x_i - \mu_B)^2$      
        $\hat {x_i} = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$        
        $y_i = \gamma \hat {x_i} + \beta$

- **[Dropout](numpy_keras/layers/dropout.py)**:
    - **Definition**: 
        Dropout is a regularization technique used in neural networks, where a random subset of neurons is "dropped out" (i.e., set to zero) during each iteration. This prevents the network from becoming too dependent on any specific neuron and encourages a more generalized model.   

- **[Activation](numpy_keras/layers/activation.py)**:
    - **Definition**:
        An activation layer in a neural network is a layer that applies a non-linear function to its input, transforming the data to introduce non-linearity into the model. This non-linearity allows the network to learn and make adjustments from errors, which is crucial for learning complex patterns.

- **[Flatten](numpy_keras/layers/flatten.py)**:
    - **Definition**:
        The flatten layer is a layer used in neural networks to flatten the input data into a one-dimensional array. This is useful when connecting a convolutional layer to a fully connected layer, as the fully connected layer requires a one-dimensional input.

- **[Input](numpy_keras/layers/input.py)**:
    - **Definition**:
        The input layer is the first layer in a neural network, which receives input data and passes it to the next layer. The input layer has no weights or biases and is used to define the shape of the input data.

### 4.3 Optimizers
Right here, we have implemented the optimizers commonly used by beginners to help them better understand the optimization process of deep learning. We believe that these optimizers are very helpful for beginners to understand the optimization process of deep learning.

- [SGD (with Momentum & Nesterov)](numpy_keras/optimizers/sgd.py)
- [Adagrad](numpy_keras/optimizers/adagrad.py)
- [Adadelta](numpy_keras/optimizers/adadelta.py)
- [Adam](numpy_keras/optimizers/adam.py)

### 4.4 Callbacks
- [EarlyStopping](numpy_keras/callbacks/early_stopping.py): When the validation loss no longer decreases, stop training.
- [Learning Rate Scheduler](numpy_keras/callbacks/lr_scheduler.py): Adjust the learning rate during training.
  - MultiplicativeLR
  - StepLR
  - MultiStepLR
  - ConstantLR
  - LinearLR
  - ExponentialLR
  - PolynomialLR
  - CosineAnnealingLR
  - ReduceLROnPlateau

> [!NOTE]
> We believe that the interface of `Keras` regarding learning rate schedulers is not very elegant. From the perspective of usability, it is very worthwhile to optimize the interface of `Keras` regarding learning rate schedulers. Although our implementation in this part did not meet our expected standards, we still decided to adjust the implementation of `LearningRateScheduler` to improve the rationality and usability of the interface. In other words, this part of the implementation is quite different from `Keras`. We also believe that the `torch.optim.lr_scheduler` module of `PyTorch` is very worth learning from. It provides many learning rate schedulers and has a very elegant interface. Therefore, this part of the implementation is mainly inspired by `PyTorch`. Thanks again!
> 
> In addition, although the parameters of the learning rate scheduler are slightly adjusted, the overall interface is very similar to `Keras`. We believe that such adjustments are reasonable because they make the interface more user-friendly and more intuitive, especially for beginners.

## :sparkles: 5. Conclusion
`NumPy-Keras` is a multi-layer perceptron (MLP) library implemented using `NumPy`. It aims to provide a simple and easy-to-understand neural network implementation to help users learn and teach deep learning. The library does not depend on any third-party deep learning frameworks such as `TensorFlow` or `PyTorch` and only requires `Python` and `NumPy` to run, making it suitable for beginners to understand the basic principles of deep learning. `NumPy-Keras` provides an API similar to `Keras` and supports features such as visualization of the training process, progress bar display, and training history charts. With this lightweight framework, users can quickly build and train MLP models and explore the basic concepts of deep learning.

## :sparkles: 6. Version Log
- v1
  - v1.0.0 
    - 2023.08.27: Initial release.
  - v1.1.0
    - 2023.09.05
      - **feat**：Integrate the `tqdm` library's `tqdm()` function into the training process.
      - **refactor**：Enhance the `Sequential` class to support the `add` method.
      - **perf**：Improve the performance of the `MultilayerPerceptron` class.
    - 2023.10.02
      - **feat**：Released the dataset.
      - **docs**：Fixed several typos in `README.md`.
    - 2023.10.07
      - **feat**：Introduced a `Jupyter Notebook` named `mlp_quickstart.ipynb` for smooth integration with `Google Colab`.
      - **docs**：For a clearer and better user browsing experience, and integrated emojis and polished content to improve the `README.md`.
  - v1.2.0 
    - 2023.10.18
      - **refactor**：Streamlined the codebase and restructured the codebase to ensure a clearer and more intuitive user experience.
      - **feat**：Successfully integrated the architecture construction and performance testing using the built-in datasets of `Google Colab`.
      - **perf**：Introduced the additional methods to the `MultilayerPerceptron`.
      - **feat**：Integrated `EarlyStopping` and `LearningRateScheduler` to provide refined training control.
      - **docs**：Completed the docstrings for some modules.
  - v1.2.1
    - 2023.11.22
      - **style**: Simplified the implementation, especially the user interface.
      - **perf**: Augmented some functions.
    - 2023.11.24
      - **docs**: Completed the docstrings for all modules, enhancing the clarity and understanding of the entire codebase.
- v2
  - v2.0.0 
    - 2024.12.25
      - **refactor**: Refactored the entire codebase to improve code quality and maintainability.
        - **depd**: Employed pure `numpy` implementation to reduce dependencies on other libraries.
        - **style**: Implemented a more `Keras`-like interface for better usability.
        - **perf**: Optimized the stability of numerical values, code style, and security.
        - **feat**: Offered an automatic training history plot for better monitoring of the training process.
        - **feat**: Provided more callback functions for better model performance optimization.
        - **feat**: Supplied the `autograd` functionality for automatic gradient calculation, avoiding manual calculation, and provided more activation functions.
        - **docs**: Completed the docstrings for all modules, enhancing the clarity and understanding of the entire codebase.