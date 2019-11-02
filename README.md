
# Convolutional Neural Network

This project shows the underlying principle of Convolutional Neural Network (CNN). This is a smart way of processing images especially when there are multiple objects within the image.  With the right filtering and pool size the different objects within the image can be processed and identified for classification.

## Getting Started

The dataset that is being trained is the [Fashion-MNIST dataset by Zalando](https://github.com/zalandoresearch/fashion-mnist). If the Jit library is causing problems (`from numba import jit`) just omit it or comment it out, and remove `@jit` signatures from functions. You can use various GPU optimization methods, reference [here](https://developer.nvidia.com/how-to-cuda-python) for using my program with CUDA from NVIDIA. 


## Prerequisites
TensorFlow is used to load the data, therefore TensorFlow will need to be installed to access the Keras library. Note that the data can be dowloaded on your system if you are having problems utilizing TensorFlow on your system python. This program is written in Python 3.5, so it would be good to use Python 3.5 and above to avoid system compatibility issues. 

### Install

MatplotLib and Numpy libraries are also needed.

Matplotlib library
```
pip install matplotlib
```

Numpy library
```
pip install numpy
```

Keras library
```
pip install Keras
```

## Running the tests

Run `cnn_test.py` and this will generate 3 pickle files with optimal parameters after training. Also for evaluation purposes, the mean squared error and log loss are computed as well as plotted. It is always good to analyze the graphs generated to test for convergence which is a good way to tell if the algorithm is learning. The accuracy is computed and printed for each epoch (iteration) in the console.

You can change learning rate parameter `eta` and the batch size `batch_size` . The skip size`skip_size` is just how much data will be skipped for the total number of training data, either way the training data will always be 80 percent. Therefore test/training ratio is 20/80.

