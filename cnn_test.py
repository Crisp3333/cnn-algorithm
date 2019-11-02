import cnn as cn
import pickle
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist


# Normalize data
def normalize_data(trainX, trainY, testX, testY):
    ((trainX, trainY), (testX, testY)) = fashion_mnist.load_data()
    trainXY = []
    train_max = np.amax(trainX)
    test_max = np.amax(testX)
    if train_max >= test_max:
        trainX = ((trainX + 1) / (train_max + 1e+6))
        testX = ((testX + 1) / (train_max + 1e+6))
    else:
        trainX = ((trainX + 1) / (test_max + 1e+6))
        testX = ((testX + 1) / (test_max + 1e+6))
    for i, key in enumerate(trainX):
        # Initialize a unit vector
        vec = np.zeros((10, 1))
        # Represent the unit vector of the classifier
        # Change value of vec[y_train[i]] to 1
        # y_train[i] is the label value range 0-9
        # hence, the label shall act as the index position
        # on where the "1" should be.
        vec[trainY[i]] = 1
        trainXY.append((key, vec))
    testXY = np.array(list(zip(testX, testY)))
    return trainXY, testXY


def main():
    """""
    Collect data
    """""
    # Get data from keras dataset.
    ((trainX, trainY), (testX, testY)) = fashion_mnist.load_data()
    # Call method to normalize data
    trainXY, testXY = normalize_data(trainX, trainY, testX, testY)
    learning_rate = 44e-1  # learning rate
    # Initialize neural network with training and test data
    cnn = cn.RCNN(trainXY, testXY)
    skip_size = 5000
    # Call stochastic gradient descent method to train data
    cnn.sgd(epoch=100, eta=learning_rate, batch_size=8, skip_size=skip_size)

    """""
    Here we have the options to store optimal parameters from the network.
    We will save the in a pickle file
    """""
    fc_weights, fc_biases = cnn.get_fc_param()
    conv_weights = cnn.get_conv_params()
    with open('fc_weights.pkl', 'wb') as f:
        pickle.dump(fc_weights, f)
    with open('fc_biases.pkl', 'wb') as f:
        pickle.dump(fc_biases, f)
    with open('conv_weights.pkl', 'wb') as f:
        pickle.dump(conv_weights, f)
    fc1, pool_size, conv_size = cnn.params()
    params = {'fc_layer': fc1, 'pool_size': pool_size, 'conv_size': conv_size}
    with open('params.pkl', 'wb') as f:
        pickle.dump(params, f)

    """""
    Below we have collected various metrics for evaluation from the network
    """""
    # Get mse and log loss from network
    mse_cost, log_loss = cnn.learning_data()
    print("Accuracy")
    # Get accuracy
    accuracy = cnn.accuracy()
    print(accuracy)
    # MEAN SQUARED ERROR metrics
    print("MSE costs below")
    print(mse_cost)
    plt.plot(mse_cost)
    plt.title('MSE cost Learning Curve for CNN')
    plt.ylabel("MSE Error")
    plt.xlabel("epoch")
    plt.show()
    # Log loss
    print("log loss below")
    print(log_loss)
    plt.plot(log_loss)
    plt.title('Log Loss Learning Curve for CNN')
    plt.ylabel("Log Loss")
    plt.xlabel("epochs")
    plt.show()


if __name__ == "__main__":
    main()


