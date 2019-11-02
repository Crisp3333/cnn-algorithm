import math
import random
import numpy as np
from numba import jit


class RCNN:
    fc0 = 324
    fc2 = 10
    error = 0
    mse_error = 0
    log_loss_error = 0
    mse_cost = []
    log_loss_cost = []
    cost = []
    num_layers = 5
    rep_layers = 1
    conv_filter_size = 12  # 5x5;00
    pool_filter_size = 4  # 2x2
    conv_rep = 1
    acc = []

    def __init__(self, train, test):
        self.train = train
        self.test = test
        # The middle layers inclucdes convolutional, layer maxpool and relu layer
        self.fc1 = (28 + (self.conv_rep*(self.rep_layers*(-self.conv_filter_size + 1))
                          + self.rep_layers*(-self.pool_filter_size + 1)))**2
        # Rep layers + input(1) + flatten (1) + output(1)
        self.sections = self.rep_layers + 3
        sizes_conv = [self.conv_filter_size]
        config_layers = [j for i in range(0, self.rep_layers) for j in sizes_conv]
        self.conv_weights = [np.random.randn(x, x) for x in config_layers]
        cb = []
        x = 28
        for i in range(0, self.rep_layers):
            x = x - self.conv_filter_size + 1
            cb.append(x)
            x = x - self.pool_filter_size + 1
        self.fc_weights = np.random.randn(self.fc2, self.fc1)
        self.fc_biases = np.random.randn(self.fc2, 1)

    # Stochastic gradient descent
    def sgd(self, epoch, eta, batch_size, skip_size):
        train_size = (60000.0/skip_size)*batch_size  # represents 80 percent
        print("train size: ", train_size)
        test_size = (train_size*20.0)/80.0
        print("test size calculated: ", test_size)
        test_skip = int(math.ceil((10000.0/test_size)))
        random.shuffle(self.test)
        test = [self.test[k] for k in range(0, len(list(self.test)), test_skip)]
        print("test size actual: ", len(list(test)))
        actual_test_size = len(list(test))
        # Print percentage of training data to testing data
        print("train percentage: {}, test percentage: {}".format(
            train_size/(train_size+actual_test_size)
            , actual_test_size/(train_size+actual_test_size))
        )
        for i in range(epoch):
            # Shuffle data
            random.shuffle(self.train)
            # Create batch
            mini_batch = [self.train[k:k+batch_size] for k in range(0, len(self.train), skip_size)]
            avg_log_loss = []  # Get average log los
            avg_mse = []  # Get ave MSE
            for batch in mini_batch:
                avg_mse_batch = []
                avg_log_batch = []
                for x, y in batch:
                    self.backprop(x, y, eta)
                    avg_mse_batch.append(self.mse_error)
                    avg_log_batch.append(self.log_loss_error)
                avg_mse.append(np.mean(avg_mse_batch))
                avg_log_loss.append(np.mean(avg_log_batch))
            self.evaluation(test_data=test, it=i)
            self.log_loss_cost.append(np.mean(avg_log_loss))
            self.mse_cost.append(np.mean(avg_mse))

    @jit
    def feed_forward(self, x):
        a = x
        for i in range(self.rep_layers):
            # convolutional layer
            a = self.convolutional_layer(inp=a, filter_weight=self.conv_weights[i])
            # ReLU layer
            a = self.selu(inp=a)
            # Maxpool layer
            a = self.maxpool_layer(inp=a, pool_filter_size=self.pool_filter_size)
        # Reformat data for flattening layer
        a = a.flatten('F')
        a_shape = a.shape
        # Flatten layer
        a = a.reshape((a_shape[0], 1))
        a = self.selu(inp=a)
        # Fully connected layer
        z = np.dot(self.fc_weights, a) + self.fc_biases
        return self.sigmoid(z)

    # Evaluation
    def evaluation(self, test_data, it):
        print("------------------------------------------------")
        print("Evaluate")
        # Get test results
        test_result = [(np.argmax(self.feed_forward(x)), y) for (x, y) in test_data]
        # Count correct predictions
        result = sum(int(x == y) for (x, y) in test_result)
        print("Total: {}, correct: {}, wrong: {}".format(len(test_data), result, len(test_data) - result))
        print("Epoch: {}, accuracy: {}%".format(it, (result / len(test_data)) * 100))
        # Store result
        self.acc.append((result/len(test_result))*100)

    # Involves feed forward and backpropagation for calculating errors
    def backprop(self, x, y, eta):
        # Store activations
        activations = [x]
        a = x
        zs = []
        """""
        Feed forward stage
        """""
        for i in range(self.rep_layers):
            activation = []
            # Conv layer
            a = self.convolutional_layer(inp=a, filter_weight=self.conv_weights[i])
            # a = a+self.conv_biases[i]
            activation.append(a)
            # Relu
            a = self.selu(inp=a)
            activation.append(a)
            # Max pool
            a = self.maxpool_layer(inp=a, pool_filter_size=self.pool_filter_size)
            activation.append(a)
            activations.append(activation)
        # Flatten layer
        mx, my = a.shape
        a = a.flatten('F')
        a_shape = a.shape
        a = a.reshape((a_shape[0], 1))
        zs.append(a)
        a = self.selu(inp=a)
        activations.append(a)
        # Fully Connected layer
        z = np.dot(self.fc_weights, a) + self.fc_biases
        a = self.sigmoid(z)
        # Store activations
        activations.append(a)

        """""
        Backpropagation Stage
        """""
        # Get delta derivative from output layer
        delta = self.cost_derivative(output=activations[-1], y=y, z=z) * self.sigmoid_prime(z=z)
        # Calculate new biases for output layer
        self.fc_biases -= delta * eta
        # Flatten Relu activation
        nabla_w = np.dot(delta, activations[-2].transpose())
        a = np.array(activations[-3][-1].flatten('F'))
        a_shape = a.shape
        a.reshape((a_shape[0], 1))
        div = self.selu_prime(inp=zs[0])
        delta = np.dot(self.fc_weights.transpose(), delta) * div
        self.fc_weights -= (nabla_w * eta)
        # Reshape delta to previous layer (max-pool) dimension
        delta = delta.reshape((mx, my), order='F')
        for i in range(self.rep_layers):
            # MAXPOOL backpass
            # Represent delta in previous dimension
            delta = self.maxpool_prime_with_delta(
                previous_layer=activations[-i-3][-2],
                delta=delta, pool_filter_size=self.pool_filter_size
            )
            # Relu backpass
            sp = self.selu_prime(inp=activations[-i-3][-3])
            delta = delta*sp
            if (-i-3) == (-self.sections + 1):
                # Previous layer input layer
                delta_w = self.convolutional_layer(inp=activations[-i-3-1], filter_weight=delta)
            else:
                # Previous layer maxpool
                delta_w = self.convolutional_layer(inp=activations[-i-3-1][-1], filter_weight=delta)
            delta = self.backprop_convo(inp=delta, filter_weight=self.conv_weights[-i-1])
            self.conv_weights[-i-1] -= (delta_w * eta)

    # Convolutional layer calculation
    @jit
    def convolutional_layer(self, inp, filter_weight):
        """
        :param inp: Data that convolution will be performed on
        :param filter_weight: the fitler weight for the current layer
        :return output: Returns matrix of reduced size after convolution is performed
        """
        # Get size of input
        x, y = inp.shape
        x1 = 0
        # Get size of filter
        # filter_weight = np.rot90(filter_weight, 2)
        x2, y2_shape = filter_weight.shape
        y2 = y2_shape
        output = []  # For storing output
        while y2 < y + 1 or x2 < x + 1:
            # Stores convolution operation for each row
            output_row = []
            # Re-initialize  y1 and y2 values to traverse columns
            y1 = 0
            y2 = y2_shape
            while y2 < y + 1:
                # Obtain inp segment traversed by filter
                fil = inp[x1:x2, y1:y2]
                # Perform dot product like operation with filter weights
                # and the required input slices
                output_row.append(np.sum(np.multiply(fil, filter_weight)))
                # Step size 1 incrementation for columns
                y1 += 1
                y2 += 1
            # Step size 1 incrementation for rows
            x1 += 1
            x2 += 1
            # Results of convolution layer
            output.append(np.array(output_row))
        return np.array(output)

    # Max pool layer calculation
    @jit
    def maxpool_layer(self, inp, pool_filter_size):
        """
        :param inp: Accepts an array of data that maxpool will be performed
        :param filter_weight: the fitler weight for the current layer
        :return output: Returns matrix of reduced size after pooling is performed
        """
        x, y = inp.shape
        x1 = 0
        x2 = pool_filter_size
        y2 = pool_filter_size
        output = []
        while y2 < y + 1 or x2 < x + 1:
            # Stores maximal value operation for each row
            output_row = []
            # re-initialize  y1 and y2 values to traverse columns
            y1 = 0
            y2 = pool_filter_size
            while y2 < y + 1:
                # Obtain inputs segment traversed by filter
                fil = inp[x1:x2, y1:y2]
                # Max pool operation and storing
                max_value = np.amax(fil)
                output_row.append(max_value)
                y1 += 1
                y2 += 1
            # Step size 1
            x1 += 1
            x2 += 1
            output.append(output_row)
        return np.array(output)

    @jit
    def backprop_convo(self, inp, filter_weight):
        """
        :param inp: Data that backward convolution will be performed on
        :param filter_weight: the fitler weight for the current layer
        :return output: Returns matrix of increased size after backward convolution is performed
        """
        # Where inp is delta and filter_weights the
        # weight of the convolution layer
        filter_weight = np.rot90(filter_weight, 2)  # rotate two times for 180 rotation (kernel flip)
        fx, fy = filter_weight.shape
        # Code for rotating weight procedure
        delta = inp
        dx, dy = delta.shape
        new_delta = []
        # Set up selection variables for filter and delta
        # row/column navigation
        # Section "y" are columns variables and "x" are row variables
        # Delta selection
        dx1 = 0
        dx2 = 1
        dy1 = 0
        dy2 = 1
        # Filter selection
        fx1 = fx - 1
        fx2 = fx
        fy1 = fy - 1
        fy2 = fy
        while dx1 < dx or dx2 < dx + 1:
            dtl = []
            while (dy1 < dy) or (dy2 < dy + 1):
                # Matching delta and filter weight selection
                dd = delta[dx1:dx2, dy1:dy2]  # delta selection
                ff = filter_weight[fx1:fx2, fy1:fy2]  # filter selection
                dtl.append(np.sum(np.multiply(dd, ff)))
                # Increment upper selection for filter weight
                if fy1 == 0 and dy1 < dy:
                    dy1 += 1
                if fy1 > 0:
                    fy1 -= 1
                if dy2 < dy + 1:
                    dy2 += 1
                if dy2 == dy + 1 and fy2 > 1:
                    fy2 -= 1
            # These conditions are identical to the conditions in the inner while loop
            if fx1 == 0 and dx1 < dx:
                dx1 += 1
            if fx1 > 0:
                fx1 -= 1
            if dx2 < dx + 1:
                dx2 += 1
            if dx2 == dx + 1 and fx2 > 1:
                fx2 -= 1
            # Reset selection index for columns
            dy1 = 0
            dy2 = 1
            fy1 = fy - 1
            fy2 = fy
            # Append row to new_delta
            new_delta.append(dtl)
        return np.array(new_delta)

    def up_scale_delta_pool(self, delta, pool_filter_size):
        """
        :param delta: the delta matrix being up-scaled to the previous layer
        :param pool_filter_size: filter size of pooling layer
        :return output: Output variable
        """
        output = []
        for i in delta:
            output_row = []
            for j in i:
                for k in range(0, pool_filter_size):
                    # Push delta to column
                    output_row.append(j)
            for r in range(0, pool_filter_size):
                # Append delta to row
                output.append(output_row)
        return np.array(output)

    @jit
    def relu(self, inp):
        # Perform ReLu operation
        return np.where(inp < 0, 0, inp)

    @jit
    def relu_prime(self, inp):
        return np.where(inp < 0, 0, 1)

    def maxpool_prime(self, previous_layer, pool_filter_size):
        """
        :param previous_layer: activations from the layer before pooling layer
        :param pool_filter_size: gets the filter size for max pool
        :return: max_prime return a delta with
        """
        x, y = previous_layer.shape
        max_prime = np.zeros((x, y))
        x1 = 0
        x2 = pool_filter_size
        y2 = pool_filter_size
        row = 0
        while y2 < y + 1 or x2 < x + 1:
            col = 0
            # Re-initialize  y1 and y2 values to traverse columns
            y1 = 0
            y2 = pool_filter_size
            while y2 < y + 1:
                # Obtain inputs segment traversed by filter
                fil = previous_layer[x1:x2, y1:y2]
                max_value = np.amax(fil)
                # Calculate the derivative and perform operation of
                # of the of correct deltas/value in the correct place
                max_p = np.where(fil == max_value, 1, 0)
                # Check if array is equal
                if not np.array_equal(max_p, max_prime):
                    max_prime[x1:x2, y1:y2] += max_p
                y1 += 1
                y2 += 1
                col += 1
            x1 += 1
            x2 += 1
            row += 1
        return max_prime

    @jit
    def maxpool_prime_with_delta(self, previous_layer, delta, pool_filter_size):
        x, y = previous_layer.shape
        # Maxpool derivative multiplied by delta
        max_prime_delta = np.zeros((x, y))
        x1 = 0
        x2 = pool_filter_size
        y2 = pool_filter_size
        row = 0
        while y2 < y + 1 or x2 < x + 1:
            col = 0
            # Re-initialize  y1 and y2 values to traverse columns
            y1 = 0
            y2 = pool_filter_size
            while y2 < y + 1:
                # Obtain inputs segment traversed by filter
                cd = delta[(row, col)]
                fil = previous_layer[x1:x2, y1:y2]
                max_value = np.amax(fil)
                # Calculate the derivative and perform operation of
                # of the of correct deltas/value in the correct place
                max_p = np.where(fil == max_value, 1, 0)
                max_prime_delta[x1:x2, y1:y2] += (max_p * cd)
                y1 += 1
                y2 += 1
                col += 1
            x1 += 1
            x2 += 1
            row += 1
        return max_prime_delta

    def cost_derivative(self, output, y, z):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        self.mse(output, y)
        self.get_loss_numerically_stable(y=y, z=z)
        # return (output_activations - y)
        return output - y

    @jit
    def mse(self, output_activations, y):
        n = len(output_activations)
        # self.error = (np.sum((output_activations - y) ** 2)) / n
        self.mse_error = (np.sum((output_activations - y) ** 2)) / n

    @jit
    def get_loss_numerically_stable(self, y, z):
        self.log_loss_error = -1 * np.sum(y * (z + (-z.max() - np.log(np.sum(np.exp(z - z.max()))))))

    @jit
    def stable_softmax(self, z):
        a = np.exp(z - max(z)) / np.sum(np.exp(z - max(z)))
        return a

    @jit
    def stable_softmaxs(self, z):
        z -= np.max(z)
        sm = (np.exp(z).T / np.sum(np.exp(z), axis=0)).T
        return sm

    @jit
    def softmax_grad(self, z, a, y):
        da = (-y / (a + 1e-20))  # avoid division by zero
        matrix = np.matmul(a, np.ones((1, self.fc2))) * (np.identity(self.fc2) - np.matmul(np.ones((self.fc2, 1)), a.T))
        self.mse(output_activations=a, y=y)
        self.get_loss_numerically_stable(y=y, z=z)
        return np.matmul(matrix, da)

    def sigmoid(self, z):
        """The sigmoid function."""
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_prime(self, z):
        """Derivative of the sigmoid function."""
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def softmax_grads(self, s, a, y):
        self.error = self.mse(output_activations=a, y=y)
        print()
        # Reshape the 1-d softmax to 2-d so that np.dot will do the matrix multiplication
        s = s.reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)

    @jit
    def selu(self, inp):
        a = 1.67326
        lam = 1.0507
        return np.where(inp < 0, lam*(a*(np.exp(inp) - 1)), lam*inp)

    @jit
    def selu_prime(self, inp):
        a = 1.67326
        lam = 1.0507
        return np.where(inp < 0, lam*(a*np.exp(inp)), lam*1)

    def elu(self, inp):
        a = 1.3
        return np.where(inp <= 0, a*(np.exp(inp)-1), inp)

    def elu_prime(self, inp):
        a = 1.3
        return np.where(inp <= 0, self.elu(inp)+a, 1)

    def learning_data(self):
        return self.mse_cost, self.log_loss_cost

    def accuracy(self):
        return self.acc

    def get_fc_param(self):
        return self.fc_weights, self.fc_biases

    def get_conv_params(self):
        return self.conv_weights

    def params(self):
        return self.fc1, self.pool_filter_size, self.conv_filter_size