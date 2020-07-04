import tensorflow as tf
import numpy as np

# Used to stop a warning from tf about AVX2 usage
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Randomize the seed
np.random.seed(seed=None)


class Model:
    def __init__(self):
        self.layers = np.empty(0)

    def add(self, layer):
        self.layers = np.append(self.layers, layer)

    def propagate_forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        self.output = input

    def propagate_back(self, error):
        for layer in reversed(self.layers):
            error = layer.back(error)

    def fit(self, x, y, batch_size=15, epochs=1):
        self.x, self.y, self.batch_size, self.epochs = x, y, batch_size, epochs
        # for epoch in range(self.epochs):
        #     for i in range(len(self.x)):
        #         self.propagate_forward(self.x[:batch_size])
        #         i += batch_size
        self.propagate_forward(self.x[:self.batch_size])

        self.y_hat = np.zeros((self.batch_size, self.output.shape[1]))
        for i in range(self.batch_size):
            for j in range(self.output.shape[1]):
                if j == self.y[i]:
                    self.y_hat[i][j] = 1.0
                else:
                    self.y_hat[i][j] = 0.0

        errors = 2 * (self.y_hat - self.output)
        error = np.zeros(self.output.shape[1])
        for i in range(errors.shape[1]):
            for j in range(batch_size):
                error[i] += errors[j][i]
        print(error)

    def evaluate(self):
        pass


class DenseLayer:
    def __init__(self, n_input, n_output, activation_function="relu"):
        self.weights = np.random.uniform(-1, 1, (n_input, n_output))
        self.activations = np.zeros(n_output)
        self.biases = np.random.uniform(-1, 1, n_output)
        self.activation_function = activation_function
        self.output = np.zeros(n_output)

    def forward(self, input):
        self.output = np.dot(input, self.weights) + self.biases

        if self.activation_function == "relu":
            self.output = tf.nn.relu(self.output)
        elif self.activation_function == "sigmoid":
            self.output = tf.nn.sigmoid(self.output)
        else:
            print("Invalid activation function.")

        return self.output

    def back(self):
        pass
