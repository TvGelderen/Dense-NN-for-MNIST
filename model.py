import numpy as np
import tensorflow as tf

# Randomize the seed
np.random.seed(seed=None)


class Model:
    def add(self, layer):
        self.layer = layer


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
            self.output = tf.relu(self.output)
        elif self.activation_function == "sigmoid":
            self.output = tf.sigmoid(self.output)
        else:
            print("Invalid activation function.")


model = Model()
model.add(DenseLayer(784, 16))
print(model.layer.weights.shape)

print("End of file")
