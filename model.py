import numpy as np
import tensorflow as tf

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
        print(input)


class DenseLayer:
    def __init__(self, n_input, n_output, activation_function="relu"):
        self.weights = np.random.uniform(-1, 1, (n_input, n_output))
        self.activations = np.zeros(n_output)
        self.biases = np.random.uniform(-1, 1, n_output)
        self.activation_function = activation_function
        self.output = np.zeros(n_output)

    def forward(self, input):
        self.output = self.biases
        if self.activation_function == "relu":
            self.output = tf.nn.relu(self.output)
        elif self.activation_function == "sigmoid":
            self.output = tf.nn.sigmoid(self.output)
        else:
            print("Invalid activation function.")


(train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
# Change the type to float so as to make sure we get decimal values when normalizing
train_x, test_x = train_x.astype('float32'), test_x.astype('float32')
# Normalizing by dividing by the max RGB value
train_x, test_x = train_x/255, test_x/255
# Transpose the arrays so they can be input to the input nodes
train_x, test_x = train_x.transpose(0, 1, 2).reshape(-1, 784), test_x.transpose(0, 1, 2).reshape(-1, 784)

model = Model()
model.add(DenseLayer(784, 16))
model.add(DenseLayer(16, 16))

model.propagate_forward(train_x[0])

print("End of file")
