from tensorflow.python.keras.datasets import mnist
import numpy as np
from math import exp
import sys

class Network:
    def __init__(self):    
        (self.trainX, self.trainY), (self.testX, self.testY) = mnist.load_data()
        # Input layer consists of 784 neurons, one for each item in the matrices
        self.n_in = 784
        #the two hidden layers both contain 16 neurons
        self.n_h1 = 16
        self.n_h2 = 16
        # The output layer contains 10 neurons, one for each possible number
        self.n_out = 10
        # Initialize weights
        self.weights_l0_l1 = np.random.rand(self.n_in,self.n_h1)
        self.weights_l1_l2 = np.random.rand(self.n_h1,self.n_h2)
        self.weights_l2_l3 = np.random.rand(self.n_h2,self.n_out)
        