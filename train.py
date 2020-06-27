from tensorflow.python.keras.datasets import mnist
import numpy as np
from math import exp
import sys

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + exp(-x))

if __name__ == '__main__':
    (trainX, trainY), (testX, testY) = mnist.load_data()
    # X contains the images, Y contains the corresponding number
    
    # Input layer consists of 784 neurons, one for each item in the matrices
    n_in = 784
    #the two hidden layers both contain 16 neurons
    n_h1 = 16
    n_h2 = 16
    # The output layer contains 10 neurons, one for each possible number
    n_out = 10
    
    # Initialize weights
    weights = [np.random.rand(n_h1,n_in), np.random.rand(n_h2,n_h1), np.random.rand(n_out,n_h2)]
    # Initialize activation values
    activations = [np.zeros(n_in), np.zeros(n_h1), np.zeros(n_h2), np.zeros(n_out)]
    # Initialize biases
    biases = [np.random.rand(n_h1), np.random.rand(n_h2), np.random.rand(n_out)]
    
    # Change the type to float so as to make sure we get decimal values when normalizing
    trainX = trainX.astype('float32')
    testX = testX.astype('float32')
    # Normalizing by dividing by the max RGB value
    trainX /= 255
    testX /= 255
    # Transpose the arrays so they are easy to put into the input nodes
    trainX = trainX.transpose(0,2,1).reshape(-1,784)
    testX = testX.transpose(0,2,1).reshape(-1,784)
    
    # Determine number of epochs
    epochs = 1
    #epochs = input("Please enter number of epochs: ")
        
    # Iterate through the epochs
    for epoch in range(epochs):
        print("Epoch {}/{}:".format(epoch+1,epochs))
        
        # Iterate through all training images
        for trainIndex in range(1):
            sys.stdout.write("\rTraining {}/60000".format(trainIndex+1))
            # Add the input
            for i in range(n_in):
                activations[0][i] = trainX[trainIndex][i]
            # Calculate the activations through the layers
            for i in range(3):
                for j in range(len(activations[i+1])):
                    # Reset the activation sum
                    activation = 0
                    # Sum the activations of the neurons from the previous layers times the weights of the connections
                    for k in range(len(activations[i])):
                        # It is more common to refer to the weights between layer i and i+1 as weights i+1, but due to the way the matrix is costructed these will be referred to as weights i
                        activation += weights[i][j][k] * activations[i][k]
                    # Add the bias
                    activation += biases[i][j]
                    # Apply the sigmoid function to get the activation
                    activations[i+1][j] = sigmoid(activation)
                    
                
            
            sys.stdout.flush()
    
    # Save the weights to files
    np.save('weights\weightslayer1.npy', weights[0])
    np.save('weights\weightslayer2.npy', weights[1])
    np.save('weights\weightslayer3.npy', weights[2])