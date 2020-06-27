from tensorflow.python.keras.datasets import mnist
import numpy as np
from math import exp
import sys

# Sigmoid activation function
def sigmoid(x):
    return (1 / (1 + exp(-x)))

# Derivative of the sigmoid
def derivSigmoid(arr):
    return (arr * (1.0 - arr))

if __name__ == '__main__':
    (trainX, trainY), (testX, testY) = mnist.load_data()
    # X contains the images, Y contains the corresponding number
    
    # Input layer consists of 784 neurons, one for each item in the matrices
    n_in = 784
    # The two hidden layers both contain 16 neurons
    n_h1 = 16
    n_h2 = 16
    # The output layer contains 10 neurons, one for each possible number
    n_out = 10
    
    # Initialize weights from a uniform distribution between 0.0 and 0.5
    weights = [np.random.uniform(0.0, 0.5, (n_in,n_h1)), np.random.uniform(0.0, 0.5, (n_h1,n_h2)), np.random.uniform(0.0, 0.5, (n_h2,n_out))]
    # Initialize the matrix for the activation values
    activations = [np.zeros(n_in), np.zeros(n_h1), np.zeros(n_h2), np.zeros(n_out)]
    # Initialize biases from a uniform distribution between 0.0 an 1.0
    biases = [np.random.rand(n_h1), np.random.rand(n_h2), np.random.rand(n_out)]
    # Initalize the error matrix
    errors = [np.zeros(n_h1), np.zeros(n_h2), np.zeros(n_out)]
    
    # Change the type to float so as to make sure we get decimal values when normalizing
    trainX = trainX.astype('float32')
    testX = testX.astype('float32')
    # Normalizing by dividing by the max RGB value
    trainX /= 255
    testX /= 255
    # Transpose the arrays so they can be input to the input nodes
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
            sys.stdout.write("\rTraining {}/60000\n".format(trainIndex+1))
            # FORWARDPROPAGATION
            # Add the input
            for i in range(n_in):
                activations[0][i] = trainX[trainIndex][i]
            # Calculate the activations through the layers
            for l in range(len(activations)-1):
                # To calculate the activations of the neurons in layer l+1 we take the following dot product and add the respective bias
                activations[l+1] = activations[l].dot(weights[l]) + biases[l]
                # Apply the sigmoid to get the final activation
                for i in range(len(activations[l+1])):
                    #print("z: {} \t\t a: {}".format(activations[l+1][i], sigmoid(activations[l+1][i])))
                    activations[l+1][i] = sigmoid(activations[l+1][i])
                    
            # Create an expected output vector
            y = np.zeros(n_out)
            for i in range(n_out):
                if i == trainY[trainIndex]:
                    y[i] = 1.0
                else:
                    y[i] = 0.0
            
            # BACKPROPAGATION
            # Calculate the cost
            a = y-activations[3]
            cost = a.dot(a)/2
            
            # Calculate the error in the output layer
            errors[2] = np.multiply((activations[3] - y), derivSigmoid(activations[3]))
            
            # Propagate the error backwards
            errors[1] = np.multiply((weights[2].dot(errors[2])), derivSigmoid(activations[2]))
            errors[0] = np.multiply((weights[1].dot(errors[1])), derivSigmoid(activations[1]))
                    
            # Update the biases
            for l in range(len(biases)):
                biases[l] -= errors[l]
            
                        
            
            sys.stdout.flush()
    
    # Save the weights to files
    np.save('weights\weightslayer1.npy', weights[0])
    np.save('weights\weightslayer2.npy', weights[1])
    np.save('weights\weightslayer3.npy', weights[2])