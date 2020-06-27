# Simple-NN-for-the-MNIST-dataset

This repository contains a simple neural net for recognizing handwritten numbers using the MNIST dataset

The NN has 784 input nodes, one for each of the pixels in the images, two hidden layers of 16 neurons and an output layer of 10 neurons, one for each of the possible numbers.

After training, the weights are stored in the weights folder as numpy arrays, where they can later be retrieved to use them again, which allows for only training the network once.
