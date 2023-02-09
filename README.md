# Neural-Network-from-scratch
Neural Network on C++ using only standard libraries.

File [network.h](nn/network.h) implements basic instruments for neural networks, like arrays, layers, network, backpropagation with gradient descent etc.

File [train.cpp](nn/train.cpp) creates and trains neural network on [mnist dataset](data) to recognise handwritten digits. Also function to read csv files.

File [predict.cpp](nn/predict.cpp) implements predictions using trained network on images through comand line interface.
