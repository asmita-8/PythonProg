##Full Neural Network implementation from scratch:

##Implement a 2-layer (input layer and output layer) neural network from scratch for the following dataset.
##This includes implementing forward and backward passes fromscratch. Print the training loss and plot it over 1000 iterations.

import numpy as np
import matplotlib.pyplot as plt
def update_weights(w, dl_by_dw, learning_rate):
    return w - learning_rate * dl_by_dw
def backprop(x, z, a, y):
    L = 0.5 * (y - a) ** 2  ###Loss function
    dl_by_dl = 1    ###Derivative of loss wrt itself
    dl_by_da = -(y - a)  ###Derivative of loss wrt a
    dl_by_dz = dl_by_da * a * (1 - a)  ###Derivative of loss wrt z
    dl_by_dw = dl_by_dz * x.T  ###Derivative of loss wrt weights
    return L, dl_by_dw
def forward(x, w, b):
    z = np.dot(w, x) + b  ###forward pass
    a = 1 / (1 + np.exp(-z))  ###Sigmoid activation function
    return z, a
def main():
    x = np.array([0, 1, 1]).reshape(-1, 1)  ###Shape is now 3x1
    y = np.array([0])  ###Expected output
    w = np.random.uniform(size=(1, 3))  ###Shape is 1x3
    b = np.random.uniform(size=(1, 1))  ###Bias term
    learning_rate = 0.01
    num_iterations = 1000
    losses = []
    for i in range(num_iterations):
        z, a = forward(x, w, b)
        loss, dl_by_dw = backprop(x, z, a, y)
        w = update_weights(w, dl_by_dw, learning_rate)
        losses.append(loss[0][0])  ###Storing loss for plotting
        if i % 100 == 0:  ###Printing every 100 iterations
            print(f"Iteration {i}: Loss = {loss[0][0]}")
    ###Plotting
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Iterations')
    plt.show()
if __name__ == "__main__":
    main()




###Implement a 3-layer (input layer, hidden layer and output layer) neural network from scratch for the XOR operation.
###This includes implementing forward and backward passes from scratch.
import numpy as np
from matplotlib import pyplot as plt
def update_weights(weights, dl_by_dw1, dl_by_dw2, learning_rate):
    weights[-1] -= learning_rate * dl_by_dw2
    weights[-2] -= learning_rate * dl_by_dw1
    return weights
def backprop(z, a, x, y, weights, biases):
    ### Loss function
    loss = 0.5 * ((y - a[-1]) ** 2)
    dl_by_da = -(y - a[-1])

    ### Backpropagate through the output layer
    da_dz2 = a[-1] * (1 - a[-1])
    dl_by_dz2 = dl_by_da * da_dz2
    dz2_dw2 = a[-2]  # This is the output from the hidden layer

    dl_by_dw2 = np.dot(dl_by_dz2, dz2_dw2.T)

    ### Backpropagate through the hidden layer
    dz2_da1 = weights[-1]
    dl_by_da1 = np.dot(dz2_da1.T, dl_by_dz2)

    da1_dz1 = a[-2] * (1 - a[-2])
    dl_by_dz1 = dl_by_da1 * da1_dz1
    dz1_dw1 = x.T

    dl_by_dw1 = np.dot(dl_by_dz1, dz1_dw1)

    return loss, dl_by_dw1, dl_by_dw2
def forward(weights, biases, x):
    a = [x]
    z = []
    for w, b in zip(weights, biases):
        z_curr = np.dot(w, a[-1]) + b
        z.append(z_curr)
        a_curr = 1 / (1 + np.exp(-z_curr))  ### Sigmoid activation
        a.append(a_curr)
    return z, a
def main():
    # Define weights, biases, and inputs
    weights = [np.random.uniform(size=(2, 2)), np.random.uniform(size=(1, 2))]  # w1 and w2
    biases = [np.random.uniform(size=(2, 1)), np.random.uniform(size=(1, 1))]  # b1 and b2
    x = np.array([0, 1]).reshape(-1, 1)  # Shape is now 2x1
    y = np.array([1])  # Expected output
    learning_rate = 0.01
    num_iterations = 1000
    losses = []
    for i in range(num_iterations):
        z, a = forward(weights, biases, x)
        loss, dl_by_dw1, dl_by_dw2 = backprop(z, a, x, y, weights, biases)
        weights = update_weights(weights, dl_by_dw1, dl_by_dw2,learning_rate)
        losses.append(loss[0][0])  ###Storing loss for plotting
        if i % 100 == 0:  ###Printing every 100 iterations
            print(f"Iteration {i}: Loss = {loss[0][0]}")
            ###Plotting
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Iterations')
    plt.show()
if __name__ == "__main__":
    main()

