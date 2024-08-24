###Full Neural Network implementation from scratch:

###Implement a 2-layer (input layer and output layer) neural network from scratch for the following dataset.
###This includes implementing forward and backward passes fromscratch. Print the training loss and plot it over 1000 iterations.

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

