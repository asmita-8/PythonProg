###calculate the final activation function(XOR FUNCTION)for a neural network having
###1 hidden layer(2 nodes)and 1 output layer(1 node) for which weights and biases are given.
###FNN
import numpy as np
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def neural_network(weights, biases, X):
    a = np.reshape(X, (len(X), 1))
    for w, b in zip(weights, biases):
        z = np.dot(w, a) + b
        a = sigmoid(z)
    return a
def main():
    ###Defining weights, biases and inputs
    weights = [np.array([[20, 20], [-20, -20]]), np.array([[20, 20]])]   ###w1 and w2
    biases = [np.array([[-10], [30]]), np.array([[-30]])]   ###b1 and b2
    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]   ###Inputs of XOR function
    for X in inputs:
        output = neural_network(weights, biases, X)
        print(output)
if __name__ == "__main__":
    main()

###results for comparison:
print("For 0, 0 approximate answer is 0---->4.54391049e-05")
print("For 0, 1 approximate answer is 1---->0.99995452")
print("For 1, 0 approximate answer is 1---->0.99995452")
print("For 1, 1 approximate answer is 0---->4.54391049e-05")


