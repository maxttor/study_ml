import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

def get_error(deltas, sums, weights):
    """
    compute error on the previous layer of network
    deltas - ndarray of shape (n, n_{l+1})
    sums - ndarray of shape (n, n_l)
    weights - ndarray of shape (n_{l+1}, n_l)
    """
    return (deltas.dot(weights) * sigmoid_prime(sums)).mean(axis=0)

deltas = np.array([[0.3,  0.2], [0.3,  0.2]])
sums = np.array([[0, 1, 1], [0, 2, 2]])
weights = np.array([[ 0.7,  0.2,  0.7], [ 0.8,  0.3,  0.6]])

print(get_error(deltas, sums, weights))


