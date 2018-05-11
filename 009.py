import numpy as np


class NeuronNetwork:
    def __init__(self, weights):
        self.weights = weights
        self.activation_values = []

    def activation(self, x, lin=False):
        """сигмоидальная функция, работает и с числами, и с векторами (поэлементно)"""
        if lin:
            return np.array(
                [[(lambda x, i: max(x, 0) if i == 0 else self.activation(x))(x[i][0], i)] for i in range(len(x))]
            )

        return 1 / (1 + np.exp(-x))

    def activation_derivative(self, a, lin=False):
        """производная сигмоидальной функции, работает и с числами, и с векторами (поэлементно)"""
        if lin:
            return np.array(
                [[(lambda x, i: (lambda x: 1 if x > 0 else 0)(x) if i == 0 else self.activation_derivative(x))(a[i][0], i)]
                 for i in range(len(a))]
            )

        return a * (1 - a)

    def predict(self, a):
        for i in range(len(self.weights)):
            z = self.weights[i].T.dot(a)
            a = self.activation(z, i == 0)
            self.activation_values.append(a)
        return a

    def J_quadratic(self, X, y):
        assert y.shape[1] == 1, 'Incorrect y shape'

        return 0.5 * np.mean((self.predict(X) - y) ** 2)

    def gradient(self, y_hat, y):
        assert y_hat.shape == y.shape and y_hat.shape[1] == 1, 'Incorrect shapes'

        return y_hat - y

    def train(self, X, y):
        a_L = self.predict(X)
        nabla_j = self.gradient(a_L, y)
        delta_l = nabla_j * self.activation_derivative(a_L)
        print(a_L)

        for i in range(len(self.activation_values) - 2, -1, -1):
            delta_l = self.weights[i + 1].dot(delta_l)
            delta_l = delta_l * self.activation_derivative(self.activation_values[i], i == 0)
            print(delta_l)


weights = [
    np.array([[0.7, 0.8], [0.2, 0.3], [0.7, 0.6]]),
    np.array([[0.2], [0.4]])
]

network = NeuronNetwork(weights)
res = network.train(np.array([[0], [1], [1]]), np.array([[1]]))

print('dw1', 1*-0.01829328)
print('dw2', 1*-0.00751855)
