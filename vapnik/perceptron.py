import numpy as np
from math import sqrt


class Perceptron(object):
    def __init__(self, n, learning_rate=1e-2):
        self.weights = np.random.normal(size=n+1)
        self.learning_rate = learning_rate

    def predict(self, X):
        activation = self.weights.dot(X)
        if activation > 0:
            return 1
        else:
            return -1

    def predict_batch(self, X):
        prediction = X.dot(self.weights) >= 0
        idx_positive = (prediction == True)
        idx_negative = (prediction == False)

        result = np.zeros(prediction.shape, dtype=np.int)
        result[idx_positive] = 1
        result[idx_negative] = -1
        return result

    def fit(self, X, y):
        y_hat = self.predict(X)
        delta = self.learning_rate * (y - y_hat) * X
        self.weights += delta


class PerceptronVC1(Perceptron):
    def __init__(self, n, learning_rate=1e-2):
        Perceptron.__init__(self, n, learning_rate)
        self.weights = np.zeros(n+1)
        v1 = np.array([1, 1, 1, 0], dtype=np.float)
        v2 = np.array([0, 0, 0, 1], dtype=np.float)
        self.v1 = v1 / np.linalg.norm(v1)
        self.v2 = v2 / np.linalg.norm(v2)

    def fit(self, X, y):
        y_hat = self.predict(X)

        if y - y_hat == 0:
            return

        delta = self.learning_rate * (y - y_hat) * X
        new_weights = self.weights + delta

        self.weights = np.dot(self.v1, new_weights)*self.v1 + np.dot(self.v2, new_weights)*self.v2
        # self.weights = new_weights