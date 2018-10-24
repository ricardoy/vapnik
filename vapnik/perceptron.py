import numpy as np


class Perceptron(object):
    def __init__(self, n, learning_rate=1e-1):
        self.weights = np.random.normal(size=n)
        self.learning_rate = learning_rate

    def predict(self, X):
        activation = self.weights.dot(X)
        if activation >= 0:
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


class PerceptronGeneral(Perceptron):
    def __init__(self, n, kernel, learning_rate=1e-2):
        Perceptron.__init__(self, n, learning_rate)
        self.weights = np.zeros(n+1)
        self.kernel = []
        for k in kernel:
            v = np.array(k, dtype=np.float)
            self.kernel.append(v / np.linalg.norm(v))

    def fit(self, X, y):
        y_hat = self.predict(X)

        if y - y_hat == 0:
            return

        delta = self.learning_rate * (y - y_hat) * X
        new_weights = self.weights + delta

        aux = np.dot(self.kernel[0], new_weights)*self.kernel[0]
        for i in range(1, len(self.kernel)):
            aux = aux + (np.dot(self.kernel[i], new_weights)*self.kernel[i])

        self.weights = aux
