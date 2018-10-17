import numpy as np

from vapnik.perceptron import PerceptronVC1
from vapnik.data_generator import get_plane_points, sample_points


def pocket_perceptron(X, Y, epochs):
    p = PerceptronVC1(3)
    best_result = -1
    best_weights = None
    for _ in range(epochs):
        for x, y in zip(X, Y):
            p.fit(x, y)

            predicted = p.predict_batch(X)
            accuracy = np.sum(predicted == Y)

            if accuracy > best_result:
                best_result = accuracy
                best_weights = p.weights.copy()
                print('best result:', best_result / X.shape[0])

    p.weights = best_weights
    return p


def main():
    a = 1
    b = -1
    c = 2
    d = -3
    n = 1000

    epochs = 10
    noise = 0.0

    x, y, z = sample_points(a, b, c, d, n=n, side=1, noise=noise)
    X_pos = np.stack((x, y, z), axis=-1)
    y_pos = np.ones(n, dtype=np.int)

    x, y, z = sample_points(a, b, c, d, n=n, side=-1, noise=noise)
    X_neg = np.stack((x, y, z), axis=-1)
    y_neg = -1 * np.ones(n, dtype=np.int)

    X = np.concatenate((X_pos, X_neg), axis=0)
    Y = np.concatenate((y_pos, y_neg), axis=0)

    ones = np.reshape(np.array(np.ones(2*n)), (2*n, 1))

    X = np.append(ones, X, axis=1)

    p = pocket_perceptron(X, Y, epochs)

    ok = 0
    total = 0
    for x, y in zip(X, Y):
        y_hat = p.predict(x)
        total += 1
        if y == y_hat:
            ok += 1

    print(ok / total)

    print(p.weights)


if __name__ == '__main__':
    main()