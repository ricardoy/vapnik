from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
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


def plot_plane(a, b, c, d, ax, color):
    x, y, z = get_plane_points(a, b, c, d)
    ax.plot_surface(x, y, z, alpha=0.2, color=color)


def main():
    a = -1
    b = -2
    c = 3
    d = 10
    n = 1000

    epochs = 10
    noise = 0.0

    x_pos, y_pos, z_pos = sample_points(a, b, c, d, n=n, side=1, noise=noise)
    X_pos = np.stack((x_pos, y_pos, z_pos), axis=-1)
    Y_pos = np.ones(n, dtype=np.int)

    x_neg, y_neg, z_neg = sample_points(a, b, c, d, n=n, side=-1, noise=noise)
    X_neg = np.stack((x_neg, y_neg, z_neg), axis=-1)
    Y_neg = -1 * np.ones(n, dtype=np.int)

    X = np.concatenate((X_pos, X_neg), axis=0)
    Y = np.concatenate((Y_pos, Y_neg), axis=0)

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

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot_plane(a, b, c, d, ax, 'r')

    d, a, b, c = p.weights
    plot_plane(a, b, c, d, ax, 'g')

    ax.scatter(x_pos, y_pos, z_pos, c='r', marker='o')
    ax.scatter(x_neg, y_neg, z_neg, c='b', marker='^')
    # print(z_pos)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


if __name__ == '__main__':
    main()