from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

from vapnik.perceptron import PerceptronGeneral
from vapnik.data_generator import get_plane_points, sample_points

va = np.array([1, 0, 0, 0])
vb = np.array([0, 1, 0, 0])
vc = np.array([0, 0, 1, 0])
vd = np.array([0, 0, 0, 1])

KERNELS = {
    1: [
        [va], [vb], [vc], [vd], [va+vb], [va+vc], [va+vd], [vb+vc], [vb+vd], [vc+vd]
    ],
    2: [
        [va, vb], [va, vc], [va, vd], [vb, vc], [vb, vd], [vc, vd],
        [va+vb, vc], [va+vb, vd], [va+vb, vc+vd],
        [va+vc, vb], [va+vc, vd], [va+vc, vb+vd],
        [va+vd, vb], [va+vd, vc], [va+vd, vb+vc],
        [va+vb+vc, vd], [va+vb+vd, vc], [va+vc+vd, vb], [vb+vc+vd, va]
    ],
    3: [
        [va, vb, vc], [va, vb, vd], [vb, vc, vd],
        [va+vb, vc, vd], [va+vc, vb, vd], [va+vd, vb, vc],
        [vb+vc, va, vd], [vb+vd, va, vc],
        [vc+vd, va, vb]
    ],
    4: [
        [va, vb, vc, vd]
    ]
}


def pocket_perceptron(X, Y, kernel, epochs):
    p = PerceptronGeneral(3, kernel)

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
                # print('best result:', best_result / X.shape[0])

    p.weights = best_weights
    return p


def plot_plane(a, b, c, d, ax, color):
    x, y, z = get_plane_points(a, b, c, d)
    ax.plot_surface(x, y, z, alpha=0.2, color=color)


def main():
    a = -1
    b = -2
    c = -4
    d = 3
    n = 100

    epochs = 200
    noise = 0.2

    x_pos, y_pos, z_pos = sample_points(a, b, c, d, n=n, side=1, noise=noise)
    X_pos = np.stack((x_pos, y_pos, z_pos), axis=-1)
    Y_pos = np.ones(n, dtype=np.int)

    x_neg, y_neg, z_neg = sample_points(a, b, c, d, n=n, side=-1, noise=noise)
    X_neg = np.stack((x_neg, y_neg, z_neg), axis=-1)
    Y_neg = -1 * np.ones(n, dtype=np.int)

    X = np.concatenate((X_pos, X_neg), axis=0)
    Y = np.concatenate((Y_pos, Y_neg), axis=0)

    ones = np.reshape(np.array(np.ones(2*n)), (2*n, 1))

    X = np.append(X, ones, axis=1)

    for vc in [1, 2, 3, 4]:
        print('VC: %d' % (vc))
        for kernel in KERNELS[vc]:
            p = pocket_perceptron(X, Y, kernel, epochs)

            ok = 0
            total = 0
            for x, y in zip(X, Y):
                y_hat = p.predict(x)
                total += 1
                if y == y_hat:
                    ok += 1

            print('%.4f' % (ok / total), p.weights)

            a, b, c, d = p.weights


if __name__ == '__main__':
    main()