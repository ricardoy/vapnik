from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from vapnik.perceptron import Perceptron
from vapnik.data_generator import get_plane_points, sample_points
from sklearn.model_selection import train_test_split

DIMENSION = 4

va = np.array([True, 0, 0, 0], dtype=np.bool)
vb = np.array([0, True, 0, 0], dtype=np.bool)
vc = np.array([0, 0, True, 0], dtype=np.bool)
vd = np.array([0, 0, 0, True], dtype=np.bool)

KERNELS = {
    1: [
        ('ax + 0y + 0z + 0', [va]),
        ('0x + ay + 0z + 0', [vb]),
        ('0x + 0y + az + 0', [vc]),
        ('0x + 0y + 0z + a', [vd]),
        ('ax + ay + 0z + 0', [va+vb]),
        ('ax + 0y + az + 0', [va+vc]),
        ('ax + 0y + 0z + a', [va+vd]),
        ('0x + ay + az + 0', [vb+vc]),
        ('0x + ay + 0z + a', [vb+vd]),
        ('0x + 0y + az + a', [vc+vd]),
        ('ax + ay + az + 0', [va+vb+vc]),
        ('ax + ay + 0z + a', [va+vb+vd]),
        ('ax + 0y + az + a', [va+vc+vd]),
        ('0x + ay + az + a', [vb+vc+vd]),
        ('ax + ay + az + a', [va+vb+vc+vd])
    ],

    2: [
        ('ax + by + 0z + 0', [va, vb]),
        ('ax + 0y + bz + 0', [va, vc]),
        ('ax + 0y + 0z + b', [va, vd]),
        ('0x + ay + bz + 0', [vb, vc]),
        ('0x + ay + 0z + b', [vb, vd]),
        ('0x + 0y + az + b', [vc, vd]),

        ('ax + by + bz + 0', [vb+vc, va]),
        ('ax + by + 0z + b', [vb+vd, va]),
        ('ax + by + 0z + b', [vc+vd, va]),
        ('ax + by + az + 0', [va+vc, vb]),
        ('ax + by + 0z + a', [va+vd, vb]),
        ('0x + ay + bz + b', [vc+vd, vb]),
        ('ax + ay + bz + 0', [va+vb, vc]),
        ('ax + 0y + bz + a', [va+vd, vc]),
        ('0x + ay + bz + a', [vb+vd, vc]),
        ('ax + ay + 0z + b', [va+vb, vd]),
        ('ax + 0y + az + b', [va+vc, vd]),
        ('0x + ay + az + b', [vb+vc, vd]),

        ('ax + ay + bz + b', [va+vb, vc+vd]),
        ('ax + by + az + b', [va+vc, vb+vd]),
        ('ax + by + bz + a', [va+vd, vb+vc]),

        ('ax + ay + az + b', [va+vb+vc, vd]),
        ('ax + ay + bz + a', [va+vb+vd, vc]),
        ('ax + by + az + a', [va+vc+vd, vb]),
        ('bx + ay + az + a', [vb+vc+vd, va])
    ],

    3: [
        ('ax + by + cz + 0', [va, vb, vc]),
        ('ax + by + 0z + c', [va, vb, vd]),
        ('0x + ay + bz + c', [vb, vc, vd]),
        ('ax + ay + bz + c', [va+vb, vc, vd]),
        ('ax + by + az + c', [va+vc, vb, vd]),
        ('ax + by + cz + a', [va+vd, vb, vc]),
        ('ax + by + bz + c', [vb+vc, va, vd]),
        ('ax + by + cz + a', [vb+vd, va, vc]),
        ('ax + by + cz + c', [vc+vd, va, vb])
    ],

    4: [
        ('ax + by + cz + d', [va, vb, vc, vd])
    ]
}


def pocket_perceptron(X, Y, epochs):
    p = Perceptron(X.shape[1])

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


def convert_X(X, kernels):
    cols = []
    for kernel in kernels:
        col = np.sum(X * kernel, axis=-1)
        cols.append(col)

    result = np.stack(cols, axis=-1)
    return result


def run_all_experiments(verbose=True):
    a = 502
    b = 6332
    c = -100
    d = -3222
    n = 4000
    train_size = 5

    if verbose:
        print('Objective: ', (a, b, c, d))

    epochs = 200
    noise = 0.4

    X, Y = generate_dataset(a, b, c, d, n//2, noise)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = (n - train_size), random_state = 42, stratify=Y)

    if verbose:
        print('n', X.shape, 'train', X_train.shape, 'test', X_test.shape)

    best_scores = []
    all_eins = []
    all_eouts = []
    for vc in [1, 2, 3, 4]:

        if verbose:
            print('VC: %d' % (vc))

        min_eout = float('inf')
        associated_ein = None
        for representation, kernel in KERNELS[vc]:
            X_train_converted = convert_X(X_train, kernel)
            p = pocket_perceptron(X_train_converted, Y_train, epochs)

            predicted = p.predict_batch(X_train_converted)
            ein = np.sum(predicted != Y_train) / (Y_train.shape[0])
            # ein = np.sum(predicted != Y_train)

            all_eins.append((vc, ein))

            X_test_converted = convert_X(X_test, kernel)
            predicted = p.predict_batch(X_test_converted)
            eout = np.sum(predicted != Y_test) / (Y_test.shape[0])
            # eout = np.sum(predicted != Y_test)

            all_eouts.append((vc, eout))

            if eout < min_eout:
                min_eout = eout
                associated_ein = ein

            if verbose:
                print('%s | ein: %.4f | eout: %.4f' % (representation, ein, eout), '|', p.weights)

            # a, b, c, d = p.weights
        best_scores.append((associated_ein, min_eout))

    eins = np.array([x[0] for x in best_scores])
    eouts = np.array([x[1] for x in best_scores])

    if verbose:
        plot_scatter(all_eins, all_eouts, a, b, c, d, noise)
        # plot_graphic(eins, eouts)

    return eins, eouts


def generate_dataset(a, b, c, d, n, noise):
    x_pos, y_pos, z_pos = sample_points(a, b, c, d, n=n, side=1, noise=noise)
    X_pos = np.stack((x_pos, y_pos, z_pos), axis=-1)
    Y_pos = np.ones(n, dtype=np.int)
    x_neg, y_neg, z_neg = sample_points(a, b, c, d, n=n, side=-1, noise=noise)
    X_neg = np.stack((x_neg, y_neg, z_neg), axis=-1)
    Y_neg = -1 * np.ones(n, dtype=np.int)
    X = np.concatenate((X_pos, X_neg), axis=0)
    Y = np.concatenate((Y_pos, Y_neg), axis=0)
    ones = np.reshape(np.array(np.ones(2 * n)), (2 * n, 1))
    X = np.append(X, ones, axis=1)
    return X, Y


def plot_scatter(all_eins, all_eouts, a, b, c, d, noise):
    x = [t[0] for t in all_eins]
    y = [t[1] for t in all_eins]

    fig = plt.figure()
    fig.suptitle('%.1fx %+.1fy %+.1fz %+.1f, noise=%.2f' % (a, b, c, d, noise))
    ax = fig.add_subplot(111)
    ax.scatter(x, y, label='Ein', marker='o', alpha=0.2)

    y = [t[1] for t in all_eouts]
    ax.scatter(x, y, label='Eout', marker='^', alpha=0.2)

    ax.legend()
    ax.set_ylim([0, 1.])
    ax.set_ylabel('Error')
    ax.set_xlabel('VC-dimension')
    ax.tick_params(axis='x', which='major', labelsize=8)
    # plt.xticks()

    # for x, y in zip(interval, eouts):
    #     ax.annotate('%.5f'%(y), xy=(x, y))
    #
    # for x, y in zip(interval, eins):
    #     ax.annotate('%.5f'%(y), xy=(x, y))

    plt.show()


def plot_graphic(eins, eouts):
    interval = [1, 2, 3, 4]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(interval, eins, label='Ein', marker='o')
    ax.plot(interval, eouts, label='Eout', marker='^')
    # ay = fig.add_subplot(111)
    ax.legend()
    if np.max(eouts) <= 1:
        ax.set_ylim([0, 1.])
    ax.set_ylabel('Error')
    ax.set_xlabel('VC-dimension')
    ax.tick_params(axis='x', which='major', labelsize=8)
    # plt.xticks()

    for x, y in zip(interval, eouts):
        ax.annotate('%.5f'%(y), xy=(x, y))

    for x, y in zip(interval, eins):
        ax.annotate('%.5f'%(y), xy=(x, y))

    plt.show()


def main():
    avg_eins = np.zeros(DIMENSION)
    avg_eouts = np.zeros(DIMENSION)
    n = 100
    for _ in tqdm(range(n)):
        eins, eouts = run_all_experiments(False)
        avg_eins = avg_eins + eins/n
        avg_eouts = avg_eouts + eouts/n

    plot_graphic(avg_eins, avg_eouts)


if __name__ == '__main__':
    # main()
    run_all_experiments()
