import numpy as np

from tqdm import tqdm
from vapnik.perceptron import Perceptron
from vapnik.data_generator import sample_points
from sklearn.model_selection import train_test_split

from visualization.plot import plot_hyperplanes, plot_scatter, plot_average_errors

DIMENSION = 5

va = np.array([True, 0, 0, 0], dtype=np.bool)
vb = np.array([0, True, 0, 0], dtype=np.bool)
vc = np.array([0, 0, True, 0], dtype=np.bool)
vd = np.array([0, 0, 0, True], dtype=np.bool)

KERNELS = {
    0: [
        (('0x + 0y + 0z + 0'), [np.zeros(4, dtype=np.bool)])
    ],

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


def convert_X(X, kernels):
    cols = []
    for kernel in kernels:
        col = np.sum(X * kernel, axis=-1)
        cols.append(col)

    result = np.stack(cols, axis=-1)
    return result


def train_test_val_split(X, Y, train_size, validation_proportion):
    n = X.shape[0]
    X_train, X_test, Y_train, Y_test = \
        train_test_split(X, Y, test_size=(n - train_size), stratify=Y)
        # train_test_split(X, Y, test_size = (n - train_size), random_state = 42, stratify=Y)

    X_train, X_val, Y_train, Y_val = \
        train_test_split(X_train, Y_train, test_size=validation_proportion, stratify=Y_train)

    return X_train, X_test, X_val, Y_train, Y_test, Y_val


def run_all_experiments(verbose=True):
    a = 1
    b = 34
    c = -3
    d = -100
    n = 1003
    negative_proportion = 0.4
    noise = 0.3

    train_size = 30
    validation_proportion = 0.3
    epochs = 100

    if verbose:
        print('Objective: ', (a, b, c, d))

    X, Y = generate_dataset(a, b, c, d, n, negative_proportion, noise)

    X_train, X_test, X_val, Y_train, Y_test, Y_val = \
        train_test_val_split(X, Y, train_size, validation_proportion=validation_proportion)

    if verbose:
        print('n', X.shape, 'train', X_train.shape, 'test', X_test.shape, 'val', X_val.shape)

    best_scores = []
    all_eins = []
    all_eouts = []
    all_evals = []
    for n_param in [0, 1, 2, 3, 4]:

        if verbose:
            print('# parameters: %d' % (n_param))

        min_eout = float('inf')
        associated_ein = None
        associated_eval = None
        for representation, kernel in KERNELS[n_param]:
            # if np.sum(kernel) != 4:
            #     continue
            X_train_converted = convert_X(X_train, kernel)
            p = pocket_perceptron(X_train_converted, Y_train, epochs)

            predicted = p.predict_batch(X_train_converted)
            ein = np.sum(predicted != Y_train) / (Y_train.shape[0])
            # ein = np.sum(predicted != Y_train)

            all_eins.append((n_param, ein))

            X_test_converted = convert_X(X_test, kernel)
            predicted = p.predict_batch(X_test_converted)
            eout = np.sum(predicted != Y_test) / (Y_test.shape[0])
            # eout = np.sum(predicted != Y_test)

            all_eouts.append((n_param, eout))

            X_val_converted = convert_X(X_val, kernel)
            predicted = p.predict_batch(X_val_converted)
            eval = np.sum(predicted != Y_val) / (Y_val.shape[0])

            all_evals.append((n_param, eval))

            if eout < min_eout:
                min_eout = eout
                associated_ein = ein
                associated_eval = eval


            if verbose:
                print('%s | ein: %.4f | eval: %.4f |  eout: %.4f' % (representation, ein, eval, eout), '|', p.weights)

            # a, b, c, d = p.weights
        best_scores.append((associated_ein, associated_eval, min_eout))

    eins = np.array([x[0] for x in best_scores])
    eouts = np.array([x[1] for x in best_scores])
    evals = np.array([x[2] for x in best_scores])

    if verbose:
        plot_scatter(all_eins, all_evals, all_eouts, a, b, c, d, noise)
        # plot_graphic(eins, eouts)

        plot_hyperplanes('Ein', a, b, c, d, p, X_train, Y_train)
        plot_hyperplanes('Eval', a, b, c, d, p, X_val, Y_val)
        plot_hyperplanes('Eout', a, b, c, d, p, X_test, Y_test)

    return eins, evals, eouts


def generate_dataset(a, b, c, d, n, negative_proportion, noise):
    n_neg = int(n * negative_proportion)
    n_pos = n - n_neg
    x_pos, y_pos, z_pos = sample_points(a, b, c, d, n=n_pos, side=1, noise=noise)
    X_pos = np.stack((x_pos, y_pos, z_pos), axis=-1)
    Y_pos = np.ones(n_pos, dtype=np.int)
    x_neg, y_neg, z_neg = sample_points(a, b, c, d, n=n_neg, side=-1, noise=noise)
    X_neg = np.stack((x_neg, y_neg, z_neg), axis=-1)
    Y_neg = -1 * np.ones(n_neg, dtype=np.int)
    X = np.concatenate((X_pos, X_neg), axis=0)
    Y = np.concatenate((Y_pos, Y_neg), axis=0)
    ones = np.reshape(np.array(np.ones(n)), (n, 1))
    X = np.append(X, ones, axis=1)
    return X, Y


def main():
    avg_eins = np.zeros(DIMENSION)
    avg_evals = np.zeros(DIMENSION)
    avg_eouts = np.zeros(DIMENSION)

    n = 100
    for _ in tqdm(range(n)):
        eins, evals, eouts = run_all_experiments(False)
        avg_eins = avg_eins + eins/n
        avg_eouts = avg_eouts + eouts/n
        avg_evals = avg_evals + evals/n

    plot_average_errors(avg_eins, avg_evals, avg_eouts)


if __name__ == '__main__':
    # main()
    run_all_experiments()
