import numpy as np


K = 10000


def randrange(n, vmin, vmax):
    '''
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    '''
    return (vmax - vmin)*np.random.rand(n) + vmin


def sample_points(a, b, c, d, side=1, noise=0.1, n=10):
    if side not in [1, -1]:
        raise Exception('invalid argument: ', side)
    x = randrange(n, -K, K)
    y = randrange(n, -K, K)
    z = (-a*x -b*y -d) / c

    distance_to_hyperplane = randrange(n, 0, K/2)

    if side == 1:
        keep_label = np.random.choice([1, -1], p=[1-noise, noise], size=n)
    else:
        keep_label = np.random.choice([1, -1], p=[noise, 1-noise], size=n)

    x = x + keep_label * distance_to_hyperplane * a
    y = y + keep_label * distance_to_hyperplane * b
    z = z + keep_label * distance_to_hyperplane * c

    # x = x + keep_label * K * a
    # y = y + keep_label * K * b
    # z = z + keep_label * K * c

    return x, y, z


def get_plane_points(a, b, c, d):
    x, y = np.meshgrid([-K, K], [-K, K])
    z = (-a*x -b*y -d) / c

    return x, y, z


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