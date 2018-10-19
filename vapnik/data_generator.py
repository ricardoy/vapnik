import numpy as np


K = 1000


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

    return x, y, z


def get_plane_points(a, b, c, d):
    x, y = np.meshgrid([-K, K], [-K, K])
    z = (-a*x -b*y -d) / c

    return x, y, z
