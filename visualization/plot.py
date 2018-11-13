from mpl_toolkits.mplot3d import axes3d
import numpy as np
from matplotlib import pyplot as plt

from vapnik.data_generator import get_plane_points


def plot_hyperplanes(title, a, b, c, d, p, X, Y):
    fig = plt.figure()
    fig.suptitle(title)
    ax = fig.add_subplot(111, projection='3d')
    plot_plane(a, b, c, d, ax, color='r')

    a, b, c, d = p.weights
    plot_plane(a, b, c, d, ax, color='g')

    X_pos = X[Y == 1]
    ax.scatter(X_pos[:,0], X_pos[:,1], X_pos[:,2], c='r', marker='o')
    X_neg = X[Y == -1]
    ax.scatter(X_neg[:,0], X_neg[:,1], X_neg[:,2], c='b', marker='^')
    # print(z_pos)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


def plot_scatter(all_eins, all_evals, all_eouts, a, b, c, d, noise):
    x = [t[0] for t in all_eins]
    y = [t[1] for t in all_eins]

    fig = plt.figure()
    fig.suptitle('%.1fx %+.1fy %+.1fz %+.1f, noise=%.2f' % (a, b, c, d, noise))
    ax = fig.add_subplot(111)
    # ax.scatter(x, y, label='Ein', marker='o', alpha=0.2)
    #
    y = [t[1] for t in all_evals]
    ax.scatter(x, y, label='Eval', marker='*', alpha=0.2)

    y = [t[1] for t in all_eouts]
    ax.scatter(x, y, label='Eout', marker='^', alpha=0.2)

    ax.legend()
    ax.set_ylim([0, 1.])
    ax.set_ylabel('Error')
    ax.set_xlabel('Number of parameters')
    ax.tick_params(axis='x', which='major', labelsize=8)
    # plt.xticks()

    # for x, y in zip(interval, eouts):
    #     ax.annotate('%.5f'%(y), xy=(x, y))
    #
    # for x, y in zip(interval, eins):
    #     ax.annotate('%.5f'%(y), xy=(x, y))

    plt.show()


def plot_average_errors(eins, evals, eouts):
    interval = [0, 1, 2, 3, 4]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(interval, eins, label='Ein', marker='o')
    ax.plot(interval, eouts, label='Eout', marker='^')
    ax.plot(interval, evals, label='Eval', marker='_')
    # ay = fig.add_subplot(111)
    ax.legend()
    if np.max(eouts) <= 1:
        ax.set_ylim([0, 1.])
    ax.set_ylabel('Error')
    ax.set_xlabel('Number of parameters')
    ax.tick_params(axis='x', which='major', labelsize=8)
    # plt.xticks()

    for x, y in zip(interval, eouts):
        ax.annotate('%.5f'%(y), xy=(x, y))

    for x, y in zip(interval, eins):
        ax.annotate('%.5f'%(y), xy=(x, y))

    for x, y in zip(interval, evals):
        ax.annotate('%.5f'%(y), xy=(x, y))

    plt.show()


def plot_plane(a, b, c, d, ax, color):
    x, y, z = get_plane_points(a, b, c, d)
    ax.plot_surface(x, y, z, alpha=0.2, color=color)