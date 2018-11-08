from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

# Fixing random state for reproducibility
# np.random.seed(19680801)
from vapnik.data_generator import sample_points, get_plane_points




'''
def get_test_data(delta=0.05):

    from matplotlib.mlab import  bivariate_normal
    x = y = np.arange(-3.0, 3.0, delta)
    X, Y = np.meshgrid(x, y)

    Z1 = bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
    Z2 = bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
    Z = Z2 - Z1

    X = X * 10
    Y = Y * 10
    Z = Z * 500
    return X, Y, Z

'''


def plot_plane(a, b, c, d, ax):
    x, y, z = get_plane_points(a, b, c, d)
    ax.plot_surface(x, y, z, alpha=0.2)


def main():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # # sample data
    # x, y, z = axes3d.get_test_data(0.05)
    # ax.plot_wireframe(x,y,z, rstride=1, cstride=1)

    noise = 0.
    a = 1
    b = 1
    c = -1
    d = 2
    plot_plane(a, b, c, d, ax)


    # For each set of style and range settings, plot n random points in the box
    # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    n = 1000
    for color, marker, side in [('r', 'o', 1), ('b', '^', -1)]:
        x, y, z = sample_points(a, b, c, d, side=side, n=n, noise=noise)
        ax.scatter(x, y, z, c=color, marker=marker)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


if __name__ == '__main__':
    main()