import numpy as np


def source(x, y, a, b, k):
    pi = np.pi
    sin = np.sin
    -(a*pi)**2 * sin(a*pi*x) * sin(b*pi*y) - \
        (b*pi)**2 * sin(a*pi*x)*np.sin(b*pi*y) + \
        k**2*sin(a*pi*x)*sin(b*pi*y)


def solution(x, y, a, b):
    return np.sin(a*np.pi*x) * np.sin(b*np.pi*y)


if __name__ == "__main__":
    nx = 256
    ny = 256

    data = np.empty((nx, ny, 3))

    for i, x in enumerate(np.linspace(-1, 1, nx)):
        for j, y in enumerate(np.linspace(-1, 1, ny)):
            # An-isotropic
            u = solution(x, y, 1, 4)
            data[i, j, 0] = x
            data[i, j, 1] = y
            data[i, j, 2] = u

    np.save("helmholtz-analytic-stiff.npy", data)
