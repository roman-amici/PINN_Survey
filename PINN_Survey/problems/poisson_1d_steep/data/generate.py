import numpy as np


def f(x):
    pi = np.pi
    return -0.1*(8*pi)**2 * np.sin(8*pi*x) - \
        2 * (80**2)*np.tanh(80*x)*(1/np.cosh(80*x)**2)


def solution(x):
    return 0.1*np.sin(np.pi*8*x) + np.tanh(80*x)


if __name__ == "__main__":
    n = 256
    data = np.empty((n, 2))

    for i, x in enumerate(np.linspace(-1, 1)):
        data[i, 0] = x
        data[i, 1] = solution(x)

    np.save("poisson_1d_steep.npy", data)
