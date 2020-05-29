import numpy as np


def f(x, y):
    pi = np.pi
    sin = np.sin
    tanh = np.tanh
    cosh = np.cosh

    t1 = -0.1*(2*pi)**2 * sin(2*pi*x) - 2*100*tanh(10*x)*(1/cosh(10*x))**2
    t2 = 0.1*sin(2*pi*x) + tanh(2*pi*x)
    t3 = -(2*pi)**2 * sin(2*pi*y)

    return t1 * np.sin(2*np.pi*y) + t2*t3


def solution(x, y):
    return (0.1*np.sin(2*np.pi*x) + np.tanh(10*x))*np.sin(2*np.pi*y)


if __name__ == "__main__":
    n = 256
    data = np.empty((n, n, 3))

    for i, x in enumerate(np.linspace(-1, 1, n)):
        for j, y in enumerate(np.linspace(-1, 1, n)):
            idx = i*n + j
            data[i, j, 0] = x
            data[i, j, 1] = y
            data[i, j, 2] = solution(x, y)

    np.save("poisson_2d_steep.npy", data)
