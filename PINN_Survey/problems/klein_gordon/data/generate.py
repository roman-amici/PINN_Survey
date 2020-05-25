import numpy as np


def solution(x, t):
    return x*np.cos(5*np.pi*t) + (x*t)**3


def forcing(x, t, alpha, beta, gamma, k):
    pi = np.pi
    cos = np.cos

    return -x*25*pi**2 * cos(5*pi*t) + 6*t*x**3 + \
        6*alpha*x*t**3 + \
        beta*(x*cos(5*pi*t) + (x*t)**3) + \
        gamma*(x*cos(5*pi*t) + (x*t)**3)**k


if __name__ == "__main__":
    nx = 256
    nt = 256

    data = np.empty((nx, nt, 3))

    for i, x in enumerate(np.linspace(0, 1, nx)):
        for j, t in enumerate(np.linspace(0, 1, nt)):
            u = solution(x, t)
            data[i, j, 0] = x
            data[i, j, 1] = t
            data[i, j, 2] = u

    np.save("klein-gordon.npy", data)
