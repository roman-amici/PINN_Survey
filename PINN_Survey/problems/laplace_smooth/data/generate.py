import numpy as np


def solution(x, y):
    top = 2*(1+y)
    bottom = (3+x)**2 + (1 + y)**2

    return top / bottom


if __name__ == "__main__":
    nx = 256
    ny = 256

    data = np.empty((nx, ny, 3))

    for i, x in enumerate(np.linspace(-1, 1, nx)):
        for j, y in enumerate(np.linspace(-1, 1, ny)):
            u = solution(x, y)
            data[i, j, 0] = x
            data[i, j, 1] = y
            data[i, j, 2] = u

    np.save("laplace_smooth.npy", data)
