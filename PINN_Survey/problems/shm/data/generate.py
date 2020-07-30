import numpy as np


def solution_true(X, omega):
    return np.sin(X * omega)


def generate(n, omega, bounds=[0, 1]):
    X = np.linspace(bounds[0], bounds[1], n)[:, None]
    U = solution_true(X, omega)

    return X, U


if __name__ == "__main__":
    X, U = generate(1024, 4 * np.pi)

    data = np.empty((1024, 2))

    data[:, 0] = X[:, 0]
    data[:, 1] = U[:, 0]

    np.save("shm.npy", data)
