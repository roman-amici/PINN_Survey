import os
import numpy as np
import PINN_Base.util as util


def poisson_load_raw():
    path = os.path.dirname(os.path.abspath(__file__))
    filename = f"{path}/poisson_1d_steep.npy"
    XU = np.load(filename)

    return XU


def poisson_flat():

    XU = poisson_load_raw()
    X = XU[:, 0][:, None]
    U = XU[:, 1][:, None]

    return X, U


def poisson_bounds():

    X, U = poisson_flat()

    X_bounds = np.vstack([X[0:1, :], X[-1:-2, :]])
    U_bounds = np.vstack([U[0:1, :], U[-1:-2, :]])

    return X, U, X_bounds, U_bounds
