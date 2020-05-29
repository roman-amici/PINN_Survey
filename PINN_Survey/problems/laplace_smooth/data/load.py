import os
import numpy as np
import PINN_Base.util as util


def to_grid(XU):

    u = XU[:, :, 2]

    x = XU[:, 0, 0]
    y = XU[0, :, 1]

    return [x, y, u]


def laplace_load_raw():

    path = os.path.dirname(os.path.abspath(__file__))
    filename = f"{path}/laplace_smooth.npy"
    XU = np.load(filename)

    return XU


def load_laplace_flat():

    XU = laplace_load_raw()

    n = XU.shape[0]*XU.shape[1]

    X = np.reshape(XU[:, :, 0:2], (n, 2))
    U = np.reshape(XU[:, :, 2], (n, 1))

    return X, U, to_grid(XU)


def load_laplace_grid():
    return to_grid(laplace_load_raw())


def load_laplace_bounds():
    X, U, [x, y, u] = load_laplace_flat()

    n = x.shape[0]

    X_left, U_left = util.get_u_const_idx_2d(x, y, u, idx_2=0)
    X_right, U_right = util.get_u_const_idx_2d(x, y, u, idx_2=(n-1))
    X_bottom, U_bottom = util.get_u_const_idx_2d(x, y, u, idx_1=0)
    X_top, U_top = util.get_u_const_idx_2d(x, y, u, idx_1=(n-1))

    return X, U, [X_left, X_top, X_right, X_bottom], [U_left, U_top, U_right, U_bottom], [x, y, u]
