import os
import numpy as np
import PINN_Base.util as util

# TODO: Refactor so that this and helmholtz use the same code.


def to_grid(XU):

    u = XU[:, :, 2]

    x = XU[:, 0, 0]
    t = XU[0, :, 1]

    return [x, t, u]


def klein_gordon_load_raw():
    path = os.path.dirname(os.path.abspath(__file__))
    filename = f"{path}/klein_gordon.npy"
    XU = np.load(filename)

    return XU


def load_klein_gordon_flat():
    XU = klein_gordon_load_raw()

    n = XU.shape[0]*XU.shape[1]

    X = np.reshape(XU[:, :, 0:2], (n, 2))
    U = np.reshape(XU[:, :, 2], (n, 1))

    return X, U, to_grid(XU)


def load_klein_gordon_grid():
    XU = klein_gordon_load_raw()

    return to_grid(XU)


def load_klein_gordon_bounds():
    X, U, [x, t, u] = load_klein_gordon_flat()

    n = x.shape[0]

    X_init, U_init = util.get_u_const_idx_2d(x, t, u, idx_2=0)
    X_bottom, U_bottom = util.get_u_const_idx_2d(x, t, u, idx_1=0)
    X_top, U_top = util.get_u_const_idx_2d(x, t, u, idx_1=(n-1))

    return X, U, [X_init, X_top, X_bottom], [U_init, U_top, U_bottom], [x, t, u]
