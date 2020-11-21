import os
import numpy as np
import PINN_Base.util as util


def load_black_scholes_split():
    path = os.path.dirname(os.path.abspath(__file__))
    X_train = np.load(f"{path}/black_scholes_train_X.npy")
    X_test = np.load(f"{path}/black_scholes_test_X.npy")

    U_train = np.load(f"{path}/black_scholes_train_U.npy")
    U_test = np.load(f"{path}/black_scholes_test_U.npy")

    return X_train, U_train, X_test, U_test


def load_heston_split():
    path = os.path.dirname(os.path.abspath(__file__))
    X_train = np.load(f"{path}/heston_train_X.npy")
    X_test = np.load(f"{path}/heston_test_X.npy")

    U_train = np.load(f"{path}/heston_train_U.npy")
    U_test = np.load(f"{path}/heston_test_U.npy")

    return X_train, U_train, X_test, U_test
