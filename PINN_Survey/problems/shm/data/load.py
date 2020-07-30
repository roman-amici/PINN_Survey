from PINN_Survey.problems.shm.data.generate import generate, solution_true
import numpy as np


def generate_shm_bounds(omega, n, n_df):
    X_true, U_true = generate(n, omega)

    X = np.array([0.05, 0.1])[:, None]
    U = solution_true(X, omega)
    X_df, _ = generate(n_df, omega)

    return X_true, U_true, X, U, X_df
