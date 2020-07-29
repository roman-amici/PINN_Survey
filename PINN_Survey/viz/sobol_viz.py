import numpy as np
from PINN_Survey.viz.loss_viz import wrap, unwrap
import tensorflow as tf


def run_on_matrix_of_weights(model, X, U, X_df, M, w0):

    losses = np.empty(M.shape[0])
    for i in range(M.shape[0]):

        w_flat = M[i, :]
        w = wrap(w_flat, w0)

        losses[i] = model.run_with_weights(X, U, X_df, w, model.loss)

    return losses


def sobol_expansion(model, X, U, X_df, w0, bounds=[-1e-2, 1e2], n=150):

    w0_flat = unwrap(w0)
    m = len(w0_flat)

    A = np.random.uniform(low=bounds[0], high=bounds[1], size=(n, m))
    B = np.random.uniform(low=bounds[0], high=bounds[1], size=(n, m))

    # A and B are perturbations around w0
    for i in range(A.shape[0]):
        A[i, :] += w0_flat
        B[i, :] += w0_flat

    losses_A = run_on_matrix_of_weights(model, X, U, X_df, A, w0)
    losses_B = run_on_matrix_of_weights(model, X, U, X_df, B, w0)

    # Matrix to store the resuts for each submatrix
    progbar = tf.keras.utils.Progbar(m)
    losses_AB = np.empty((m, n))
    for j in range(m):
        AB = A.copy()
        AB[:, j] = B[:, j]

        losses_AB[j, :] = run_on_matrix_of_weights(model, X, U, X_df, AB, w0)

        progbar.update(j + 1)

    return losses_A, losses_B, losses_AB
