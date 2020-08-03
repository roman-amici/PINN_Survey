import numpy as np
from PINN_Survey.viz.loss_viz import wrap, unwrap
import tensorflow as tf


def sobol_coefficients(losses_A, losses_B, losses_AB):

    m = losses_AB.shape[0]

    var_exp = np.empty(m)
    exp_var = np.empty(m)

    for j in range(m):
        var_exp[j] = np.mean(losses_B * (losses_AB[j, :] - losses_A))
        exp_var = np.mean((losses_A - losses_AB[j, :])**2) / 2

    return exp_var / (exp_var + var_exp)


def run_on_matrix_of_weights(model, X, U, X_df, M, w0):

    losses = np.empty(M.shape[0])
    for i in range(M.shape[0]):

        w_flat = M[i, :]
        w = wrap(w_flat, w0)

        losses[i] = model.run_with_weights(X, U, X_df, w, model.loss)

    return losses


def run_on_submatrix_of_weights(model, X, U, X_df, M, w0, w_flat, w_start, w_end):
    w_flat = w_flat.copy()
    losses = np.empty(M.shape[0])
    for i in range(M.shape[0]):
        w_sub = M[i, :]
        w_flat[w_start:w_end] = w_sub

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


def get_last_layer_weight_index(w0):
    assert(len(w0) == 2)

    weight_layers = w0[0]

    weights_start = 0
    for w_layer in weight_layers[:-1]:
        weights_start += w_layer.size

    weights_end = weights_start + weight_layers[-1].size

    return weights_start, weights_end


def sobol_last_layer(model, X, U, X_df, w0, bounds=[-1e-2, 1e-2], n=200):
    w0_flat = unwrap(w0)

    w_start, w_end = get_last_layer_weight_index(w0)
    w0_last = w0_flat[w_start:w_end]

    m = w_end - w_start

    A = np.random.uniform(low=bounds[0], high=bounds[1], size=(n, m))
    B = np.random.uniform(low=bounds[0], high=bounds[1], size=(n, m))

    for i in range(A.shape[0]):
        A[i, :] += w0_last
        B[i, :] += w0_last

    losses_A = run_on_submatrix_of_weights(
        model, X, U, X_df, A, w0, w0_flat, w_start, w_end)
    losses_B = run_on_submatrix_of_weights(
        model, X, U, X_df, B, w0, w0_flat, w_start, w_end)

    progbar = tf.keras.utils.Progbar(m)
    losses_AB = np.empty((m, n))
    for j in range(m):
        AB = A.copy()
        AB[:, j] = B[:, j]

        losses_AB[j, :] = run_on_submatrix_of_weights(
            model, X, U, X_df, AB, w0, w0_flat, w_start, w_end)

        progbar.update(j + 1)

    return losses_A, losses_B, losses_AB


def run_matrix_of_hyperparams(model_constructor, X, U, X_df, M):
    losses = np.empty(M.shape[0])

    progbar = tf.keras.utils.Progbar(M.shape[0])
    for i in range(M.shape[0]):
        hyperparams = M[i, :]
        model = model_constructor(
            int(hyperparams[0]), int(hyperparams[1]), hyperparams[2])

        model.train_BFGS(X, U, X_df, False)
        losses[i] = model.get_loss(X, U, X_df)

        progbar.update(i + 1)

    return losses


def sobol_hyperparams(model_constructor, X, U, X_df, bounds_low=[8, 4, 0], bounds_high=[100, 16, 10], n=100):
    A = np.random.uniform(bounds_low, bounds_high, (n, 3))
    B = np.random.uniform(bounds_low, bounds_high, (n, 3))

    losses_A = run_matrix_of_hyperparams(model_constructor, X, U, X_df, A)
    losses_B = run_matrix_of_hyperparams(model_constructor, X, U, X_df, B)

    losses_AB = np.empty((3, n))
    for j in range(3):

        print(f"{j+1}/3")
        AB = A.copy()
        AB[:, j] = B[:, j]

        losses_AB[j, :] = run_matrix_of_hyperparams(
            model_constructor, X, U, X_df, AB)

    return losses_A, losses_B, losses_AB
