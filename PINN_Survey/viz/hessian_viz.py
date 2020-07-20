import numpy as np
from enum import Enum
from PINN_Survey.viz.loss_viz import weights_unit_vector, weights_unit_vector_layerwise, unwrap
from scipy.linalg import eigvalsh_tridiagonal
from tensorflow.keras.utils import Progbar


class Normalization(Enum):
    unit = 0
    layer = 1,
    svd = 2,


def make_T(alphas, betas):
    m = len(alphas)
    T = np.zeros((m, m))
    T[0, 0] = alphas[0]
    T[0, 1] = betas[0]

    for i in range(1, m - 1):
        T[i, i - 1] = betas[i - 1]
        T[i, i] = alphas[i]
        T[i, i + 1] = betas[i]

    T[-1, -1] = alphas[-1]
    T[-1, -2] = betas[-1]

    return T


def reorthogonalize(vs, w):
    for v in vs:
        w = w - ((w.T @ v) * v)

    return w


def random_unit_vector(size):
    v = np.random.normal(size=size)
    return v / np.linalg.norm(v)


def lanczos_mat_free(matvec_fn, n, m, v_start=None, T_matrix=True):

    if v_start is None:
        v = random_unit_vector(n)
    else:
        v = v_start
    vs = [v]

    w = matvec_fn(v)
    alpha = w.T @ v
    w = w - alpha * v

    alphas = []
    alphas.append(alpha)

    betas = []
    for i in range(1, m):
        beta = np.linalg.norm(w)
        if beta < 1e-6:
            print("Beta is zero, getting a new vector")
            v = reorthogonalize(vs, random_unit_vector(n))
        else:
            v = w / beta

        w = matvec_fn(v)
        alpha = w.T @ v
        w = w - alpha * v - beta * vs[-1]

        alphas.append(alpha)
        betas.append(beta)
        vs.append(v)

    if T_matrix:
        return make_T(alphas, betas)
    else:
        return alphas, betas


def viz_hessian_eigenvalue_ratio(
        model_viz, X, U, X_df, w0,
        lanczos_steps=100, grid_steps=50, weight_norm=Normalization.layer,
        min_vals=[-1, -1], max_vals=[1, 1]):

    n_params = unwrap(w0).shape[0]

    if weight_norm == Normalization.unit:
        d1 = weights_unit_vector(w0)
        d2 = weights_unit_vector_layerwise(w0)
    elif weight_norm == Normalization.layer:
        d1 = weights_unit_vector_layerwise(w0)
        d2 = weights_unit_vector_layerwise(w0)
    else:
        raise NotImplementedError("Svd not yet implemted")

    t1s = np.linspace(min_vals[0], max_vals[0], grid_steps)
    t2s = np.linspace(min_vals[1], max_vals[1], grid_steps)

    ratios = np.empty((grid_steps, grid_steps))
    progbar = Progbar(grid_steps * grid_steps)
    for i, t1 in enumerate(t1s):
        for j, t2 in enumerate(t2s):
            matvec_fn = model_viz.make_matvec_fn_with_weights(
                X, U, X_df, w0, d1, d2, t1, t2)

            alphas, betas = lanczos_mat_free(
                matvec_fn, n_params, lanczos_steps, T_matrix=False)

            eigenvalues = eigvalsh_tridiagonal(alphas, betas)
            ratios[i, j] = eigenvalues[0] / eigenvalues[-1]

            progbar.update(i * grid_steps + j + 1)

    return t1s, t2s, ratios
