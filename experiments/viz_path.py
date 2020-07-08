import sys
sys.path.insert(0, "../")

from PINN_Survey.problems.burgers.v1 import Burgers
from PINN_Survey.problems.helmholtz.v1 import Helmholtz
from PINN_Survey.problems.burgers.data.load import load_burgers_bounds
from PINN_Survey.problems.helmholtz.data.load import load_helmholtz_bounds
from PINN_Survey.viz.loss_viz import viz_2d_svd, viz_base, viz_mesh, save_as_heightmap
from PINN_Base.util import bounds_from_data, random_choices, rel_error, percent_noise
import numpy as np
import tensorflow as tf


@viz_base
class Burgers_Viz(Burgers):
    pass


@viz_base
class Helmholtz_Viz(Helmholtz):
    pass


MAX_THREADS = 32
config = tf.ConfigProto(
    intra_op_parallelism_threads=MAX_THREADS
)


def quick_setup(loader, size=1000):

    X_true, U_true, _, _, _ = loader()

    [X, U] = random_choices(X_true, U_true, size=size)

    lower_bound, upper_bound = bounds_from_data(X_true)

    return X_true, U_true, X, U, lower_bound, upper_bound


def burgers_df_path(df_multiplier):

    X_true, U_true, X, U, lower_bound, upper_bound = quick_setup(
        load_burgers_bounds)

    X_df = random_choices(X)

    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
    nu = 0.01 / np.pi

    model_train = Burgers(lower_bound, upper_bound, layers,
                          nu, use_collocation_residual=False, session_config=config, df_multiplier=df_multiplier)

    fetches = [model_train.get_all_weight_variables()]
    vals = model_train.train_BFGS(X, U, X_df, False, fetches)
    U_hat = model_train.predict(X_true)
    print(rel_error(U_true, U_hat))

    vals = [v[0] for v in vals]

    w0 = model_train.get_all_weights()

    model_viz = Burgers_Viz(lower_bound, upper_bound,
                            layers, nu, use_collocation_residual=False, session_config=config, df_multiplier=df_multiplier)

    fig, ax, [_, _, loss, _] = viz_2d_svd(
        model_viz, X, U, X_df, w0, vals, 1024, 45)

    fig.savefig(
        f"burgers-df-path-contour-L({df_multiplier}).png")

    save_as_heightmap(
        f"burgers-df-path-heightmap-L({df_multiplier}).png", np.log(loss))


def burgers_regularization_path(data_noise, df_multiplier):

    X_true, U_true, X, U, lower_bound, upper_bound = quick_setup(
        load_burgers_bounds)

    if data_noise != 0:
        U = percent_noise(U, data_noise)

    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
    nu = 0.01 / np.pi

    model_train = Burgers(lower_bound, upper_bound, layers,
                          nu, use_differential_points=False, session_config=config, df_multiplier=df_multiplier)

    fetches = [model_train.get_all_weight_variables()]
    vals = model_train.train_BFGS(X, U, None, False, fetches)
    U_hat = model_train.predict(X_true)
    print(rel_error(U_true, U_hat))

    vals = [v[0] for v in vals]

    w0 = model_train.get_all_weights()

    model_viz = Burgers_Viz(lower_bound, upper_bound,
                            layers, nu, use_differential_points=False, session_config=config, df_multiplier=df_multiplier)

    fig, ax, [_, _, loss, _] = viz_2d_svd(
        model_viz, X, U, None, w0, vals, 1024, 45)

    fig.savefig(
        f"burgers-regularization-path-contour-N({data_noise})-L({df_multiplier}).png")

    save_as_heightmap(
        f"burgers-regularization-path-heightmap-N({data_noise})-L({df_multiplier}).png", np.log(loss))


def helmholtz_df_path(df_multiplier):

    X_true, U_true, X, U, lower_bound, upper_bound = quick_setup(
        load_helmholtz_bounds)

    X_df = random_choices(X)

    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
    nu = 0.01 / np.pi

    model_train = Helmholtz(lower_bound, upper_bound, layers,
                            1, 4, use_collocation_residual=False, session_config=config, df_multiplier=df_multiplier)

    fetches = [model_train.get_all_weight_variables()]
    vals = model_train.train_BFGS(X, U, X_df, False, fetches)
    U_hat = model_train.predict(X_true)
    print(rel_error(U_true, U_hat))

    vals = [v[0] for v in vals]

    w0 = model_train.get_all_weights()

    model_viz = Helmholtz_Viz(lower_bound, upper_bound,
                              layers, 1, 4, use_collocation_residual=False, session_config=config, df_multiplier=df_multiplier)

    fig, ax, [_, _, loss, _] = viz_2d_svd(
        model_viz, X, U, X_df, w0, vals, 1024, 45)

    fig.savefig(
        f"helmholtz-df-path-contour-L({df_multiplier}).png")

    save_as_heightmap(
        f"helmholtz-df-path-heightmap-L({df_multiplier}).png", np.log(loss))


def helmholtz_regularization_path(data_noise, df_multiplier):

    X_true, U_true, X, U, lower_bound, upper_bound = quick_setup(
        load_helmholtz_bounds)

    if data_noise != 0:
        U = percent_noise(U, data_noise)

    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]

    model_train = Helmholtz(lower_bound, upper_bound, layers,
                            1, 4, use_differential_points=False, session_config=config, df_multiplier=df_multiplier)

    fetches = [model_train.get_all_weight_variables()]
    vals = model_train.train_BFGS(X, U, None, False, fetches)
    U_hat = model_train.predict(X_true)
    print(rel_error(U_true, U_hat))

    vals = [v[0] for v in vals]

    w0 = model_train.get_all_weights()

    model_viz = Helmholtz_Viz(lower_bound, upper_bound,
                              layers, 1, 4, use_differential_points=False, session_config=config, df_multiplier=df_multiplier)

    fig, ax, [_, _, loss, _] = viz_2d_svd(
        model_viz, X, U, None, w0, vals, 1024, 45)

    fig.savefig(
        f"helmholtz-regularization-path-contour-N({data_noise})-L({df_multiplier}).png")

    save_as_heightmap(
        f"helmholtz-regularization-path-heightmap-N({data_noise})-L({df_multiplier}).png", np.log(loss))


if int(sys.argv[1]) == 0:
    burgers_regularization_path(.1, 0)
    burgers_regularization_path(.1, 1e-3)
    burgers_regularization_path(.1, 1e-2)
    burgers_regularization_path(.1, 1e-1)
    burgers_regularization_path(.1, 1)
elif int(sys.argv[1]) == 1:
    helmholtz_regularization_path(.1, 0)
    helmholtz_regularization_path(.1, 1e-3)
    helmholtz_regularization_path(.1, 1e-2)
    helmholtz_regularization_path(.1, 1e-1)
    helmholtz_regularization_path(.1, 1)
elif int(sys.argv[1]) == 2:
    burgers_df_path(1.0)
    burgers_df_path(0.1)
    burgers_df_path(1e-2)
elif int(sys.argv[1]) == 3:
    helmholtz_df_path(1.0)
    helmholtz_df_path(0.1)
    helmholtz_df_path(1e-2)
