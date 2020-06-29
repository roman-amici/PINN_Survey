import sys
sys.path.insert(0, "../")

from PINN_Survey.problems.burgers.v1 import Burgers, Burgers_Sphere_Mesh
from PINN_Survey.problems.helmholtz.v1 import Helmholtz, Helmholtz_Sphere_Mesh
from PINN_Survey.problems.burgers.data.load import load_burgers_bounds
from PINN_Survey.problems.helmholtz.data.load import load_helmholtz_bounds
from PINN_Survey.viz.loss_viz import viz_2d_layer_norm, viz_base, viz_mesh, save_as_heightmap
from PINN_Base.util import bounds_from_data, random_choices, rmse, rel_error
import numpy as np
import tensorflow as tf


@viz_base
class Burgers_Viz(Burgers):
    pass


@viz_base
class Helmholtz_Viz(Helmholtz):
    pass


@viz_mesh
class Burgers_Sphere_Mesh_Viz(Burgers_Sphere_Mesh):
    pass


@viz_mesh
class Helmholtz_Sphere_Mesh_Viz(Helmholtz_Sphere_Mesh):
    pass


MAX_THREADS = 32
config = tf.ConfigProto(
    intra_op_parallelism_threads=MAX_THREADS
)


def quick_setup(loader):

    X_true, U_true, X_bounds, U_bounds, _ = loader()

    X = np.vstack(X_bounds)
    U = np.vstack(U_bounds)
    X_df = random_choices(X_true)

    lower_bound, upper_bound = bounds_from_data(X_true)

    return X_true, U_true, X, U, X_df, lower_bound, upper_bound


def burgers_boundary_base():

    X_true, U_true, X, U, X_df, lower_bound, upper_bound = quick_setup(
        load_burgers_bounds)

    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
    nu = 0.01 / np.pi
    model = Burgers(lower_bound, upper_bound, layers,
                    nu, use_collocation_residual=False, session_config=config)
    model.train_BFGS(X, U, X_df, True)
    U_hat = model.predict(X_true)
    print(rel_error(U_true, U_hat))

    w0 = model.get_all_weights()

    model_viz = Burgers_Viz(lower_bound, upper_bound,
                            layers, nu, use_collocation_residual=False, session_config=config)

    _, _, [_, _, loss] = viz_2d_layer_norm(model_viz, X, U, X_df, w0, 512)

    save_as_heightmap("burgers-boundary-base.png", loss)


def burgers_boundary_mesh():
    X_true, U_true, X, U, X_df, lower_bound, upper_bound = quick_setup(
        load_burgers_bounds)

    layers_mesh = [2, 20, 20, 20, 20, 20, 20]
    layers_approx = [2, 20, 20, 1]
    nu = 0.01 / np.pi
    model = Burgers_Sphere_Mesh(lower_bound, upper_bound, layers_approx, layers_mesh,
                                nu, use_collocation_residual=False, session_config=config)
    model.train_BFGS(X, U, X_df, True)
    U_hat = model.predict(X_true)
    print(rel_error(U_true, U_hat))

    w0 = model.get_all_weights()

    model_viz = Burgers_Sphere_Mesh_Viz(lower_bound, upper_bound, layers_approx, layers_mesh,
                                        nu, use_collocation_residual=False, session_config=config)

    _, _, [_, _, loss] = viz_2d_layer_norm(model_viz, X, U, X_df, w0, 512)

    save_as_heightmap("burgers-boundary-mesh.png", loss)


def helmholtz_boundary_base(df_multiplier):

    X_true, U_true, X, U, X_df, lower_bound, upper_bound = quick_setup(
        load_helmholtz_bounds)

    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
    model = Helmholtz(lower_bound, upper_bound, layers,
                      1, 4, use_collocation_residual=False, session_config=config, df_multiplier=df_multiplier)
    model.train_BFGS(X, U, X_df, True)
    U_hat = model.predict(X_true)
    print(rel_error(U_true, U_hat))

    w0 = model.get_all_weights()

    model_viz = Helmholtz_Viz(lower_bound, upper_bound,
                              layers, 1, 4, use_collocation_residual=False, session_config=config, df_multiplier=df_multiplier)

    _, _, [_, _, loss] = viz_2d_layer_norm(model_viz, X, U, X_df, w0, 512)

    save_as_heightmap(f"helmholtz-boundary-base-{df_multiplier}.png", loss)


def helmholtz_boundary_mesh(df_multiplier):
    X_true, U_true, X, U, X_df, lower_bound, upper_bound = quick_setup(
        load_helmholtz_bounds)

    layers_mesh = [2, 20, 20, 20, 20, 20, 20]
    layers_approx = [2, 20, 20, 1]
    model = Helmholtz_Sphere_Mesh(lower_bound, upper_bound, layers_approx, layers_mesh,
                                  1, 4, use_collocation_residual=False, session_config=config, df_multiplier=df_multiplier)
    model.train_BFGS(X, U, X_df, True)
    U_hat = model.predict(X_true)
    print(rel_error(U_true, U_hat))

    w0 = model.get_all_weights()

    model_viz = Helmholtz_Sphere_Mesh_Viz(lower_bound, upper_bound, layers_approx, layers_mesh,
                                          1, 4, use_collocation_residual=False, session_config=config, df_multiplier=df_multiplier)

    _, _, [_, _, loss] = viz_2d_layer_norm(model_viz, X, U, X_df, w0, 512)

    save_as_heightmap(f"helmholtz-boundary-mesh-{df_multiplier}.png", loss)


if __name__ == "__main__":
    burgers_boundary_base()
    burgers_boundary_mesh()
    helmholtz_boundary_base(1.0)
    helmholtz_boundary_base(1e-2)
    helmholtz_boundary_mesh(1.0)
    helmholtz_boundary_mesh(1e-2)
