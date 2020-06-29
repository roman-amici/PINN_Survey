import sys
sys.path.insert(0, "../")

from PINN_Survey.problems.burgers.v1 import Burgers, Burgers_Sphere_Mesh
from PINN_Survey.problems.helmholtz.v1 import Helmholtz, Helmholtz_Sphere_Mesh
from PINN_Survey.problems.burgers.data.load import load_burgers_bounds
from PINN_Survey.problems.helmholtz.data.load import load_helmholtz_bounds
from PINN_Survey.viz.loss_viz import viz_2d_layer_norm, viz_base, viz_mesh, save_as_heightmap
from PINN_Base.util import bounds_from_data, random_choices, rel_error, percent_noise
import numpy as np
import tensorflow as tf


class Burgers_MLP(Burgers):

    def _loss(self, U_hat, U_hat_df):
        self.mse = tf.reduce_mean(tf.square(self.U - U_hat))
        return self.mse


@viz_base
class Burgers_MLP_Viz(Burgers_MLP):
    pass


class Helmholtz_MLP(Helmholtz):

    def _loss(self, U_hat, U_hat_df):
        self.mse = tf.reduce_mean(tf.square(self.U - U_hat))
        return self.mse


@viz_base
class Helmholtz_MLP_Viz(Helmholtz_MLP):

    def _loss(self, U_hat, U_hat_df):
        self.mse = tf.reduce_mean(tf.square(self.U - U_hat))
        return self.mse


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


def quick_setup(loader, size=1000):

    X_true, U_true, _, _, _ = loader()

    [X, U] = random_choices(X_true, U_true, size=size)

    lower_bound, upper_bound = bounds_from_data(X_true)

    return X_true, U_true, X, U, lower_bound, upper_bound


def burgers_MLP_base(data_noise):
    X_true, U_true, X, U, lower_bound, upper_bound = quick_setup(
        load_burgers_bounds)

    if data_noise != 0:
        U = percent_noise(U, data_noise)

    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
    nu = 0.01 / np.pi
    model = Burgers_MLP(lower_bound, upper_bound, layers,
                        nu, use_differential_points=False, session_config=config)
    model.train_BFGS(X, U, print_loss=True)
    U_hat = model.predict(X_true)
    print(rel_error(U_true, U_hat))

    w0 = model.get_all_weights()

    model_viz = Burgers_MLP_Viz(lower_bound, upper_bound,
                                layers, nu, use_differential_points=False, session_config=config)

    _, _, [_, _, loss] = viz_2d_layer_norm(model_viz, X, U, None, w0, 512)

    save_as_heightmap(f"burgers-MLP-base-{data_noise}.png", loss)


def burgers_regularization_base(data_noise):

    X_true, U_true, X, U, lower_bound, upper_bound = quick_setup(
        load_burgers_bounds)

    if data_noise != 0:
        U = percent_noise(U, data_noise)

    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
    nu = 0.01 / np.pi
    model = Burgers(lower_bound, upper_bound, layers,
                    nu, use_differential_points=False, session_config=config)
    model.train_BFGS(X, U, print_loss=True)
    U_hat = model.predict(X_true)
    print(rel_error(U_true, U_hat))

    w0 = model.get_all_weights()

    model_viz = Burgers_Viz(lower_bound, upper_bound,
                            layers, nu, use_differential_points=False, session_config=config)

    _, _, [_, _, loss] = viz_2d_layer_norm(model_viz, X, U, None, w0, 512)

    save_as_heightmap(f"burgers-regularization-base-{data_noise}.png", loss)


def burgers_regularization_mesh(data_noise):

    X_true, U_true, X, U, lower_bound, upper_bound = quick_setup(
        load_burgers_bounds)

    if data_noise != 0:
        U = percent_noise(U, data_noise)

    layers_mesh = [2, 20, 20, 20, 20, 20, 20]
    layers_approx = [2, 20, 20, 1]
    nu = 0.01 / np.pi
    model = Burgers_Sphere_Mesh(lower_bound, upper_bound, layers_approx, layers_mesh,
                                nu, use_differential_points=False, session_config=config)
    model.train_BFGS(X, U, print_loss=True)
    U_hat = model.predict(X_true)
    print(rel_error(U_true, U_hat))

    w0 = model.get_all_weights()

    model_viz = Burgers_Sphere_Mesh_Viz(lower_bound, upper_bound, layers_approx, layers_mesh,
                                        nu, use_differential_points=False, session_config=config)

    _, _, [_, _, loss] = viz_2d_layer_norm(model_viz, X, U, None, w0, 512)

    save_as_heightmap(f"burgers-regularization-mesh-{data_noise}.png", loss)


def helmholtz_regularization_base(data_noise):

    X_true, U_true, X, U, lower_bound, upper_bound = quick_setup(
        load_helmholtz_bounds)

    if data_noise != 0:
        U = percent_noise(U, data_noise)

    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
    model = Helmholtz(lower_bound, upper_bound, layers,
                      1, 4, use_differential_points=False, session_config=config)
    model.train_BFGS(X, U, print_loss=True)
    U_hat = model.predict(X_true)
    print(rel_error(U_true, U_hat))

    w0 = model.get_all_weights()

    model_viz = Helmholtz_Viz(lower_bound, upper_bound,
                              layers, 1, 4, use_differential_points=False, session_config=config)

    _, _, [_, _, loss] = viz_2d_layer_norm(model_viz, X, U, None, w0, 512)

    save_as_heightmap(f"helmholtz-regularization-base-{data_noise}.png", loss)


def helmholtz_MLP_base(data_noise):

    X_true, U_true, X, U, lower_bound, upper_bound = quick_setup(
        load_helmholtz_bounds)

    if data_noise != 0:
        U = percent_noise(U, data_noise)

    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
    model = Helmholtz_MLP(lower_bound, upper_bound, layers,
                          1, 4, use_differential_points=False, session_config=config)
    model.train_BFGS(X, U, print_loss=True)
    U_hat = model.predict(X_true)
    print(rel_error(U_true, U_hat))

    w0 = model.get_all_weights()

    model_viz = Helmholtz_MLP_Viz(lower_bound, upper_bound,
                                  layers, 1, 4, use_differential_points=False, session_config=config)

    _, _, [_, _, loss] = viz_2d_layer_norm(model_viz, X, U, None, w0, 512)

    save_as_heightmap(f"helmholtz-MLP-base-{data_noise}.png", loss)


def helmholtz_regularization_mesh(data_noise):

    X_true, U_true, X, U, lower_bound, upper_bound = quick_setup(
        load_helmholtz_bounds)

    if data_noise != 0:
        U = percent_noise(U, data_noise)

    layers_mesh = [2, 20, 20, 20, 20, 20, 20]
    layers_approx = [2, 20, 20, 1]
    model = Helmholtz_Sphere_Mesh(lower_bound, upper_bound, layers_approx, layers_mesh,
                                  1, 4, use_differential_points=False, session_config=config)
    model.train_BFGS(X, U, print_loss=True)
    U_hat = model.predict(X_true)
    print(rel_error(U_true, U_hat))

    w0 = model.get_all_weights()

    model_viz = Helmholtz_Sphere_Mesh_Viz(lower_bound, upper_bound, layers_approx, layers_mesh,
                                          1, 4, use_differential_points=False, session_config=config)

    _, _, [_, _, loss] = viz_2d_layer_norm(model_viz, X, U, None, w0, 512)

    save_as_heightmap(f"helmholtz-regularization-mesh-{data_noise}.png", loss)


if __name__ == "__main__":
    burgers_MLP_base(0)
    burgers_MLP_base(1e-2)
    burgers_MLP_base(1e-1)
    burgers_regularization_base(0)
    burgers_regularization_base(1e-2)
    burgers_regularization_base(1e-1)
    helmholtz_MLP_base(0)
    helmholtz_MLP_base(1e-1)
    helmholtz_MLP_base(1e-2)
    helmholtz_regularization_base(0)
    helmholtz_regularization_base(1e-1)
    helmholtz_regularization_base(1e-2)
    burgers_regularization_mesh(0)
    burgers_regularization_mesh(1e-1)
    burgers_regularization_mesh(1e-2)
