import sys
sys.path.insert(0, "../")

from PINN_Survey.problems.burgers.v1 import Burgers
from PINN_Survey.problems.burgers.data.load import load_burgers_bounds
from PINN_Survey.viz.loss_viz import viz_2d_layer_norm, viz_base, viz_mesh, save_as_heightmap
from PINN_Base.util import bounds_from_data, random_choices, rel_error, percent_noise
import numpy as np
import tensorflow as tf


class Burgers_Advection(Burgers):
    def _residual(self, u, x, _=None):

        du = tf.gradients(u, x)[0]

        u_x = du[:, 0]
        u_t = du[:, 1]

        return u_t + u[:, 0] * u_x


class Burgers_Diffusion(Burgers):

    def _residual(self, u, x, _=None):

        du = tf.gradients(u, x)[0]

        u_x = du[:, 0]
        u_t = du[:, 1]

        du2 = tf.gradients(u_x, x)[0]
        u_xx = du2[:, 0]

        return u_t - self.nu * u_xx


class Burgers_Overfull(Burgers):

    def _residual(self, u, x, _=None):
        du = tf.gradients(u, x)[0]

        u_x = du[:, 0]
        u_t = du[:, 1]

        dux2 = tf.gradients(u_x, x)[0]
        u_xx = dux2[:, 0]

        dut2 = tf.gradients(u_t, x)[0]
        u_tt = dut2[:, 1]

        return u + u_x + u[:, 0] * u_x + u_tt + u[:, 0] * u_t


@viz_base
class Burgers_Advection_Viz(Burgers_Advection):
    pass


@viz_base
class Burgers_Diffusion_Viz(Burgers_Diffusion):
    pass


@viz_base
class Burgers_Overfull_Viz(Burgers_Overfull):
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


def burgers_regularization(data_noise, df_multiplier, train, viz):

    X_true, U_true, X, U, lower_bound, upper_bound = quick_setup(
        load_burgers_bounds)

    if data_noise != 0:
        U = percent_noise(U, data_noise)

    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
    nu = 0.01 / np.pi
    model = train(lower_bound, upper_bound, layers,
                  nu, use_differential_points=False, session_config=config, df_multiplier=df_multiplier)
    model.train_BFGS(X, U, print_loss=True)
    U_hat = model.predict(X_true)
    print(rel_error(U_true, U_hat))

    w0 = model.get_all_weights()

    model_viz = viz(lower_bound, upper_bound,
                    layers, nu, use_differential_points=False, session_config=config, df_multiplier=df_multiplier)

    _, _, [_, _, loss] = viz_2d_layer_norm(model_viz, X, U, None, w0, 512)

    if train == Burgers_Advection:
        model_name = "advection"
    elif train == Burgers_Diffusion:
        model_name = "diffusion"
    elif train == Burgers_Overfull:
        model_name = "overfull"

    save_as_heightmap(
        f"burgers-{model_name}-N({data_noise})-L({df_multiplier}).png", np.log(loss))


if __name__ == "__main__":
    if int(sys.argv[1]) == 0:
        burgers_regularization(
            0.1, 0, Burgers_Advection, Burgers_Advection_Viz)
        burgers_regularization(
            0.1, 1e-3, Burgers_Advection, Burgers_Advection_Viz)
        burgers_regularization(
            0.1, 1e-2, Burgers_Advection, Burgers_Advection_Viz)
        burgers_regularization(
            0.1, 1e-1, Burgers_Advection, Burgers_Advection_Viz)
        burgers_regularization(
            0.1, 1.0, Burgers_Advection, Burgers_Advection_Viz)
    elif int(sys.argv[1]) == 1:
        burgers_regularization(
            0.1, 0, Burgers_Diffusion, Burgers_Diffusion_Viz)
        burgers_regularization(
            0.1, 1e-3, Burgers_Diffusion, Burgers_Diffusion_Viz)
        burgers_regularization(
            0.1, 1e-2, Burgers_Diffusion, Burgers_Diffusion_Viz)
        burgers_regularization(
            0.1, 1e-1, Burgers_Diffusion, Burgers_Diffusion_Viz)
        burgers_regularization(
            0.1, 1.0, Burgers_Diffusion, Burgers_Diffusion_Viz)
    elif int(sys.argv[1]) == 2:
        burgers_regularization(
            0.1, 0, Burgers_Overfull, Burgers_Overfull_Viz)
        burgers_regularization(
            0.1, 1e-3, Burgers_Overfull, Burgers_Overfull_Viz)
        burgers_regularization(
            0.1, 1e-2, Burgers_Overfull, Burgers_Overfull_Viz)
        burgers_regularization(
            0.1, 1e-1, Burgers_Overfull, Burgers_Overfull_Viz)
        burgers_regularization(
            0.1, 1.0, Burgers_Overfull, Burgers_Overfull_Viz)
