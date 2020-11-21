import sys
sys.path.insert(0, "../")

from PINN_Survey.problems.black_scholes_surrogate.v1 import BlackScholesSurrogate
from PINN_Survey.problems.black_scholes_surrogate.data.load import load_black_scholes_split, load_heston_split
from PINN_Base.util import bounds_from_data, random_choices, rmse, rel_error
from PINN_Survey.viz.loss_viz import viz_2d_layer_norm, viz_base, viz_mesh, save_as_heightmap
import numpy as np
import tensorflow as tf


@viz_base
class BlackScholes_Viz(BlackScholesSurrogate):
    pass


MAX_THREADS = 32
config = tf.ConfigProto(
    intra_op_parallelism_threads=MAX_THREADS
)


def black_scholes(optimizer="Adam", data_source="black_scholes", reg=1.0):

    if data_source == "black_scholes":
        X_train, U_train, X_test, U_test = load_black_scholes_split()
    else:
        X_train, U_train, X_test, U_test = load_heston_split()

    lower_bound, upper_bound = bounds_from_data(X_train)

    layers = [4, 20, 20, 20, 20, 20, 20, 20, 20, 1]
    model = BlackScholesSurrogate(lower_bound, upper_bound,
                                  layers, use_differential_points=False, session_config=config, df_multiplier=reg)

    if optimizer == "Adam":
        model.train_Adam_batched(
            X_train, U_train, None, batch_size=64, epochs=10)
    else:
        model.train_BFGS(X_train, U_train, None, True)

    w0 = model.get_all_weights()
    model_viz = BlackScholes_Viz(
        lower_bound, upper_bound, layers, use_differential_points=False, session_config=config, df_multiplier=reg)

    _, _, loss = viz_2d_layer_norm(
        model_viz, X_train, U_train, None, w0, 512)

    save_as_heightmap(
        f"black_scholes_train_{optimizer}_{reg}.png", np.log(loss))

    _, _, loss = viz_2d_layer_norm(
        model_viz, X_test, U_test, None, w0, 512)

    save_as_heightmap(
        f"black_scholes_test_{optimizer}_{reg}.png", np.log(loss))


if __name__ == "__main__":
    black_scholes("Adam", "black_scholes", 1.0)
    black_scholes("Adam", "black_scholes", 0.0)
    black_scholes("Adam", "heston", 1.0)
    black_scholes("Adam", "heston", 0.0)
