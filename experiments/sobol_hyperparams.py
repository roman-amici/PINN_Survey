import sys
sys.path.append('../')
from PINN_Survey.problems.burgers.data.load import load_burgers_bounds
from PINN_Base.util import bounds_from_data, random_choice, random_choices
from PINN_Survey.viz.loss_viz import viz_base
from PINN_Survey.viz.sobol_viz import sobol_hyperparams, sobol_coefficients
from PINN_Survey.problems.burgers.v1 import Burgers
import numpy as np
import matplotlib.pyplot as plt


def run_test():

    X_true, U_true, X_boundary, U_boundary, _ = load_burgers_bounds()

    lower_bound, upper_bound = bounds_from_data(X_true)

    X = np.vstack(X_boundary)
    U = np.vstack(U_boundary)

    X_df = random_choice(X_true)

    def model_constructor(width, depth, l):
        layers = [2] + [width] * depth + [1]

        model_train = Burgers(0.01 / np.pi, lower_bound, upper_bound,
                              layers, use_collocation_residual=False, df_multiplier=l)

        return model_train

    losses_A, losses_B, losses_AB = sobol_hyperparams(
        model_constructor, X, U, X_df, bounds_high=[60, 16, 1.0], n=25)

    return sobol_coefficients(losses_A, losses_B, losses_AB)


if __name__ == "__main__":
    coefs = run_test()
    print(coefs)
    with open("result.txt", "w+") as f:
        f.write(coefs.__str__())
