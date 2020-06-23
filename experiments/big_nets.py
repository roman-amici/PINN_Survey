import sys
sys.path.insert(0, "../")

from PINN_Survey.problems.burgers.v1 import Burgers
from PINN_Survey.problems.burgers.data.load import load_burgers_bounds
from PINN_Survey.problems.helmholtz.data.load import load_helmholtz_bounds
from PINN_Survey.problems.helmholtz.v1 import Helmholtz
from PINN_Base.util import bounds_from_data, random_choice
import numpy as np
import tensorflow as tf


def rmsprop_init_optimizers(self):
    self.optimizer_RMSProp = tf.train.RMSPropOptimizer(
        learning_rate=.001).minimize(self.loss)


def rmsprop_train(self, X, U, X_df, batch_size, epochs):
    self._train_stochastic_optimizer(
        self.optimizer_RMSProp, X, U, X_df, batch_size, epochs)


class Burgers_Base_RMSProp(Burgers):

    def _init_optimizers(self):
        rmsprop_init_optimizers(self)

    def train_RMSProp_batched(self, X, U, X_df, batch_size, epochs):
        rmsprop_train(self, X, U, X_df, batch_size, epochs)


class Helmholtz_Base_RMSProp(Helmholtz):
    def _init_optimizers(self):
        rmsprop_init_optimizers(self)

    def train_RMSProp_batched(self, X, U, X_df, batch_size, epochs):
        rmsprop_train(self, X, U, X_df, batch_size, epochs)


MAX_THREADS = 32
config = tf.ConfigProto(
    intra_op_parallelism_threads=MAX_THREADS
)


def burgers_big_adam():

    X_true, U_true, X_bounds, U_bounds, _ = load_burgers_bounds()

    X = np.vstack(X_bounds)
    U = np.vstack(U_bounds)
    X_df = random_choice(X_true)

    lower_bound, upper_bound = bounds_from_data(X_true)

    layers = [2, 256, 256, 256, 256, 256, 1]
    nu = .01 / np.pi
    model = Burgers(
        lower_bound, upper_bound, layers, nu, session_config=config, use_collocation_residual=False)

    model.train_Adam_batched(X, U, X_df, batch_size=64, epochs=50000)
    U_hat = model.predict(X_true)
    rel_error = np.linalg.norm(U_true - U_hat, 2) / np.linalg.norm(U_true, 2)

    print(rel_error)
    with open("big_log.csv", "a+") as f:
        f.write(f"burgers,adam,{rel_error}\n")


def burgers_big_rmsprop():
    X_true, U_true, X_bounds, U_bounds, _ = load_burgers_bounds()

    X = np.vstack(X_bounds)
    U = np.vstack(U_bounds)
    X_df = random_choice(X_true)

    lower_bound, upper_bound = bounds_from_data(X_true)

    layers = [2, 256, 256, 256, 256, 256, 1]
    nu = .01 / np.pi
    model = Burgers_Base_RMSProp(
        lower_bound, upper_bound, layers, nu, session_config=config, use_collocation_residual=False)

    model.train_RMSProp_batched(X, U, X_df, batch_size=64, epochs=50000)
    U_hat = model.predict(X_true)
    rel_error = np.linalg.norm(U_true - U_hat, 2) / np.linalg.norm(U_true, 2)

    print(rel_error)
    with open("big_log.csv", "a+") as f:
        f.write(f"burgers,rms,{rel_error}\n")


def helmholtz_big_adam():
    X_true, U_true, X_bounds, U_bounds, _ = load_helmholtz_bounds()

    X = np.vstack(X_bounds)
    U = np.vstack(U_bounds)
    X_df = random_choice(X_true)

    lower_bound, upper_bound = bounds_from_data(X_true)

    layers = [2, 256, 256, 256, 256, 256, 1]
    a = 1
    b = 4
    model = Helmholtz(
        lower_bound, upper_bound, layers, a, b, session_config=config, use_collocation_residual=False, df_multiplier=1e-2)

    model.train_Adam_batched(X, U, X_df, batch_size=64, epochs=50000)
    U_hat = model.predict(X_true)
    rel_error = np.linalg.norm(U_true - U_hat, 2) / np.linalg.norm(U_true, 2)

    print(rel_error)
    with open("big_log.csv", "a+") as f:
        f.write(f"helmholtz,adam,{rel_error}\n")


def helmholtz_big_rmsprop():
    X_true, U_true, X_bounds, U_bounds, _ = load_helmholtz_bounds()

    X = np.vstack(X_bounds)
    U = np.vstack(U_bounds)
    X_df = random_choice(X_true)

    lower_bound, upper_bound = bounds_from_data(X_true)

    layers = [2, 256, 256, 256, 256, 256, 1]
    a = 1
    b = 4
    model = Helmholtz_Base_RMSProp(
        lower_bound, upper_bound, layers, a, b, session_config=config, use_collocation_residual=False, df_multiplier=1e-2)

    model.train_RMSProp_batched(X, U, X_df, batch_size=64, epochs=1)
    U_hat = model.predict(X_true)
    rel_error = np.linalg.norm(U_true - U_hat, 2) / np.linalg.norm(U_true, 2)

    print(rel_error)
    with open("big_log.csv", "a+") as f:
        f.write(f"helmholtz,rms,{rel_error}\n")


if __name__ == "__main__":
    if int(sys.argv[1]) == 0:
        burgers_big_adam()
    elif int(sys.argv[1]) == 1:
        burgers_big_rmsprop()
    elif int(sys.argv[1]) == 2:
        helmholtz_big_adam()
    elif int(sys.argv[1]) == 3:
        helmholtz_big_rmsprop()
