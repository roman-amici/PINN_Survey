import sys
sys.path.append('../')
from PINN_Survey.problems.burgers.data.load import load_burgers_bounds
from PINN_Survey.problems.helmholtz.data.load import load_helmholtz_bounds
from PINN_Base.util import bounds_from_data, random_choice, random_choices, percent_noise
from PINN_Survey.problems.burgers.v1 import Burgers
from PINN_Survey.problems.helmholtz.v1 import Helmholtz
from PINN_Survey.viz.loss_viz import viz_base, viz_2d, save_as_heightmap, make_contour_svd
from PINN_Survey.viz.hessian_viz import viz_hessian_eigenvalue_ratio, Normalization
import numpy as np
import matplotlib.pyplot as plt
import pickle
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


def quick_setup_df(loader, size=10000):

    X_true, U_true, X_boundary, U_boundary, _ = loader()

    X = np.vstack(X_boundary)
    U = np.vstack(U_boundary)

    X_df = random_choice(X_true, size=size)

    lower_bound, upper_bound = bounds_from_data(X_true)

    return X_true, U_true, X, U, X_df, lower_bound, upper_bound


def quick_setup(loader, size=1000):

    X_true, U_true, _, _, _ = loader()

    [X, U] = random_choices(X_true, U_true, size=size)

    lower_bound, upper_bound = bounds_from_data(X_true)

    return X_true, U_true, X, U, lower_bound, upper_bound


def Burgers_bounds(df_multiplier):
    X_true, U_true, X, U, X_df, lower_bound, upper_bound = quick_setup_df(
        load_burgers_bounds)

    layers = [2, 20, 20, 20, 20, 20, 20, 1]
    nu = 0.01 / np.pi
    model = Burgers(lower_bound, upper_bound, layers,
                    nu, use_collocation_residual=False, add_grad_ops=True, session_config=config,
                    df_multiplier=df_multiplier)

    model.train_BFGS(X, U, X_df, True)
    w0 = model.get_all_weights()

    model_viz = Burgers_Viz(lower_bound, upper_bound, layers,
                            nu, use_collocation_residual=False, add_grad_ops=True, session_config=config,
                            df_multiplier=df_multiplier)

    t1s_hess, t2s_hess, ratios, (d1, d2), = viz_hessian_eigenvalue_ratio(
        model_viz, X, U, X_df, w0, 100, 50)
    t1s_loss, t2s_loss, loss_vals = viz_2d(
        model_viz, X, U, X_df, w0, d1, d2, 250)

    name = f"Burgers_Bounds_l{df_multiplier}"

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Burgers Boundary (l={df_multiplier})")

    pallet = axes[1].contourf(
        t1s_hess, t2s_hess, np.log(np.abs(ratios)), levels=30)
    axes[1].set_title("$log(|\\lambda_{min} / \\lambda_{max}|)$")
    fig.colorbar(pallet, ax=axes[1])

    pallet = axes[0].contour(t1s_loss, t2s_loss, np.log(loss_vals), levels=30)
    axes[0].set_title("log(loss)")
    fig.savefig(f"{name}_hessians.png")

    U_hat = model.predict(X_true)
    error = np.sqrt(np.mean((U_true[:, 0] - U_hat[:, 0])**2))

    data = {
        "t1s_hess": t1s_hess, "t2s_hess": t2s_hess, "ratios": ratios,
        "t1s_loss": t1s_loss, "t2s_loss": t2s_loss, "loss_vals": loss_vals,
        "rmse": error
    }

    with open(f"{name}_data.pkl", "wb+") as f:
        pickle.dump(data, f)


def Burgers_regularization(data_noise, df_multiplier):
    X_true, U_true, X, U, lower_bound, upper_bound = quick_setup(
        load_burgers_bounds)

    if data_noise != 0:
        U = percent_noise(U, data_noise)

    layers = [2, 20, 20, 20, 20, 20, 20, 1]
    nu = 0.01 / np.pi
    model = Burgers(lower_bound, upper_bound, layers,
                    nu, use_differential_points=False, add_grad_ops=True, session_config=config,
                    df_multiplier=df_multiplier)

    model.train_BFGS(X, U, None, True)
    w0 = model.get_all_weights()

    model_viz = Burgers_Viz(lower_bound, upper_bound, layers,
                            nu, use_differential_points=False, add_grad_ops=True, session_config=config,
                            df_multiplier=df_multiplier)

    t1s_hess, t2s_hess, ratios, (d1, d2), = viz_hessian_eigenvalue_ratio(
        model_viz, X, U, None, w0, 100, 50)
    t1s_loss, t2s_loss, loss_vals = viz_2d(
        model_viz, X, U, None, w0, d1, d2, 250)

    name = f"Burgers_Regularization_l{df_multiplier}_n{data_noise}"

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Burgers Noisy Sensor (l={df_multiplier})")

    pallet = axes[1].contourf(
        t1s_hess, t2s_hess, np.log(np.abs(ratios)), levels=30)
    axes[1].set_title("$log(|\\lambda_{min} / \\lambda_{max}|)$")
    fig.colorbar(pallet, ax=axes[1])

    pallet = axes[0].contour(t1s_loss, t2s_loss, np.log(loss_vals), levels=30)
    axes[0].set_title("log(loss)")
    fig.savefig(f"{name}_hessians.png")

    U_hat = model.predict(X_true)
    error = np.sqrt(np.mean((U_true[:, 0] - U_hat[:, 0])**2))

    data = {
        "t1s_hess": t1s_hess, "t2s_hess": t2s_hess, "ratios": ratios,
        "t1s_loss": t1s_loss, "t2s_loss": t2s_loss, "loss_vals": loss_vals,
        "rmse": error
    }

    with open(f"{name}_data.pkl", "wb+") as f:
        pickle.dump(data, f)


def Helmholtz_bounds(df_multiplier):
    X_true, U_true, X, U, X_df, lower_bound, upper_bound = quick_setup_df(
        load_helmholtz_bounds)

    layers = [2, 20, 20, 20, 20, 20, 20, 1]
    a = 1.0
    b = 4.0
    model = Helmholtz(lower_bound, upper_bound, layers,
                      a, b, use_collocation_residual=False, add_grad_ops=True, session_config=config,
                      df_multiplier=df_multiplier)

    model.train_BFGS(X, U, X_df, True)
    w0 = model.get_all_weights()

    model_viz = Helmholtz_Viz(lower_bound, upper_bound, layers,
                              a, b, use_collocation_residual=False, add_grad_ops=True, session_config=config,
                              df_multiplier=df_multiplier)

    t1s_hess, t2s_hess, ratios, (d1, d2), = viz_hessian_eigenvalue_ratio(
        model_viz, X, U, X_df, w0, 100, 50)
    t1s_loss, t2s_loss, loss_vals = viz_2d(
        model_viz, X, U, X_df, w0, d1, d2, 250)

    name = f"Helmholtz_Bounds_l{df_multiplier}"

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Helmholtz Boundary (l={df_multiplier})")

    pallet = axes[1].contourf(
        t1s_hess, t2s_hess, np.log(np.abs(ratios)), levels=30)
    axes[1].set_title("$log(|\\lambda_{min} / \\lambda_{max}|)$")
    fig.colorbar(pallet, ax=axes[1])

    pallet = axes[0].contour(t1s_loss, t2s_loss, np.log(loss_vals), levels=30)
    axes[0].set_title("log(loss)")
    fig.savefig(f"{name}_hessians.png")

    U_hat = model.predict(X_true)
    error = np.sqrt(np.mean((U_true[:, 0] - U_hat[:, 0])**2))

    data = {
        "t1s_hess": t1s_hess, "t2s_hess": t2s_hess, "ratios": ratios,
        "t1s_loss": t1s_loss, "t2s_loss": t2s_loss, "loss_vals": loss_vals,
        "rmse": error
    }

    with open(f"{name}_data.pkl", "wb+") as f:
        pickle.dump(data, f)


def Helmholtz_regularization(data_noise, df_multiplier):
    X_true, U_true, X, U, lower_bound, upper_bound = quick_setup(
        load_helmholtz_bounds)

    if data_noise != 0:
        U = percent_noise(U, data_noise)

    layers = [2, 20, 20, 20, 20, 20, 20, 1]
    a = 1.0
    b = 4.0
    model = Helmholtz(lower_bound, upper_bound, layers,
                      a, b, use_differential_points=False, add_grad_ops=True, session_config=config,
                      df_multiplier=df_multiplier)

    model.train_BFGS(X, U, None, True)
    w0 = model.get_all_weights()

    model_viz = Helmholtz_Viz(lower_bound, upper_bound, layers,
                              a, b, use_differential_points=False, add_grad_ops=True, session_config=config,
                              df_multiplier=df_multiplier)

    t1s_hess, t2s_hess, ratios, (d1, d2), = viz_hessian_eigenvalue_ratio(
        model_viz, X, U, None, w0, 100, 50)
    t1s_loss, t2s_loss, loss_vals = viz_2d(
        model_viz, X, U, None, w0, d1, d2, 250)

    name = f"Helmholtz_Regularization_l{df_multiplier}_n{data_noise}"

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Helmholtz Noisy Sensor (l={df_multiplier})")

    pallet = axes[1].contourf(
        t1s_hess, t2s_hess, np.log(np.abs(ratios)), levels=30)
    axes[1].set_title("$log(|\\lambda_{min} / \\lambda_{max}|)$")
    fig.colorbar(pallet, ax=axes[1])

    axes[0].contour(t1s_loss, t2s_loss, np.log(loss_vals), levels=30)
    axes[0].set_title("log(loss)")
    fig.savefig(f"{name}_hessians.png")

    U_hat = model.predict(X_true)
    error = np.sqrt(np.mean((U_true[:, 0] - U_hat[:, 0])**2))

    data = {
        "t1s_hess": t1s_hess, "t2s_hess": t2s_hess, "ratios": ratios,
        "t1s_loss": t1s_loss, "t2s_loss": t2s_loss, "loss_vals": loss_vals,
        "rmse": error
    }

    with open(f"{name}_data.pkl", "wb+") as f:
        pickle.dump(data, f)


def Burgers_bounds_path(df_multiplier):
    X_true, U_true, X, U, X_df, lower_bound, upper_bound = quick_setup_df(
        load_burgers_bounds)

    layers = [2, 20, 20, 20, 20, 20, 20, 1]
    nu = 0.01 / np.pi
    model = Burgers(lower_bound, upper_bound, layers,
                    nu, use_collocation_residual=False, add_grad_ops=True, session_config=config,
                    df_multiplier=df_multiplier)

    fetches = [model.get_all_weight_variables()]
    vals = model.train_BFGS(X, U, X_df, False, fetches)
    vals = [v[0] for v in vals]
    w0 = model.get_all_weights()

    model_viz = Burgers_Viz(lower_bound, upper_bound, layers,
                            nu, use_collocation_residual=False, add_grad_ops=True, session_config=config,
                            df_multiplier=df_multiplier)

    t1s_hess, t2s_hess, ratios, (d1, d2), (points, singular_values) = viz_hessian_eigenvalue_ratio(
        model_viz, X, U, X_df, w0, 100, 50, Normalization.svd, epoch_weights=vals)

    min_vals = np.min(points, axis=0) - 1.0
    max_vals = np.max(points, axis=0) + 1.0

    t1s_loss, t2s_loss, loss_vals = viz_2d(
        model_viz, X, U, X_df, w0, d1, d2, 250, min_vals=min_vals, max_vals=max_vals)

    name = f"Burgers_Bounds_path_l{df_multiplier}"

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Burgers Boundary (l={df_multiplier})")

    pallet = axes[1].contourf(
        t1s_hess, t2s_hess, np.log(np.abs(ratios)), levels=30)
    axes[1].set_title("$log(|\\lambda_{min} / \\lambda_{max}|)$")
    fig.colorbar(pallet, ax=axes[1])

    make_contour_svd(axes[0], t1s_loss, t2s_loss,
                     loss_vals, points, singular_values)
    axes[0].set_title("log(loss)")
    fig.savefig(f"{name}_hessians.png")

    U_hat = model.predict(X_true)
    error = np.sqrt(np.mean((U_true[:, 0] - U_hat[:, 0])**2))

    data = {
        "t1s_hess": t1s_hess, "t2s_hess": t2s_hess, "ratios": ratios,
        "t1s_loss": t1s_loss, "t2s_loss": t2s_loss, "loss_vals": loss_vals,
        "rmse": error
    }

    with open(f"{name}_data.pkl", "wb+") as f:
        pickle.dump(data, f)


def Burgers_Regularization_path(data_noise, df_multiplier):
    X_true, U_true, X, U, lower_bound, upper_bound = quick_setup(
        load_burgers_bounds)

    if data_noise != 0:
        U = percent_noise(U, data_noise)

    layers = [2, 20, 20, 20, 20, 20, 20, 1]
    nu = 0.01 / np.pi
    model = Burgers(lower_bound, upper_bound, layers,
                    nu, use_differential_points=False, add_grad_ops=True, session_config=config,
                    df_multiplier=df_multiplier)

    fetches = [model.get_all_weight_variables()]
    vals = model.train_BFGS(X, U, None, False, fetches)
    vals = [v[0] for v in vals]
    w0 = model.get_all_weights()

    model_viz = Burgers_Viz(lower_bound, upper_bound, layers,
                            nu, use_differential_points=False, add_grad_ops=True, session_config=config,
                            df_multiplier=df_multiplier)

    t1s_hess, t2s_hess, ratios, (d1, d2), (points, singular_values) = viz_hessian_eigenvalue_ratio(
        model_viz, X, U, None, w0, 100, 50, Normalization.svd, epoch_weights=vals)

    min_vals = np.min(points, axis=0) - 1.0
    max_vals = np.max(points, axis=0) + 1.0

    t1s_loss, t2s_loss, loss_vals = viz_2d(
        model_viz, X, U, None, w0, d1, d2, 250, min_vals=min_vals, max_vals=max_vals)

    name = f"Burgers_Regularization_path_l{df_multiplier}"

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Burgers Noisy Sensor (l={df_multiplier})")

    pallet = axes[1].contourf(
        t1s_hess, t2s_hess, np.log(np.abs(ratios)), levels=30)
    axes[1].set_title("$log(|\\lambda_{min} / \\lambda_{max}|)$")
    fig.colorbar(pallet, ax=axes[1])

    make_contour_svd(axes[0], t1s_loss, t2s_loss,
                     loss_vals, points, singular_values)
    axes[0].set_title("log(loss)")
    fig.savefig(f"{name}_hessians.png")

    U_hat = model.predict(X_true)
    error = np.sqrt(np.mean((U_true[:, 0] - U_hat[:, 0])**2))

    data = {
        "t1s_hess": t1s_hess, "t2s_hess": t2s_hess, "ratios": ratios,
        "t1s_loss": t1s_loss, "t2s_loss": t2s_loss, "loss_vals": loss_vals,
        "rmse": error
    }

    with open(f"{name}_data.pkl", "wb+") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    a = int(sys.argv[1])

    if a == 0:
        Burgers_bounds(1.0)
        Burgers_bounds(.1)
        Burgers_bounds(.01)
    if a == 1:
        Helmholtz_bounds(1.0)
        Helmholtz_bounds(1e-1)
        Helmholtz_bounds(1e-2)
    if a == 2:
        Burgers_regularization(.1, 0)
        Burgers_regularization(.1, 1e-2)
        Burgers_regularization(.1, 1e-1)
        Burgers_regularization(.1, 1.0)
    if a == 3:
        Helmholtz_regularization(.1, 0)
        Helmholtz_regularization(.1, 1e-2)
        Helmholtz_regularization(.1, 1e-1)
        Helmholtz_regularization(.1, 1.0)
    if a == 4:
        Burgers_bounds_path(1.0)
    if a == 5:
        Burgers_Regularization_path(.1, 0)
