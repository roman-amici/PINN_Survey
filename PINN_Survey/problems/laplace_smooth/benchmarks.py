import PINN_Survey.benchmarking.benchmark as benchmark
import PINN_Survey.problems.laplace_smooth.v1 as laplace
from PINN_Survey.problems.laplace_smooth.data.load import load_laplace_bounds
import numpy as np
import os
import tensorflow as tf
import sys

problem_desc = {
    "equation": "Laplace",
    "description_file": ""
}

optimizer_desc = {
    "name": "L-BFGS",
}

MAX_THREADS = 32


def laplace_smooth_arch_comparison_v1(
        log_file="logs/laplace_smooth_arch_comparison_v1.json",
        n_trials=20,
        n_df=10000,
        width=20,
        depth=8):

    path = os.path.dirname(os.path.abspath(__file__))
    file_path = f"{path}/{log_file}"

    X_true, U_true, X_bounds, U_bounds, _ = load_laplace_bounds()

    X = np.vstack(X_bounds)
    U = np.vstack(U_bounds)

    idx = np.random.choice(list(range(X_true.shape[0])), size=n_df)
    X_df = X_true[idx, :]

    lower_bound = np.min(X_true, axis=0)
    upper_bound = np.max(X_true, axis=0)

    layers_base = [2] + ([width]*depth) + [1]

    layers_approx = [2] + ([width]*2) + [1]
    layers_mesh = [2] + ([width]*(depth-2))

    config = tf.ConfigProto(
        intra_op_parallelism_threads=MAX_THREADS
    )

    model_base = laplace.Laplace(
        lower_bound, upper_bound, layers_base, session_config=config)

    benchmark_laplace_smooth = benchmark.Benchmark(
        problem_desc, model_base, [X, U, X_df, X_true, U_true], optimizer_desc)

    print("Beginning Base")
    benchmark.log_benchmark(
        benchmark_laplace_smooth, n_trials, file_path)

    model_softmesh = laplace.Laplace_Soft_Mesh(
        lower_bound, upper_bound, layers_approx, layers_mesh, session_config=config)

    benchmark_laplace_smooth_softmesh = benchmark.Benchmark(
        problem_desc, model_softmesh, [X, U, X_df, X_true, U_true], optimizer_desc)

    print("Beginning soft mesh")
    benchmark.log_benchmark(
        benchmark_laplace_smooth_softmesh, n_trials, file_path)

    # model_sphere_mesh = laplace.Laplace_Sphere_Mesh(
    #     lower_bound, upper_bound, layers_approx, layers_mesh, session_config=config)

    # benchmark_laplace_smooth_spheremesh = benchmark.Benchmark(
    #     problem_desc, model_sphere_mesh, [X, U, X_df, X_true, U_true], optimizer_desc)

    # print("Beginning sphere mesh")
    # benchmark.log_benchmark(
    #     benchmark_laplace_smooth_spheremesh, n_trials, file_path)

    model_domain_transformer = laplace.Laplace_Domain_Transformer(
        lower_bound, upper_bound, 2, 1, width, depth-2, session_config=config)

    print("Beginning Domain Transformer")
    benchmark_laplace_smooth_domain_transformer = benchmark.Benchmark(problem_desc, model_domain_transformer, [
        X, U, X_df, X_true, U_true], optimizer_desc)

    benchmark.log_benchmark(
        benchmark_laplace_smooth_domain_transformer, n_trials, file_path)


def laplace_sphere_mesh_v1(
        log_file="logs/laplace_arch_comparison_v1.json",
        n_trials=20,
        n_df=10000,
        layers_mesh=[2, 20, 20, 20, 20],
        layers_approx=[2, 20, 20, 1]):

    path = os.path.dirname(os.path.abspath(__file__))
    file_path = f"{path}/{log_file}"

    config = tf.ConfigProto(
        intra_op_parallelism_threads=MAX_THREADS
    )

    X_true, U_true, X_bounds, U_bounds, _ = load_laplace_bounds()

    X = np.vstack(X_bounds)
    U = np.vstack(U_bounds)

    idx = np.random.choice(list(range(X_true.shape[0])), size=n_df)
    X_df = X_true[idx, :]

    lower_bound = np.min(X_true, axis=0)
    upper_bound = np.max(X_true, axis=0)

    model_sphere_mesh = laplace.Laplace_Sphere_Mesh(
        lower_bound, upper_bound, layers_approx, layers_mesh, session_config=config)

    benchmark_laplace_sphere_mesh = benchmark.Benchmark(
        problem_desc, model_sphere_mesh, [X, U, X_df, X_true, U_true], optimizer_desc)

    benchmark.log_benchmark(
        benchmark_laplace_sphere_mesh, n_trials, file_path)


def laplace_sample_size_scaling(
        log_file="logs/laplace_sample_size_v1.json",
        n_trials=20,
        n_df=[500, 1000, 2500, 5000, 7500, 10000],
        width=20,
        depth=8):

    path = os.path.dirname(os.path.abspath(__file__))
    file_path = f"{path}/{log_file}"

    X_true, U_true, X_bounds, U_bounds, _ = load_laplace_bounds()

    X = np.vstack(X_bounds)
    U = np.vstack(U_bounds)

    lower_bound = np.min(X_true, axis=0)
    upper_bound = np.max(X_true, axis=0)

    layers_base = [2] + ([width]*depth) + [1]

    layers_approx = [2] + ([width]*2) + [1]
    layers_mesh = [2] + ([width]*(depth-2))

    config = tf.ConfigProto(
        intra_op_parallelism_threads=MAX_THREADS
    )

    for n in n_df:
        idx = np.random.choice(list(range(X_true.shape[0])), size=n)
        X_df = X_true[idx, :]

        model_base = laplace.Laplace(
            lower_bound, upper_bound, layers_base, session_config=config)

        benchmark_laplace_smooth = benchmark.Benchmark(
            problem_desc, model_base, [X, U, X_df, X_true, U_true], optimizer_desc)

        print("Beginning Base")
        benchmark.log_benchmark(
            benchmark_laplace_smooth, n_trials, file_path)

        model_sphere_mesh = laplace.Laplace_Sphere_Mesh(
            lower_bound, upper_bound, layers_approx, layers_mesh, session_config=config)

        benchmark_laplace_smooth_spheremesh = benchmark.Benchmark(
            problem_desc, model_sphere_mesh, [X, U, X_df, X_true, U_true], optimizer_desc)

        print("Beginning sphere mesh")
        benchmark.log_benchmark(
            benchmark_laplace_smooth_spheremesh, n_trials, file_path)


def laplace_width_depth_scaling(
        log_file="logs/laplace_width_depth.json",
        n_trials=20,
        n_df=10000,
        widths=[20, 30, 50, 100],
        depths=[4, 6, 8]):

    path = os.path.dirname(os.path.abspath(__file__))
    file_path = f"{path}/{log_file}"

    X_true, U_true, X_bounds, U_bounds, _ = load_laplace_bounds()

    idx = np.random.choice(list(range(X_true.shape[0])), size=n_df)
    X_df = X_true[idx, :]

    X = np.vstack(X_bounds)
    U = np.vstack(U_bounds)

    lower_bound = np.min(X_true, axis=0)
    upper_bound = np.max(X_true, axis=0)

    config = tf.ConfigProto(
        intra_op_parallelism_threads=MAX_THREADS
    )

    for width in widths:
        for depth in depths:

            layers_base = [2] + ([width]*depth) + [1]

            layers_approx = [2] + ([width]*2) + [1]
            layers_mesh = [2] + ([width]*(depth-2))

            model_base = laplace.Laplace(
                lower_bound, upper_bound, layers_base, session_config=config)

            benchmark_laplace_smooth = benchmark.Benchmark(
                problem_desc, model_base, [X, U, X_df, X_true, U_true], optimizer_desc)

            print("Beginning Base")
            benchmark.log_benchmark(
                benchmark_laplace_smooth, n_trials, file_path)

            model_sphere_mesh = laplace.Laplace_Sphere_Mesh(
                lower_bound, upper_bound, layers_approx, layers_mesh, session_config=config)

            benchmark_laplace_smooth_spheremesh = benchmark.Benchmark(
                problem_desc, model_sphere_mesh, [X, U, X_df, X_true, U_true], optimizer_desc)

            print("Beginning sphere mesh")
            benchmark.log_benchmark(
                benchmark_laplace_smooth_spheremesh, n_trials, file_path)


def laplace_sphere_net_v1(
        log_file="logs/laplace_arch_comparison_v1.json",
        n_trials=20,
        n_df=10000,
        layers=[2, 20, 20, 20, 20, 1]):

    path = os.path.dirname(os.path.abspath(__file__))
    file_path = f"{path}/{log_file}"

    config = tf.ConfigProto(
        intra_op_parallelism_threads=MAX_THREADS
    )

    X_true, U_true, X_bounds, U_bounds, _ = load_laplace_bounds()

    X = np.vstack(X_bounds)
    U = np.vstack(U_bounds)

    idx = np.random.choice(list(range(X_true.shape[0])), size=n_df)
    X_df = X_true[idx, :]

    lower_bound = np.min(X_true, axis=0)
    upper_bound = np.max(X_true, axis=0)

    model_sphere_net = laplace.Laplace_Sphere_Net(
        lower_bound, upper_bound, layers, session_config=config)

    benchmark_laplace_sphere_net = benchmark.Benchmark(
        problem_desc, model_sphere_net, [X, U, X_df, X_true, U_true], optimizer_desc)

    benchmark.log_benchmark(
        benchmark_laplace_sphere_net, n_trials, file_path)


if __name__ == "__main__":
    for depth in [4, 5, 6, 7, 8, 9, 10]:
        inner = [20] * depth
        layers = [2] + inner + [1]
        laplace_sphere_net_v1(layers=layers)
