import PINN_Survey.benchmarking.benchmark as benchmark
import PINN_Survey.problems.poisson_2d_steep.v1 as poisson
from PINN_Survey.problems.poisson_2d_steep.data.load import load_poisson_bounds
import numpy as np
import os

problem_desc = {
    "equation": "Poisson",
    "description_file": ""
}

optimizer_desc = {
    "name": "L-BFGS",
}


def poisson_2d_steep_arch_comparison_v1(
        log_file="logs/poisson_2d_steep_arch_comparison_v1.json",
        n_trials=20,
        n_df=10000,
        width=20,
        depth=8):

    path = os.path.dirname(os.path.abspath(__file__))
    file_path = f"{path}/{log_file}"

    X_true, U_true, X_bounds, U_bounds, [
        x, t, u] = load_poisson_bounds()

    X = np.vstack(X_bounds)
    U = np.vstack(U_bounds)

    idx = np.random.choice(list(range(X_true.shape[0])), size=n_df)
    X_df = X_true[idx, :]

    lower_bound = np.min(X_true, axis=0)
    upper_bound = np.max(X_true, axis=0)

    layers_base = [2] + ([width]*depth) + [1]

    layers_approx = [2] + ([width]*2) + [1]
    layers_mesh = [2] + ([width]*(depth-2))

    model_base = poisson.Poisson(
        lower_bound, upper_bound, layers_base)

    benchmark_poisson_2d_steep = benchmark.Benchmark(
        problem_desc, model_base, [X, U, X_df, X_true, U_true], optimizer_desc)

    print("Beginning Base")
    benchmark.log_benchmark(
        benchmark_poisson_2d_steep, n_trials, file_path)

    model_softmesh = poisson.Poisson_Soft_Mesh(
        lower_bound, upper_bound, layers_approx, layers_mesh)

    benchmark_poisson_2d_steep_softmesh = benchmark.Benchmark(
        problem_desc, model_softmesh, [X, U, X_df, X_true, U_true], optimizer_desc)

    print("Beginning soft mesh")
    benchmark.log_benchmark(
        benchmark_poisson_2d_steep_softmesh, n_trials, file_path)

    model_domain_transformer = poisson.Poisson_Domain_Transformer(
        lower_bound, upper_bound, 2, 1, width, depth-2)

    print("Beginning Domain Transformer")
    benchmark_poisson_2d_steep_domain_transformer = benchmark.Benchmark(problem_desc, model_domain_transformer, [
        X, U, X_df, X_true, U_true], optimizer_desc)

    benchmark.log_benchmark(
        benchmark_poisson_2d_steep_domain_transformer, n_trials, file_path)


def poisson_sphere_mesh_v1(
        log_file="logs/poisson_arch_comparison_v1.json",
        n_trials=20,
        n_df=10000,
        layers_mesh=[2, 20, 20, 20, 20],
        layers_approx=[2, 20, 20, 1]):

    path = os.path.dirname(os.path.abspath(__file__))
    file_path = f"{path}/{log_file}"

    X_true, U_true, X_bounds, U_bounds, [x, t, u] = load_poisson_bounds()

    X = np.vstack(X_bounds)
    U = np.vstack(U_bounds)

    idx = np.random.choice(list(range(X_true.shape[0])), size=n_df)
    X_df = X_true[idx, :]

    lower_bound = np.min(X_true, axis=0)
    upper_bound = np.max(X_true, axis=0)

    model_sphere_mesh = poisson.Poisson_Sphere_Mesh(
        lower_bound, upper_bound, layers_approx, layers_mesh)

    benchmark_poisson_sphere_mesh = benchmark.Benchmark(
        problem_desc, model_sphere_mesh, [X, U, X_df, X_true, U_true], optimizer_desc)

    benchmark.log_benchmark(
        benchmark_poisson_sphere_mesh, n_trials, file_path)


if __name__ == "__main__":
    width = 20
    for depth in [4, 5, 6, 7, 8, 9, 10]:
        layers_approx = [2] + ([width]*2) + [1]
        layers_mesh = [2] + ([width]*(depth-2))
        poisson_sphere_mesh_v1(
            n_trials=25,
            layers_approx=layers_approx,
            layers_mesh=layers_mesh,
            log_file="logs/poisson_2d_steep_arch_comparison_v1.json")
