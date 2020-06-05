import PINN_Survey.benchmarking.benchmark as benchmark
import PINN_Survey.problems.helmholtz.v1 as helmholtz
from PINN_Survey.problems.helmholtz.data.load import load_helmholtz_bounds
import numpy as np
import os

problem_desc = {
    "equation": "Hemholtz",
    "description_file": "data/helmholtz-analytic-stiff-desc"
}

optimizer_desc = {
    "name": "L-BFGS",
}


def helmholtz_arch_comparison_v1(
        log_file="logs/helmholtz_arch_comparison_v1.json",
        n_trials=25,
        n_df=10000,
        width=20,
        depth=8):

    path = os.path.dirname(os.path.abspath(__file__))
    file_path = f"{path}/{log_file}"

    X_true, U_true, X_bounds, U_bounds, [x, t, u] = load_helmholtz_bounds()

    X = np.vstack(X_bounds)
    U = np.vstack(U_bounds)

    idx = np.random.choice(list(range(X_true.shape[0])), size=n_df)
    X_df = X_true[idx, :]

    a = 1
    b = 4

    lower_bound = np.min(X_true, axis=0)
    upper_bound = np.max(X_true, axis=0)

    layers_base = [2] + ([width]*depth) + [1]

    layers_approx = [2] + ([width]*2) + [1]
    layers_mesh = [2] + ([width]*(depth-2))

    model_base = helmholtz.Helmholtz(
        lower_bound, upper_bound, layers_base, a, b)

    benchmark_helmholtz = benchmark.Benchmark(
        problem_desc, model_base, [X, U, X_df, X_true, U_true], optimizer_desc)

    print("Beginning Base")
    benchmark.log_benchmark(
        benchmark_helmholtz, n_trials, file_path)

    model_softmesh = helmholtz.Helmholtz_Soft_Mesh(
        lower_bound, upper_bound, layers_approx, layers_mesh, a, b)

    benchmark_helmholtz_softmesh = benchmark.Benchmark(
        problem_desc, model_softmesh, [X, U, X_df, X_true, U_true], optimizer_desc)

    print("Beginning Soft Mesh")
    benchmark.log_benchmark(
        benchmark_helmholtz_softmesh, n_trials, file_path)

    model_domain_transformer = helmholtz.Helmholtz_Domain_Transformer(
        lower_bound, upper_bound, width, depth-2, a, b)

    print("Beginning Domain Transformer")
    benchmark_helmholtz_domain_transformer = benchmark.Benchmark(problem_desc, model_domain_transformer, [
        X, U, X_df, X_true, U_true], optimizer_desc)

    benchmark.log_benchmark(
        benchmark_helmholtz_domain_transformer, n_trials, file_path)


def helmholtz_sphere_mesh_v1(
        log_file="logs/helmholtz_arch_comparison_v1.json",
        n_trials=20,
        n_df=10000,
        layers_mesh=[2, 20, 20, 20, 20],
        layers_approx=[2, 20, 20, 1]):

    path = os.path.dirname(os.path.abspath(__file__))
    file_path = f"{path}/{log_file}"

    X_true, U_true, X_bounds, U_bounds, [x, t, u] = load_helmholtz_bounds()

    X = np.vstack(X_bounds)
    U = np.vstack(U_bounds)

    idx = np.random.choice(list(range(X_true.shape[0])), size=n_df)
    X_df = X_true[idx, :]

    a = 1
    b = 4

    lower_bound = np.min(X_true, axis=0)
    upper_bound = np.max(X_true, axis=0)

    model_sphere_mesh = helmholtz.Helmholtz_Sphere_Mesh(
        lower_bound, upper_bound, layers_approx, layers_mesh, a, b)

    benchmark_helmholtz_sphere_mesh = benchmark.Benchmark(
        problem_desc, model_sphere_mesh, [X, U, X_df, X_true, U_true], optimizer_desc)

    benchmark.log_benchmark(
        benchmark_helmholtz_sphere_mesh, n_trials, file_path)


if __name__ == "__main__":
    width = 20
    for depth in [4, 5, 6, 7, 8, 9, 10]:
        layers_approx = [2] + ([width]*2) + [1]
        layers_mesh = [2] + ([width]*(depth-2))
        helmholtz_sphere_mesh_v1(
            n_trials=25,
            layers_approx=layers_approx,
            layers_mesh=layers_mesh,
            log_file="logs/helmholtz_arch_comparison_v1.json")
