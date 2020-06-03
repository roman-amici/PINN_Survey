import PINN_Survey.benchmarking.benchmark as benchmark
import PINN_Survey.problems.klein_gordon.v1 as klein_gordon
from PINN_Survey.problems.klein_gordon.data.load import load_klein_gordon_bounds
import numpy as np
import os

problem_desc = {
    "equation": "Klein Gordon",
    "description_file": ""
}

optimizer_desc = {
    "name": "L-BFGS",
}


def klein_gordon_arch_comparison_v1(
        log_file="logs/klein_gordon_arch_comparison_v1.json",
        n_trials=20,
        n_df=10000,
        width=20,
        depth=8):

    path = os.path.dirname(os.path.abspath(__file__))
    file_path = f"{path}/{log_file}"

    X_true, U_true, X_bounds, U_bounds, [x, t, u] = load_klein_gordon_bounds()

    X = np.vstack(X_bounds)
    U = np.vstack(U_bounds)

    idx = np.random.choice(list(range(X_true.shape[0])), size=n_df)
    X_df = X_true[idx, :]

    nu = .01 / np.pi

    lower_bound = np.min(X_true, axis=0)
    upper_bound = np.max(X_true, axis=0)

    layers_base = [2] + ([width]*depth) + [1]

    layers_approx = [2] + ([width]*2) + [1]
    layers_mesh = [2] + ([width]*(depth-2))

    alpha = -1
    beta = 0
    gamma = 1
    k = 3

    model_base = klein_gordon.Klein_Gordon(
        lower_bound, upper_bound, layers_base, alpha, beta, gamma, k)

    benchmark_klein_gordon = benchmark.Benchmark(
        problem_desc, model_base, [X, U, X_df, X_true, U_true], optimizer_desc)

    print("Beginning Base")
    benchmark.log_benchmark(
        benchmark_klein_gordon, n_trials, file_path)

    model_softmesh = klein_gordon.Klein_Gordon_Soft_Mesh(
        lower_bound, upper_bound, layers_approx, layers_mesh, alpha, beta, gamma, k)

    benchmark_klein_gordon_softmesh = benchmark.Benchmark(
        problem_desc, model_softmesh, [X, U, X_df, X_true, U_true], optimizer_desc)

    print("Beginning Soft <esh")
    benchmark.log_benchmark(
        benchmark_klein_gordon_softmesh, n_trials, file_path)

    model_domain_transformer = klein_gordon.Klein_Gordon_Domain_Transformer(
        lower_bound, upper_bound, width, depth-2, alpha, beta, gamma, k)

    print("Beginning Domain Transformer")
    benchmark_klein_gordon_domain_transformer = benchmark.Benchmark(problem_desc, model_domain_transformer, [
        X, U, X_df, X_true, U_true], optimizer_desc)

    benchmark.log_benchmark(
        benchmark_klein_gordon_domain_transformer, n_trials, file_path)


if __name__ == "__main__":
    for depth in [4, 5, 6, 7, 8]:
        klein_gordon_arch_comparison_v1(
            n_trials=25, depth=depth, log_file="logs/Klein_Gordon_arch_comparison_v1.json")
