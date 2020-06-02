import PINN_Survey.benchmarking.benchmark as benchmark
import PINN_Survey.problems.burgers.v1 as burgers
from PINN_Survey.problems.burgers.data.load import load_burgers_bounds
import numpy as np
import os

problem_desc = {
    "equation": "Burgers",
    "description_file": "/data/burgers_shock_desc.txt"
}

optimizer_desc = {
    "name": "L-BFGS",
}


def burgers_benchmark_v1(
        log_file="logs/burgers_v1.json",
        n_trials=20,
        n_df=10000,
        layers=[2, 20, 20, 20, 20, 20, 20, 20, 20, 1]):

    path = os.path.dirname(os.path.abspath(__file__))

    X_true, U_true, X_bounds, U_bounds, [x, t, u] = load_burgers_bounds()

    X = np.vstack(X_bounds)
    U = np.vstack(U_bounds)

    idx = np.random.choice(list(range(X_true.shape[0])), size=n_df)
    X_df = X_true[idx, :]

    lower_bound = np.min(X_true, axis=0)
    upper_bound = np.max(X_true, axis=0)

    layers = layers
    model = burgers.Burgers(lower_bound, upper_bound, layers, 0.01/np.pi)

    benchmark_burgers = benchmark.Benchmark(
        problem_desc, model, [X, U, X_df, X_true, U_true], optimizer_desc)

    benchmark.log_benchmark(benchmark_burgers, n_trials, f"{path}/{log_file}")


if __name__ == "__main__":
    burgers_benchmark_v1(n_trials=20)
