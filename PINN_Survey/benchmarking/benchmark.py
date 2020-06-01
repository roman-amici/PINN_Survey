from typing import Dict, Any, List, Optional
import numpy as np
import abc


class Benchmark:

    def __init__(self, problem_desc, model, data, optimizer_desc, **kwargs):
        self.problem_desc = problem_desc
        self.model = model
        [X, U, X_df, X_eval, U_eval] = data
        self.X = X
        self.U = U
        self.X_df = X_df
        self.X_eval = X_eval
        self.U_eval = U_eval
        self.optimizer_desc = optimizer_desc

    def run_benchmark(self, n_trials: int, metrics=["RMSE, RelError"], **kwargs) -> Dict[str, Any]:
        '''
        Runs the benchmark and returns a summary of the results
        as a JSON friendly object
        '''
        trials = np.empty(n_trials)
        for i in range(n_trials):
            print(f"Trial {i}:")
            # TODO: Add variable optimizer support
            self.model.train_BFGS(self.X, self.U, self.X_df, True)
            U_hat = self.model.predict(self.X_eval)

    def _get_problem_desc(self):
        return self.problem_desc

    def _get_optimizer_desc(self):
        return self.optimizer_desc

    def _get_data_desc(self):
        return {
            "mode": "boundary_value",
            "n_boundary": self.X.shape[0],
            "n_interior": self.X_df.shape[0],
            "n_eval": self.X_eval.shape[0].
        }

    def get_description(self) -> Dict[str, Any]:
        '''
        Returns a description of the benchmark performed as,
        a JSON friendly dict.
        '''
        return {
            "problem": self._get_problem_desc(),
            "data": self._get_data_desc(),
            "optimizer": self._get_optimizer_desc(),
            "architecture": self._get_architecture_desc(),
        }

    def _make_run_summary(self, run: np.ndarray, metric="RMSE", **kwargs) -> Dict[str, Any]:
        '''
        Make a summary for one metric for the run of the benchmark
        '''
        return {
            "metric": metric,
            "n_trials": np.len(run),
            "mean": np.mean(run),
            "median": np.median(run),
            "stddev": np.std(run),
            "trials": run.tolist(),
        }

    def get_benchmark_log(self, summaries: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        description = self.get_description(**kwargs)
        return {**description, "metrics": summaries}


def run_benchmark(benchmark: Benchmark, n_trials: int, run_args: Optional[Dict[str, Any]] = {}, log_args: Optional[Dict[str, Any]] = {}) -> Dict[str, Any]:
    summaries = benchmark.run_benchmark(n_trials, **run_args)
    return benchmark.get_benchmark_log(summaries, **log_args)
