from typing import Dict, Any, List, Optional
import numpy as np
import abc
import subprocess
import json
import os

# TODO: Attribution from github


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


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

    def run_benchmark(self, n_trials: int, metrics=["RMSE", "RelError"], **kwargs) -> Dict[str, Any]:
        '''
        Runs the benchmark and returns a summary of the results
        as a list of (JSON friendly) dictionary objects
        '''
        trials = np.empty((n_trials, len(metrics)))
        for i in range(n_trials):
            print(f"Trial {i+1}:")
            # TODO: Add variable optimizer support
            self.model.train_BFGS(self.X, self.U, self.X_df, True)
            U_hat = self.model.predict(self.X_eval)

            for j, metric in enumerate(metrics):
                trials[i, j] = self.evaluate_metrics(
                    self.U_eval, U_hat, metric)

            self.model.reset_session()
            print("")

        summaries = []
        for j, metric in enumerate(metrics):
            summaries.append(self._get_run_summary(
                trials[:, j], metric), **kwargs)

        return summaries

    def _get_problem_desc(self):
        return self.problem_desc

    def _get_optimizer_desc(self):
        return self.optimizer_desc

    def _get_data_desc(self):
        return {
            "mode": "boundary_value",
            "n_boundary": self.X.shape[0],
            "n_interior": self.X_df.shape[0],
            "n_eval": self.X_eval.shape[0],
        }

    def evaluate_metrics(self, U_eval, U_hat, metric):
        if metric == "RMSE":
            return np.sqrt((np.mean((U_hat[:, 0] - U_eval[:, 0])**2)))
        elif metric == "RelError":
            return np.linalg.norm(U_eval-U_hat, 2)/np.linalg.norm(U_eval, 2)
        else:
            print(f"Metric {metric} not implemented")
            exit()

    def _get_architecture_desc(self):
        return self.model.get_architecture_description()

    def get_description(self) -> Dict[str, Any]:
        '''
        Returns a description of the benchmark performed as,
        a JSON friendly dict.
        '''

        git_commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"]).strip().decode("utf-8")
        return {
            "commit": git_commit,
            "problem": self._get_problem_desc(),
            "data": self._get_data_desc(),
            "optimizer": self._get_optimizer_desc(),
            "architecture": self._get_architecture_desc()
        }

    def _get_run_summary(self, run: np.ndarray, metric="RMSE", **kwargs) -> Dict[str, Any]:
        '''
        Make a summary for one metric for the run of the benchmark
        '''
        return {
            "metric": metric,
            "n_trials": run.shape[0],
            "mean": np.mean(run),
            "median": np.median(run),
            "stddev": np.std(run),
            "min": np.min(run),
            "max": np.max(run),
            "trials": run.tolist(),
        }

    def get_benchmark_log(self, summaries: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        description = self.get_description(**kwargs)
        return {**description, "metrics": summaries}


def run_benchmark(benchmark: Benchmark, n_trials: int, run_args: Optional[Dict[str, Any]] = {}, log_args: Optional[Dict[str, Any]] = {}) -> Dict[str, Any]:
    summaries = benchmark.run_benchmark(n_trials, **run_args)
    return benchmark.get_benchmark_log(summaries, **log_args)


def log_benchmark(
        benchmark: Benchmark,
        n_trials: int,
        log_file: str,
        append=True,
        run_args: Optional[Dict[str, Any]] = {},
        log_args: Optional[Dict[str, Any]] = {}):

    summaries = benchmark.run_benchmark(n_trials, **run_args)
    log = benchmark.get_benchmark_log(summaries, **log_args)

    if os.path.exists(log_file) and append:
        # TODO: Find a more efficient way to do this if logs get too big
        with open(log_file, "r") as f:
            logs = json.load(f)

        logs.append(log)
    else:
        logs = [log]

    with open(log_file, "w+") as f:
        # Avoid throwing errors on numpy types
        json.dump(logs, f, cls=NpEncoder)
