"""Experiment tracking module for PROMPTy using MLflow."""

import os
import json
import logging
from typing import Dict, Any, Optional, List
import mlflow
from mlflow.entities import Param, Metric, RunTag
from mlflow.tracking import MlflowClient
from datetime import datetime
import asyncio
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class ExperimentTracker:
    """Class for tracking experiments using MLflow."""

    def __init__(
        self,
        experiment_name: str = "prompty_experiments",
        tracking_uri: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        """Initialize the experiment tracker.

        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: URI for MLflow tracking server (default: local file system)
            tags: Dictionary of tags to add to all runs
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
        self.tags = tags or {}
        
        # Set tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Create experiment if it doesn't exist
        try:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        except:
            self.experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    @contextmanager
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        """Start a new MLflow run.

        Args:
            run_name: Name of the run
            tags: Additional tags for this run
        """
        run_tags = {**self.tags, **(tags or {})}
        run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
            tags=run_tags,
        ) as run:
            self.current_run = run
            yield run

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to the current run.

        Args:
            params: Dictionary of parameters to log
        """
        if not hasattr(self, 'current_run'):
            raise RuntimeError("No active run. Use start_run() context manager.")
        
        mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to the current run.

        Args:
            metrics: Dictionary of metrics to log
            step: Step number for the metrics
        """
        if not hasattr(self, 'current_run'):
            raise RuntimeError("No active run. Use start_run() context manager.")
        
        mlflow.log_metrics(metrics, step=step)

    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None) -> None:
        """Log artifacts to the current run.

        Args:
            local_dir: Local directory containing artifacts
            artifact_path: Path within the run's artifact directory
        """
        if not hasattr(self, 'current_run'):
            raise RuntimeError("No active run. Use start_run() context manager.")
        
        mlflow.log_artifacts(local_dir, artifact_path)

    def log_prompt(self, prompt: str, prompt_name: str = "prompt") -> None:
        """Log a prompt as an artifact.

        Args:
            prompt: The prompt text to log
            prompt_name: Name of the prompt file
        """
        if not hasattr(self, 'current_run'):
            raise RuntimeError("No active run. Use start_run() context manager.")
        
        # Create a temporary file for the prompt
        temp_file = f"{prompt_name}.txt"
        with open(temp_file, "w") as f:
            f.write(prompt)
        
        # Log the file as an artifact
        mlflow.log_artifact(temp_file)
        
        # Clean up
        os.remove(temp_file)

    def log_llm_call(
        self,
        prompt: str,
        response: str,
        model_name: str,
        temperature: float,
        max_tokens: int,
        cost: Optional[float] = None,
    ) -> None:
        """Log an LLM call with its details.

        Args:
            prompt: The prompt used
            response: The LLM response
            model_name: Name of the model used
            temperature: Temperature parameter used
            max_tokens: Maximum tokens parameter used
            cost: Cost of the API call if available
        """
        if not hasattr(self, 'current_run'):
            raise RuntimeError("No active run. Use start_run() context manager.")
        
        # Log the call details
        call_details = {
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt": prompt,
            "response": response,
        }
        
        if cost is not None:
            call_details["cost"] = cost
        
        # Create a unique filename for this call
        call_id = f"llm_call_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        call_file = f"{call_id}.json"
        
        # Save and log the call details
        with open(call_file, "w") as f:
            json.dump(call_details, f, indent=2)
        
        mlflow.log_artifact(call_file, "llm_calls")
        os.remove(call_file)

    def log_optimization_results(
        self,
        best_params: Dict[str, Any],
        best_value: float,
        n_trials: int,
        study_name: str,
    ) -> None:
        """Log optimization results.

        Args:
            best_params: Best parameters found
            best_value: Best value achieved
            n_trials: Number of trials run
            study_name: Name of the optimization study
        """
        if not hasattr(self, 'current_run'):
            raise RuntimeError("No active run. Use start_run() context manager.")
        
        # Log the results
        results = {
            "best_params": best_params,
            "best_value": best_value,
            "n_trials": n_trials,
            "study_name": study_name,
            "timestamp": datetime.now().isoformat(),
        }
        
        # Save and log the results
        results_file = "optimization_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        mlflow.log_artifact(results_file)
        os.remove(results_file)

    def get_run(self, run_id: str) -> mlflow.entities.Run:
        """Get a specific run by ID.

        Args:
            run_id: ID of the run to retrieve

        Returns:
            The run object
        """
        client = MlflowClient()
        return client.get_run(run_id)

    def get_all_runs(self, experiment_id: Optional[str] = None) -> List[mlflow.entities.Run]:
        """Get all runs for an experiment.

        Args:
            experiment_id: ID of the experiment (default: current experiment)

        Returns:
            List of run objects
        """
        client = MlflowClient()
        experiment_id = experiment_id or self.experiment_id
        return client.search_runs(experiment_id) 