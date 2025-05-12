import wandb
import os
import json
from datetime import datetime
from contextlib import contextmanager
from typing import Optional, Dict, Any


class WandbTracker:
    """
    Class for tracking experiments using Weights & Biases (W&B).
    Provides methods for logging parameters, metrics, prompts, artifacts, and LLM calls.
    """

    def __init__(
        self,
        project: str = "prompty_experiments",
        entity: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize the W&B experiment tracker.

        Args:
            project: Name of the W&B project
            entity: W&B entity (username or team name)
            tags: Optional dictionary of tags to associate with each run
        """
        self.project = project
        self.entity = entity
        self.tags = tags or {}
        self.run = None

    @contextmanager
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, Any]] = None, config: Optional[Dict[str, Any]] = None):
        """
        Start a new W&B run as a context manager.

        Args:
            run_name: Optional name for the run
            config: Optional configuration dictionary to initialize W&B config

        Yields:
            The W&B run object
        """
        run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run = wandb.init(
            project=self.project,
            entity=self.entity,
            name=run_name,
            config=config or {},
            tags=tags or list(self.tags.values())
        )
        try:
            yield self.run
        finally:
            wandb.finish()

    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log hyperparameters or configuration parameters to the current run.

        Args:
            params: Dictionary of parameter names and values
        """
        wandb.config.update(params, allow_val_change=True)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log metrics to the current run.

        Args:
            metrics: Dictionary of metric names and float values
            step: Optional step number associated with the metrics
        """
        wandb.log(metrics, step=step)

    def log_prompt(self, prompt: str, prompt_name: str = "prompt") -> None:
        """
        Save and log a prompt text as an artifact.

        Args:
            prompt: Prompt text string
            prompt_name: Optional name for the prompt file (without extension)
        """
        file_path = f"{prompt_name}.txt"
        with open(file_path, "w") as f:
            f.write(prompt)

        artifact = wandb.Artifact(name=prompt_name, type="prompt")
        artifact.add_file(file_path)
        self.run.log_artifact(artifact)

        os.remove(file_path)

    def log_llm_call(
        self,
        prompt: str,
        response: str,
        model_name: str,
        temperature: float,
        max_tokens: int,
        cost: Optional[float] = None,
    ) -> None:
        """
        Log the details of an LLM call as a JSON artifact.

        Args:
            prompt: The prompt used for the LLM call
            response: The LLM's response
            model_name: Name of the model used (e.g., gpt-4)
            temperature: Sampling temperature used in the call
            max_tokens: Maximum tokens allowed in the response
            cost: Optional cost of the API call
        """
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

        file_path = f"llm_call_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
        with open(file_path, "w") as f:
            json.dump(call_details, f, indent=2)
        wandb.log({"llm_call": wandb.Artifact(file_path, type="llm_call")})
        os.remove(file_path)

    def log_optimization_results(
        self,
        best_params: Dict[str, Any],
        best_value: float,
        n_trials: int,
        study_name: str,
    ) -> None:
        """
        Log the results of a hyperparameter optimization study.

        Args:
            best_params: Dictionary of best-found parameters
            best_value: Best metric value achieved
            n_trials: Total number of trials in the study
            study_name: Name of the optimization study
        """
        result = {
            "best_params": best_params,
            "best_value": best_value,
            "n_trials": n_trials,
            "study_name": study_name,
            "timestamp": datetime.now().isoformat(),
        }
        wandb.log(result)
