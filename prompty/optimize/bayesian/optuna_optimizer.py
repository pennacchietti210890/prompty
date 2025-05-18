"""Objective function implementation for prompt optimization."""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import optuna
import pandas as pd
import tiktoken
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel

from prompty.optimize.evals.cost_aware_evaluator import CostAwareEvaluator
from prompty.optimize.evals.dataset_evaluator import DatasetEvaluator
from prompty.optimize.evals.evaluator import Evaluator
from prompty.prompt_components.schemas import (PromptComponentCandidates,
                                               PromptComponents,
                                               PromptTemplate)
from prompty.tracking.mlflow_tracking import MlflowTracker
from prompty.tracking.wandb_tracking import WandbTracker

logger = logging.getLogger(__name__)


class SearchSpace(BaseModel):
    """Search space for prompt optimization."""

    component_candidates: Dict[str, PromptComponentCandidates]
    other_params: Dict[str, Any]


class EarlyStoppingConfig(BaseModel):
    """Configuration for early stopping mechanisms."""

    min_trials: int = 3
    max_trials: int = 10
    patience: int = 3
    min_improvement: float = 0.01
    cost_per_trial: float = 0.0
    max_total_cost: Optional[float] = None
    min_confidence: float = 0.95


class OptunaOptimizer:
    """Prompt Optimization base class based on Optuna."""

    def __init__(
        self,
        evaluator: Evaluator,
        search_space: SearchSpace,
        n_trials: int = 3,
        timeout: int = 3600,
        study_name: str = "prompt_optimization",
        direction: str = "maximize",
        experiment_tracker: Optional[MlflowTracker | WandbTracker] = None,
        early_stopping_config: Optional[EarlyStoppingConfig] = None,
    ):
        """Initialize the objective function.

        Args:
            evaluator: Evaluator to use for scoring prompts
            search_space: List of Prompt Component Candidates and other paramters sto optimise across
            n_trials: Number of trials to run
            timeout: Timeout for the optimization
            study_name: Name of the study
            direction: Direction of the optimization
            experiment_tracker: Optional experiment tracker instance
            early_stopping_config: Configuration for early stopping mechanisms
        """
        self.evaluator = evaluator
        self.search_space = search_space
        self.n_trials = n_trials
        self.timeout = timeout
        self.study_name = study_name or "prompt_optimization"
        self.direction = direction
        self.study = None
        self.experiment_tracker = (
            experiment_tracker or MlflowTracker()
        )  # if not supplied, default to Mlflow
        self.early_stopping_config = early_stopping_config or EarlyStoppingConfig()

        # Early stopping state
        self._no_improvement_count = 0
        self._best_score = float("-inf") if direction == "maximize" else float("inf")
        self._total_cost = 0.0
        self._scores_history = []
        self.trials_costs = []

    @staticmethod
    def _get_trial_cost(
        prompt: str,
        model_name: str = "gpt-4o-mini",
        cost_per_million_tokens: float = 0.1,
    ) -> float:
        """Default cost function based on token count.

        Args:
            prompt: The prompt to calculate cost for
            model_name: The name of the model to use for cost calculation
            cost_per_million_tokens: The cost per million tokens for the model
        Returns:
            Estimated cost based on token count and cost per million tokens
        """
        # Simple token count as default cost metric
        encoding = tiktoken.encoding_for_model(model_name)
        tokens = encoding.encode(prompt)
        return len(tokens) * cost_per_million_tokens / 1000000

    def _should_stop_early(self, current_score: float) -> bool:
        """Determine if optimization should stop early based on various criteria.

        Args:
            current_score: Score from the current trial

        Returns:
            True if optimization should stop, False otherwise
        """
        config = self.early_stopping_config

        # Update best score and improvement count
        if (
            self.direction == "maximize"
            and current_score > self._best_score + config.min_improvement
        ) or (
            self.direction == "minimize"
            and current_score < self._best_score - config.min_improvement
        ):
            self._best_score = current_score
            self._no_improvement_count = 0
        else:
            self._no_improvement_count += 1

        # Update cost
        self._total_cost += self.trials_costs[-1]
        self._scores_history.append(current_score)

        # Check stopping conditions
        if len(self._scores_history) < config.min_trials:
            return False

        # 1. No improvement for patience trials
        if self._no_improvement_count >= config.patience:
            logger.info(f"Stopping early: No improvement for {config.patience} trials")
            return True

        # 2. Maximum cost reached
        if config.max_total_cost and self._total_cost >= config.max_total_cost:
            logger.info(f"Stopping early: Maximum cost {config.max_total_cost} reached")
            return True

        # 3. Confidence in convergence
        if len(self._scores_history) >= config.min_trials:
            recent_scores = self._scores_history[-config.min_trials :]
            if self._has_converged(recent_scores, config.min_confidence):
                logger.info("Stopping early: Solution has converged")
                return True

        # 4. Maximum trials reached
        if len(self._scores_history) >= config.max_trials:
            logger.info(f"Stopping early: Maximum trials {config.max_trials} reached")
            return True

        return False

    def _has_converged(self, recent_scores: List[float], min_confidence: float) -> bool:
        """Check if the optimization has converged based on recent scores.

        Args:
            recent_scores: List of recent scores
            min_confidence: Minimum confidence required for convergence

        Returns:
            True if converged, False otherwise
        """
        if len(recent_scores) < 3:
            return False

        # Calculate the standard deviation of recent scores
        std_dev = np.std(recent_scores)
        mean_score = np.mean(recent_scores)

        # If standard deviation is small relative to mean, consider converged
        if mean_score != 0 and std_dev / abs(mean_score) < (1 - min_confidence):
            return True

        return False

    async def _objective_wrapper(self, trial: optuna.Trial) -> float:
        """Objective wrapper for optuna."""
        trial_suggestions_idx = {}
        trial_suggestions_comp = {}
        for component in self.search_space.component_candidates:
            trial_suggestions_idx[component] = trial.suggest_int(
                component,
                0,
                len(self.search_space.component_candidates[component].candidates) - 1,
            )
            trial_suggestions_comp[component] = self.search_space.component_candidates[
                component
            ].candidates[trial_suggestions_idx[component]]

        prompt_template = PromptTemplate()
        components = PromptComponents(**trial_suggestions_comp)
        prompt = prompt_template.load_template_from_components(components)

        trial_cost = OptunaOptimizer._get_trial_cost(prompt)
        self.trials_costs.append(trial_cost)
        self._total_cost += trial_cost
        # Log the prompt for this trial
        self.experiment_tracker.log_prompt(prompt, f"trial_{trial.number}")

        # LLM scoring
        score = await self.evaluator.evaluate(prompt)

        # Log the trial parameters with actual component values
        trial_params = {
            f"trial_{trial.number}_{k}": v for k, v in trial_suggestions_comp.items()
        }
        self.experiment_tracker.log_params(trial_params)

        metrics = {
            "score": score,
            "total_cost": self._total_cost,
            "trials_completed": len(self.trials_costs),
            "trial_cost": self.trials_costs[-1],
            "trial_number": len(self.trials_costs),
        }
        self.experiment_tracker.log_metrics(metrics, step=trial.number)
        return score

    async def optimize(self) -> Dict[str, Any]:
        """Run Optuna optimization with early stopping."""
        # Start a new experiment run
        with self.experiment_tracker.start_run(
            run_name=f"{self.study_name}_OPTUNA_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            tags={
                "optimization_direction": self.direction,
                "max_trials": str(self.early_stopping_config.max_trials),
                "early_stopping": "enabled",
            },
        ):
            # Log the search space and early stopping configuration
            config = {
                "n_trials": self.n_trials,
                "timeout": self.timeout,
                "study_name": self.study_name,
                "direction": self.direction,
                "early_stopping_config": self.early_stopping_config.dict(),
            }
            self.experiment_tracker.log_params(config)

            self.study = optuna.create_study(direction=self.direction)

            # Run trials with early stopping
            for trial_num in range(self.early_stopping_config.max_trials):
                trial = self.study.ask()
                result = await self._objective_wrapper(trial)
                self.study.tell(trial, result)

                # Check early stopping conditions
                if self._should_stop_early(result):
                    logger.info(f"Stopping optimization after {trial_num + 1} trials")
                    break

            # Get the best parameters
            best_params = self.study.best_params
            best_value = self.study.best_value

            component_params = {}
            for component, idx in self.study.best_params.items():
                if component in self.search_space.component_candidates:
                    component_params[component] = (
                        self.search_space.component_candidates[component].candidates[
                            idx
                        ]
                    )

            # Log the final results
            self.experiment_tracker.log_optimization_results(
                best_params=component_params,
                best_value=best_value,
                n_trials=len(self._scores_history),
                study_name=self.study_name,
            )

            # Return the results
            return {
                "best_params": best_params,
                "best_value": best_value,
                "trials_completed": len(self._scores_history),
                "total_cost": self._total_cost,
            }

    def save_results(self, file_path: str) -> None:
        """Save the optimization results to a JSON file.

        Args:
            file_path: Path to save the results to
        """
        if not self.study or not hasattr(self.study, "best_params"):
            raise ValueError("No optimization results to save. Run optimize() first.")

        # Convert indices to actual component text
        component_params = {}
        for component, idx in self.study.best_params.items():
            if component in self.search_space.component_candidates:
                component_params[component] = self.search_space.component_candidates[
                    component
                ].candidates[idx]

        results = {
            "best_params": component_params,
            "best_value": self.study.best_value,
            "trials_completed": len(self._scores_history),
            "total_cost": self._total_cost,
            "study_name": self.study_name,
            "direction": self.direction,
            "early_stopping_config": self.early_stopping_config.dict(),
        }

        with open(file_path, "w") as f:
            json.dump(results, f, indent=2)
