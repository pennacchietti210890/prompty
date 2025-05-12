"""Objective function implementation for prompt optimization."""

import asyncio
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import pandas as pd
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel
import tiktoken

from prompty.optimize.evals.cost_aware_evaluator import CostAwareEvaluator
from prompty.optimize.evals.dataset_evaluator import DatasetEvaluator
from prompty.optimize.evals.evaluator import Evaluator
from prompty.prompt_components.schemas import (PromptComponentCandidates,
                                               PromptComponents,
                                               PromptTemplate)
from prompty.tracking.mlflow_tracking import MlflowTracker

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


class HyperOptOptimizer:
    """Prompt Optimization base class based on hyperopt."""

    def __init__(
        self,
        evaluator: Evaluator,
        search_space: SearchSpace,
        max_evals: int = 10,
        experiment_tracker: Optional[MlflowTracker] = None,
        early_stopping_config: Optional[EarlyStoppingConfig] = None,
    ):
        """Initialize the objective function.

        Args:
            evaluator: Evaluator to use for scoring prompts
            search_space: List of Prompt Component Candidates and other paramters sto optimise across
            max_evals: Maximum number of evaluations to run
            experiment_tracker: Optional experiment tracker instance
            early_stopping_config: Configuration for early stopping mechanisms
        """
        self.evaluator = evaluator
        self.search_space = search_space
        self.max_evals = max_evals
        self.experiment_tracker = experiment_tracker or MlflowTracker()
        self.early_stopping_config = early_stopping_config or EarlyStoppingConfig()

        # Early stopping state
        self._best_score = float("-inf")
        self._no_improvement_count = 0
        self._total_cost = 0.0
        self._scores_history = []
        self.trials_costs = []
        self.best_params = None
        self.best_value = None

        # Build HyperOpt space
        self.space = {}
        for component, candidates in self.search_space.component_candidates.items():
            if not candidates.candidates:
                logger.warning(f"No candidates for component {component}")
                continue
            self.space[component] = hp.choice(component, candidates.candidates)

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
        cfg = self.early_stopping_config
        if current_score > self._best_score + cfg.min_improvement:
            self._best_score = current_score
            self._no_improvement_count = 0
        else:
            self._no_improvement_count += 1

        self._total_cost += self.trials_costs[-1]
        self._scores_history.append(current_score)

        if len(self._scores_history) < cfg.min_trials:
            return False
        if self._no_improvement_count >= cfg.patience:
            return True
        if cfg.max_total_cost and self._total_cost >= cfg.max_total_cost:
            return True
        if len(self._scores_history) >= cfg.max_trials:
            return True
        return False

    async def _objective(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Objective function for hyperopt."""
        # Ensure all required components are present
        required_components = set(PromptComponents.__fields__.keys())
        missing_components = required_components - set(params.keys())
        
        if missing_components:
            logger.warning(f"Missing components: {missing_components}")
            # Use default values for missing components
            for component in missing_components:
                if component in self.search_space.component_candidates:
                    params[component] = self.search_space.component_candidates[component].candidates[0]
                else:
                    raise ValueError(f"Missing required component {component} and no default available")

        try:
            components = PromptComponents(**params)
            prompt = PromptTemplate().load_template_from_components(components)

            trial_cost = self._get_trial_cost(prompt)
            self.trials_costs.append(trial_cost)

            score = await self.evaluator.evaluate(prompt)

            # Log each parameter with trial number in the name
            trial_num = len(self._scores_history)
            for param_name, param_value in params.items():
                self.experiment_tracker.log_params({f"trial_{trial_num}_{param_name}": param_value})

            # Log prompt with trial number
            self.experiment_tracker.log_prompt(prompt, f"trial_{trial_num}")

            # Log metrics with trial number
            metrics = {
                "score": score,
                "total_cost": self._total_cost,
                "trials_completed": trial_num + 1,
                "trial_cost": trial_cost,
                "trial_number": trial_num
            }
            self.experiment_tracker.log_metrics(metrics, step=trial_num)

            return {
                "loss": -score,  # HyperOpt minimizes
                "status": STATUS_OK,
                "score": score,
                "params": params
            }
        except Exception as e:
            logger.error(f"Error in objective function: {str(e)}")
            return {
                "loss": float("inf"),
                "status": STATUS_OK,
                "score": float("-inf"),
                "params": params
            }
    
    def optimize(self):
        """Run hyperopt optimization."""
        trials = Trials()

        async def run_objective(params):
            return await self._objective(params)

        def sync_objective(params):
            result = asyncio.run(run_objective(params))
            return result

        with self.experiment_tracker.start_run(
            run_name="hyperopt_optimization",
            tags={"optimizer": "hyperopt", "early_stopping": "enabled"},
        ):
            self.experiment_tracker.log_params({
                "max_evals": self.max_evals,
                "early_stopping_config": self.early_stopping_config.dict()
            })

            # Hard coded to simulate early stopping - otherwise fmin will run max_evals
            for i in range(self.max_evals):
                prev_len = len(trials)
                fmin(
                    fn=sync_objective,
                    space=self.space,
                    algo=tpe.suggest,
                    max_evals=prev_len + 1,  # Always do just one new trial
                    trials=trials,
                    show_progressbar=True
                )

                result = trials.results[-1]
                score = result["score"]
                if self._should_stop_early(score):
                    break

            best_result = max(trials.results, key=lambda r: r["score"])
            self.best_params = best_result["params"]
            self.best_value = best_result["score"]

            self.experiment_tracker.log_optimization_results(
                best_params=self.best_params,
                best_value=self.best_value,
                n_trials=len(self._scores_history),
                study_name="hyperopt_study"
            )

        return {
            "best_params": self.best_params,
            "best_value": self.best_value,
            "trials_completed": len(self._scores_history),
            "total_cost": self._total_cost
        }

    def save_results(self, file_path: str):
        """Save the optimization results to a JSON file."""
        with open(file_path, "w") as f:
            json.dump({
                "best_params": self.best_params,
                "best_value": self.best_value,
                "trials_completed": len(self._scores_history),
                "total_cost": self._total_cost,
                "early_stopping_config": self.early_stopping_config.dict()
            }, f, indent=2)