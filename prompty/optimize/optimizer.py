"""Objective function implementation for prompt optimization."""

from typing import Dict, List, Optional, Any, Union, Callable
import optuna
import logging
from pydantic import BaseModel
from prompty.optimize.evaluator import Evaluator
from prompty.prompt_components.schemas import PromptTemplate, PromptComponentCandidates
from langchain_core.language_models.chat_models import BaseChatModel
import pandas as pd
import json
import asyncio

logger = logging.getLogger(__name__)

class SearchSpace(BaseModel):
    """Search space for prompt optimization."""

    component_candidates: Dict[str, PromptComponentCandidates]
    other_params: Dict[str, Any]


class Optimizer:
    """Prompt Optimization base class."""

    def __init__(
        self,
        evaluator: Evaluator,
        search_space: SearchSpace,
        n_trials: int = 10,
        timeout: int = 3600,
        study_name: str = "prompt_optimization",
        direction: str = "maximize",
    ):
        """Initialize the objective function.

        Args:
            evaluator: Evaluator to use for scoring prompts
            search_space: List of Prompt Component Candidates and other paramters sto optimise across
            n_trials: Number of trials to run
            timeout: Timeout for the optimization
            study_name: Name of the study
            storage: Storage for the study
            direction: Direction of the optimization
        """
        self.evaluator = evaluator
        self.search_space = search_space
        self.n_trials = n_trials
        self.timeout = timeout
        self.study_name = study_name or "prompt_optimization"
        self.direction = direction
        self.study = None

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

        prompt = ""
        for component in trial_suggestions_comp:
            prompt += f"{trial_suggestions_comp[component]}\n\n"

        # LLM scoring
        score = await self.evaluator.evaluate(prompt)
        
        return score


    async def optimize(self) -> float:
        """Run Optuna optimization."""
        self.study = optuna.create_study(direction=self.direction)
        for _ in range(self.n_trials):
            trial = self.study.ask()
            result = await self._objective_wrapper(trial)
            self.study.tell(trial, result)

        # Get the best parameters
        best_params = self.study.best_params
        best_value = self.study.best_value

        # Return the results
        return {
            "best_params": best_params,
            "best_value": best_value,
        }

    def save_results(self, file_path: str) -> None:
        """Save the optimization results to a JSON file.

        Args:
            file_path: Path to save the results to
        """
        if not self.study or not hasattr(self.study, "best_params"):
            raise ValueError("No optimization results to save. Run optimize() first.")

        results = {
            "best_params": self.study.best_params,
            "best_value": self.study.best_value,
            "n_trials": self.n_trials,
            "study_name": self.study_name,
            "direction": self.direction,
        }

        with open(file_path, "w") as f:
            json.dump(results, f, indent=2)
