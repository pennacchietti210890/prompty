"""Prompt optimizer implementation using Optuna."""

import asyncio
import json
import os
from typing import Dict, List, Optional, Any, Union, Callable

import optuna
from optuna.samplers import TPESampler
from tqdm import tqdm

from ..prompt_templates import PromptTemplate
from .objective import ObjectiveFunction


class PromptOptimizer:
    """Optimizer for prompt templates using Bayesian optimization."""
    
    def __init__(
        self,
        objective: ObjectiveFunction,
        n_trials: int = 30,
        timeout: Optional[int] = None,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        direction: str = "maximize"
    ):
        """Initialize the prompt optimizer.
        
        Args:
            objective: Objective function to optimize
            n_trials: Number of optimization trials
            timeout: Timeout in seconds (None = no timeout)
            study_name: Name for the Optuna study
            storage: Storage for the Optuna study
            direction: "maximize" or "minimize"
        """
        self.objective = objective
        self.n_trials = n_trials
        self.timeout = timeout
        self.study_name = study_name or "prompt_optimization"
        self.storage = storage
        self.direction = direction
        
        # Create the Optuna study
        self.study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            sampler=TPESampler(),
            direction=self.direction,
            load_if_exists=True
        )
    
    async def _objective_wrapper(self, trial: optuna.Trial) -> float:
        """Wrapper for the objective function to work with Optuna.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Score from the objective function
        """
        # Get parameter info
        param_info = self.objective.get_param_info()
        
        # Suggest values for each parameter
        params = {}
        for name, info in param_info.items():
            # For now, we'll treat all parameters as categorical
            # In a real implementation, you'd want more sophisticated parameter types
            default = info.get("default_value", "")
            options = [default]
            
            # Add some variations for optimization
            # This is a simple example - in practice, you'd want better variations
            if default:
                # Add shorter and longer versions of the default
                if len(default) > 3:
                    options.append(default[:len(default)//2])
                
                # Add a more formal variation
                options.append(f"Please {default}")
                
                # Add a more concise variation
                options.append(default.replace("Please ", "").replace("please ", ""))
            
            params[name] = trial.suggest_categorical(name, options)
        
        # Call the objective function with these parameters
        return await self.objective(**params)
    
    async def optimize(self) -> Dict[str, Any]:
        """Run the optimization process.
        
        Returns:
            Dictionary with optimization results
        """
        # Define an async objective function that works with Optuna
        async def objective_async(trial):
            return await self._objective_wrapper(trial)
        
        # Create a synchronous version for Optuna
        def objective_sync(trial):
            return asyncio.run(objective_async(trial))
        
        # Run the optimization
        self.study.optimize(
            objective_sync,
            n_trials=self.n_trials,
            timeout=self.timeout
        )
        
        # Get the best parameters
        best_params = self.study.best_params
        best_value = self.study.best_value
        
        # Return the results
        return {
            "best_params": best_params,
            "best_value": best_value,
            "best_prompt": self.objective.template.fill(**best_params)
        }
    
    def save_results(self, file_path: str) -> None:
        """Save the optimization results to a JSON file.
        
        Args:
            file_path: Path to save the results to
        """
        if not hasattr(self.study, "best_params"):
            raise ValueError("No optimization results to save. Run optimize() first.")
            
        results = {
            "best_params": self.study.best_params,
            "best_value": self.study.best_value,
            "best_prompt": self.objective.template.fill(**self.study.best_params),
            "template": self.objective.template.to_dict(),
            "n_trials": self.n_trials,
            "study_name": self.study_name,
            "direction": self.direction
        }
        
        with open(file_path, "w") as f:
            json.dump(results, f, indent=2)
    
    @classmethod
    def load_from_results(cls, file_path: str, objective: ObjectiveFunction) -> "PromptOptimizer":
        """Load an optimizer from saved results.
        
        Args:
            file_path: Path to load results from
            objective: Objective function to use
            
        Returns:
            A new PromptOptimizer with loaded settings
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Results file not found: {file_path}")
            
        with open(file_path, "r") as f:
            results = json.load(f)
            
        # Create a new optimizer with the same settings
        optimizer = cls(
            objective=objective,
            n_trials=results.get("n_trials", 30),
            study_name=results.get("study_name", "prompt_optimization"),
            direction=results.get("direction", "maximize")
        )
        
        return optimizer 