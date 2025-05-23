"""Base Optimizer class for Genetic Algorithms selection."""

import asyncio
import json
import logging
import random
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import tiktoken
from deap import base, creator, tools
from pydantic import BaseModel

from prompty.optimize.evals.cost_aware_evaluator import CostAwareEvaluator
from prompty.optimize.evals.dataset_evaluator import DatasetEvaluator
from prompty.optimize.evals.evaluator import Evaluator
from prompty.prompt_components.schemas import (
    PromptComponentCandidates,
    PromptComponents,
    PromptTemplate,
)
from prompty.tracking.mlflow_tracking import MlflowTracker

logger = logging.getLogger(__name__)


class SearchSpace(BaseModel):
    """Search space for prompt optimization."""

    component_candidates: Dict[str, PromptComponentCandidates]
    other_params: Dict[str, Any]


class GAOptimizer:
    """DEAP (Genetic Algorithms)Prompt Optimization base class."""

    def __init__(
        self,
        evaluator: Evaluator,
        search_space: SearchSpace,
        population_size: int = 2,
        n_generations: int = 2,
        crossover_prob: float = 0.8,
        mutation_prob: float = 0.2,
        tournament_size: int = 3,
        elitism_size: int = 2,
        experiment_tracker: Optional[MlflowTracker] = None,
    ):
        """Initialize the GA (DEAP based) optimizer class.

        Args:
            evaluator: Evaluator to use for scoring prompts
            search_space: List of Prompt Component Candidates and other paramters sto optimise across
            population_size: Number of individuals in the population
            n_generations: Number of generations to run
            crossover_prob: Probability of crossover
            mutation_prob: Probability of mutation
            tournament_size: Tournament size for selection
            elitism_size: Number of individuals to keep from previous generation
            experiment_tracker: Optional experiment tracker instance
        """
        self.evaluator = evaluator
        self.search_space = search_space
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.tournament_size = tournament_size
        self.elitism_size = elitism_size
        self.experiment_tracker = experiment_tracker or MlflowTracker()
        self.component_names = list(search_space.component_candidates.keys())
        self.component_options = [
            len(search_space.component_candidates[name].candidates)
            for name in self.component_names
        ]
        self.best_params = None
        self.best_score = None
        self._setup_deap()

        # Cost configuration
        self.trials_costs = []
        self._total_cost = 0.0

    def calculate_total_evaluations(self):
        if self.n_generations <= 0:
            return 0
        return self.n_generations * self.population_size

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

    def _setup_deap(self):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()
        self.toolbox.register(
            "attr_int", lambda max_val: random.randint(0, max_val - 1)
        )
        self.toolbox.register(
            "individual",
            lambda: creator.Individual(
                [self.toolbox.attr_int(max_val) for max_val in self.component_options]
            ),
        )
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register(
            "mutate",
            tools.mutUniformInt,
            low=0,
            up=[x - 1 for x in self.component_options],
            indpb=0.2,
        )
        self.toolbox.register(
            "select", tools.selTournament, tournsize=self.tournament_size
        )

    async def _evaluate_individual(self, individual, idx):
        trial_suggestions_idx = {
            name: val for name, val in zip(self.component_names, individual)
        }
        trial_suggestions_comp = {
            name: self.search_space.component_candidates[name].candidates[val]
            for name, val in trial_suggestions_idx.items()
        }

        prompt_template = PromptTemplate()

        components = PromptComponents(**trial_suggestions_comp)
        prompt = prompt_template.load_template_from_components(components)

        trial_cost = GAOptimizer._get_trial_cost(prompt)
        self._total_cost += trial_cost
        self.trials_costs.append(trial_cost)

        self.experiment_tracker.log_prompt(prompt, f"individual_{idx}")
        score = await self.evaluator.evaluate(prompt)

        self.experiment_tracker.log_params(
            {f"individual_{idx}_{k}": v for k, v in trial_suggestions_comp.items()}
        )
        metrics = {
            "score": score,
            "total_cost": self._total_cost,
            "trials_completed": len(self.trials_costs),
            "trial_cost": self.trials_costs[-1],
            "trial_number": len(self.trials_costs),
        }
        self.experiment_tracker.log_metrics(metrics, step=idx)

        return (score,)

    async def optimize(self):

        with self.experiment_tracker.start_run(
            run_name=f"prompt_optimization_GA_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            tags={
                "population_size": self.population_size,
                "n_generations": self.n_generations,
                "crossover_prob": self.crossover_prob,
                "mutation_prob": self.mutation_prob,
                "tournament_size": self.tournament_size,
                "elitism_size": self.elitism_size,
            },
        ):
            population = self.toolbox.population(n=self.population_size)

            for gen in range(self.n_generations):
                logger.info(f"Generation {gen + 1}/{self.n_generations}")

                # Only evaluate individuals without valid fitness
                invalid_inds = [ind for ind in population if not ind.fitness.valid]
                fitnesses = await asyncio.gather(
                    *[
                        self._evaluate_individual(ind, idx + gen * 1000)
                        for idx, ind in enumerate(invalid_inds)
                    ]
                )
                for ind, fit in zip(invalid_inds, fitnesses):
                    ind.fitness.values = fit

                # Early stopping if all elites are preserved and no offspring possible
                if self.elitism_size == self.population_size:
                    logger.info("All individuals are elites, stopping early.")
                    break

                # Select elite individuals
                elites = tools.selBest(population, self.elitism_size)

                # Generate offspring
                offspring = self.toolbox.select(
                    population, len(population) - self.elitism_size
                )
                offspring = list(map(self.toolbox.clone, offspring))

                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < self.crossover_prob:
                        self.toolbox.mate(child1, child2)
                        del child1.fitness.values, child2.fitness.values

                for mutant in offspring:
                    if random.random() < self.mutation_prob:
                        self.toolbox.mutate(mutant)
                        del mutant.fitness.values

                # Evaluate offspring that were changed
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                new_fitnesses = await asyncio.gather(
                    *[
                        self._evaluate_individual(ind, idx + gen * 1000 + 500)
                        for idx, ind in enumerate(invalid_ind)
                    ]
                )
                for ind, fit in zip(invalid_ind, new_fitnesses):
                    ind.fitness.values = fit

                population = elites + offspring

            # Final best
            best_ind = tools.selBest(population, 1)[0]
            best_components = {
                name: self.search_space.component_candidates[name].candidates[val]
                for name, val in zip(self.component_names, best_ind)
            }
            best_score = best_ind.fitness.values[0]

            self.experiment_tracker.log_optimization_results(
                best_params=best_components,
                best_value=best_score,
                n_trials=self.n_generations * self.population_size,
                study_name="GA_prompt_optimization",
            )

            self.best_params = best_components
            self.best_score = best_score
            return {"best_params": best_components, "best_value": best_score}

    def save_results(self, file_path: str) -> None:
        """Save the optimization results to a JSON file.

        Args:
            file_path: Path to save the results to
        """
        if not self.best_params or not self.best_score:
            raise ValueError("No optimization results to save. Run optimize() first.")

        results = {
            "best_params": self.best_params,
            "best_value": self.best_score,
        }

        with open(file_path, "w") as f:
            json.dump(results, f, indent=2)
