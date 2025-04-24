"""Cost-aware evaluator for prompt optimization."""

from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd
import tiktoken
from jinja2 import DebugUndefined, Environment, Template
from langchain_core.language_models.chat_models import BaseChatModel
from tqdm import tqdm

from prompty.optimize.evals.evaluator import Evaluator

env = Environment(undefined=DebugUndefined)


class CostAwareEvaluator(Evaluator):
    """Evaluator that considers both performance and cost metrics."""

    def __init__(
        self,
        llm_provider: BaseChatModel,
        dataset: Union[str, pd.DataFrame],
        input_column: str,
        target_column: str,
        scoring_function: Callable[[str, str], float] = lambda r, t: float(
            r.content.strip() == t.strip()
        ),
        cost_function: Optional[Callable[[str], float]] = None,
        max_samples: Optional[int] = None,
        cost_weight: float = 0.5,
        performance_weight: float = 0.5,
    ):
        """Initialize the cost-aware evaluator.

        Args:
            llm_provider: LLM provider to use for generating responses
            dataset: Dataset to evaluate on (path to CSV or DataFrame)
            input_column: Column name for inputs
            target_column: Column name for target outputs
            scoring_function: Function that scores a response against the target
            cost_function: Function that calculates cost for a prompt (default: token count)
            max_samples: Maximum number of samples to use (None = use all)
            cost_weight: Weight for cost in the combined objective (0-1)
            performance_weight: Weight for performance in the combined objective (0-1)
        """
        self.llm_provider = llm_provider

        # Load dataset if it's a string (path to CSV)
        if isinstance(dataset, str):
            self.dataset = pd.read_csv(dataset)
        else:
            self.dataset = dataset

        self.input_column = input_column
        self.target_column = target_column
        self.scoring_function = scoring_function
        self.cost_function = cost_function or self._default_cost_function
        self.cost_weight = cost_weight
        self.performance_weight = performance_weight

        # Limit the number of samples if requested
        if max_samples and max_samples < len(self.dataset):
            self.dataset = self.dataset.sample(max_samples, random_state=42)

    def _default_cost_function(
        self,
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

    async def evaluate(self, prompt: str) -> float:
        """Evaluate a prompt considering both performance and cost.

        Args:
            prompt: The prompt to evaluate

        Returns:
            Dictionary containing:
            - combined_score: Weighted combination of performance and cost
            - performance_score: Raw performance score
            - cost_score: Normalized cost score
            - raw_cost: Actual cost value
        """
        performance_scores = []
        total_cost = 0.0

        # Iterate through the dataset
        for _, row in tqdm(
            self.dataset.iterrows(), total=len(self.dataset), desc="Evaluating prompt"
        ):
            # Get input and target from the dataset
            input_text = str(row[self.input_column])
            target = str(row[self.target_column])

            # Format the prompt with the input
            formatted_prompt = env.from_string(prompt).render(text=input_text)

            # Get the model's response
            response = await self.llm_provider.ainvoke(formatted_prompt)

            # Calculate performance score
            performance_score = self.scoring_function(response, target)
            performance_scores.append(performance_score)

            # Calculate cost
            total_cost += self.cost_function(formatted_prompt)

        # Calculate average performance
        avg_performance = (
            sum(performance_scores) / len(performance_scores)
            if performance_scores
            else 0.0
        )

        # Normalize cost (assuming lower is better)
        # You might want to adjust this based on your cost function
        max_expected_cost = 10.0  # Adjust this based on your use case
        normalized_cost = min(total_cost / max_expected_cost, 1.0)

        # Calculate combined score
        combined_score = (
            self.performance_weight * avg_performance
            - self.cost_weight * normalized_cost
        )

        return combined_score

        # return {
        #     "combined_score": combined_score,
        #     "performance_score": avg_performance,
        #     "cost_score": normalized_cost,
        #     "raw_cost": total_cost
        # }

    def set_weights(self, performance_weight: float, cost_weight: float) -> None:
        """Update the weights for the combined objective.

        Args:
            performance_weight: New weight for performance (0-1)
            cost_weight: New weight for cost (0-1)
        """
        if not (0 <= performance_weight <= 1 and 0 <= cost_weight <= 1):
            raise ValueError("Weights must be between 0 and 1")
        if not abs(performance_weight + cost_weight - 1.0) < 1e-6:
            raise ValueError("Weights must sum to 1")

        self.performance_weight = performance_weight
        self.cost_weight = cost_weight
