"""Evaluator classes for prompt optimization."""

import json
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Callable

import pandas as pd
from tqdm import tqdm

from langchain_core.language_models.chat_models import BaseChatModel
from jinja2 import Environment, Template, DebugUndefined

env = Environment(undefined=DebugUndefined)

class Evaluator(ABC):
    """Base abstract class for prompt evaluators."""

    @abstractmethod
    async def evaluate(self, prompt: str) -> float:
        """Evaluate a prompt and return a score.

        Args:
            prompt: The prompt to evaluate

        Returns:
            A score (higher is better)
        """
        pass


class DatasetEvaluator(Evaluator):
    """Evaluator that uses a dataset to evaluate prompts."""

    def __init__(
        self,
        llm_provider: BaseChatModel,
        dataset: Union[str, pd.DataFrame],
        input_column: str,
        target_column: str,
        scoring_function: Callable[[str, str], float] = lambda r, t: float(
            r.content.strip() == t.strip()
        ),
        max_samples: Optional[int] = None,
    ):
        """Initialize the dataset evaluator.

        Args:
            llm_provider: LLM provider to use for generating responses, of BaseChatModel type from langchain_core
            dataset: Dataset to evaluate on (path to CSV or DataFrame)
            input_column: Column name for inputs
            target_column: Column name for target outputs
            scoring_function: Function that scores a response against the target
            max_samples: Maximum number of samples to use (None = use all)
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

        # Limit the number of samples if requested
        if max_samples and max_samples < len(self.dataset):
            self.dataset = self.dataset.sample(max_samples, random_state=42)

    async def evaluate(self, prompt: str) -> float:
        """Evaluate a prompt on the dataset.

        Args:
            prompt: The prompt to evaluate

        Returns:
            Average score across the dataset (higher is better)
        """
        scores = []

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
            print(f"LLM Response: {response.content.strip()} vs. Target: {target.strip()}") 
            # Score the response
            score = self.scoring_function(response, target)
            scores.append(score)

        print(formatted_prompt)
        print(f"Scores: {sum(scores) / len(scores)}")
        # Return the average score
        return sum(scores) / len(scores) if scores else 0.0
