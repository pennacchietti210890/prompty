"""Evaluator classes for prompt optimization."""

import json
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Callable

import pandas as pd
from tqdm import tqdm

from ..llm import LLMProvider
from ..prompt_templates import PromptTemplate


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
        llm_provider: LLMProvider,
        dataset: Union[str, pd.DataFrame],
        input_column: str,
        target_column: str,
        scoring_function: Callable[[str, str], float],
        max_samples: Optional[int] = None
    ):
        """Initialize the dataset evaluator.
        
        Args:
            llm_provider: LLM provider to use for generating responses
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
        for _, row in tqdm(self.dataset.iterrows(), total=len(self.dataset), desc="Evaluating prompt"):
            # Get input and target from the dataset
            input_text = str(row[self.input_column])
            target = str(row[self.target_column])
            
            # Format the prompt with the input
            formatted_prompt = prompt.format(input=input_text)
            
            # Get the model's response
            response = await self.llm_provider.call(formatted_prompt)
            
            # Score the response
            score = self.scoring_function(response, target)
            scores.append(score)
        
        # Return the average score
        return sum(scores) / len(scores) if scores else 0.0
    
    async def evaluate_template(self, template: PromptTemplate, **kwargs) -> float:
        """Evaluate a prompt template on the dataset.
        
        Args:
            template: The prompt template to evaluate
            **kwargs: Values for template components
            
        Returns:
            Average score across the dataset (higher is better)
        """
        # Fill the template with provided values
        prompt = template.fill(**kwargs)
        
        # Evaluate the filled prompt
        return await self.evaluate(prompt) 