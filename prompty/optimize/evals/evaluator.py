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