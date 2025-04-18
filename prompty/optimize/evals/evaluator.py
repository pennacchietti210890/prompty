"""Evaluator classes for prompt optimization."""

import json
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd
from jinja2 import DebugUndefined, Environment, Template
from langchain_core.language_models.chat_models import BaseChatModel
from tqdm import tqdm

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
