"""Shared test fixtures for the prompty test suite."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from prompty.optimize.bayesian.optuna_optimizer import OptunaOptimizer, SearchSpace
from prompty.prompt_components.schemas import PromptComponentCandidates

@pytest.fixture
def mock_evaluator():
    """Create a mock evaluator with async evaluate method."""
    evaluator = MagicMock()
    evaluator.evaluate = AsyncMock()
    return evaluator

@pytest.fixture
def sample_search_space():
    """Create a minimal search space for testing."""
    return {
        'component_candidates': {
            'sys_settings': PromptComponentCandidates(candidates=['setting1', 'setting2']),
            'task_description': PromptComponentCandidates(candidates=['desc1', 'desc2']),
            'task_instructions': PromptComponentCandidates(candidates=['inst1', 'inst2']),
            'user_query': PromptComponentCandidates(candidates=['query1', 'query2']),
            'training_examples': PromptComponentCandidates(candidates=['example1', 'example2'])
        },
        'other_params': {
            'temperature': 0.7,
            'max_tokens': 100
        }
    }

@pytest.fixture
def mock_experiment_tracker():
    """Create a mock experiment tracker with essential methods."""
    tracker = MagicMock()
    tracker.log_metrics = MagicMock()
    tracker.log_params = MagicMock()
    tracker.log_prompt = MagicMock()
    tracker.log_optimization_results = MagicMock()
    tracker.start_run = MagicMock()
    return tracker

@pytest.fixture
def optuna_optimizer(mock_evaluator, sample_search_space, mock_experiment_tracker):
    """Create an Optuna optimizer instance for testing."""
    return OptunaOptimizer(
        evaluator=mock_evaluator,
        search_space=SearchSpace(**sample_search_space),
        n_trials=5,
        experiment_tracker=mock_experiment_tracker
    ) 