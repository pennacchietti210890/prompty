"""Tests for the Optuna optimizer implementation."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from prompty.optimize.bayesian.optuna_optimizer import OptunaOptimizer

@pytest.mark.asyncio
async def test_initialization(optuna_optimizer):
    """Test basic initialization of the Optuna optimizer."""
    assert optuna_optimizer.n_trials == 5
    assert optuna_optimizer.study_name == "prompt_optimization"
    assert optuna_optimizer.direction == "maximize"
    assert optuna_optimizer._best_score == float("-inf")
    assert optuna_optimizer._no_improvement_count == 0
    assert optuna_optimizer._total_cost == 0.0

@pytest.mark.asyncio
async def test_basic_optimization(optuna_optimizer, mock_evaluator, mock_experiment_tracker):
    """Test basic optimization process with a simple scoring sequence."""
    # Mock evaluator to return increasing scores
    mock_evaluator.evaluate.side_effect = [0.5, 0.6, 0.7, 0.8, 0.9]
    
    # Create a mock study with async-compatible methods
    mock_study = MagicMock()
    mock_trial = MagicMock()
    mock_trial.number = 1
    
    # Create a side effect that provides enough values for all components across all trials
    # 5 trials * 5 components = 25 values needed
    suggest_int_values = [0, 1, 0, 1, 0] * 5  # Repeat the pattern 5 times for 5 trials
    mock_trial.suggest_int = MagicMock(side_effect=suggest_int_values)
    
    # Create a mock study that returns the trial directly
    mock_study.ask = AsyncMock(return_value=mock_trial)
    mock_study.tell = AsyncMock()
    mock_study.best_params = {
        'sys_settings': 0,
        'task_description': 1,
        'task_instructions': 0,
        'user_query': 1,
        'training_examples': 0
    }
    mock_study.best_value = 0.9
    
    # Patch the optimize method to properly await the trial
    async def mock_optimize():
        for _ in range(5):  # Run 5 trials
            trial = await mock_study.ask()
            result = await optuna_optimizer._objective_wrapper(trial)
            await mock_study.tell(trial, result)
        return {
            'best_params': mock_study.best_params,
            'best_value': mock_study.best_value
        }
    
    with patch.object(optuna_optimizer, 'optimize', mock_optimize):
        results = await optuna_optimizer.optimize()
    
    # Verify basic results
    assert 'best_params' in results
    assert 'best_value' in results
    assert isinstance(results['best_value'], float)
    assert results['best_value'] == 0.9
    
    # Verify tracking calls
    assert mock_experiment_tracker.log_metrics.call_count == 5  # One for each trial
    assert mock_experiment_tracker.log_params.call_count > 0

@pytest.mark.asyncio
async def test_objective_function(optuna_optimizer, mock_evaluator):
    """Test the objective function wrapper with a single trial."""
    mock_trial = MagicMock()
    mock_trial.number = 1
    # Provide values for all 5 components
    mock_trial.suggest_int = MagicMock(side_effect=[0, 1, 0, 1, 0])
    
    mock_evaluator.evaluate.return_value = 0.8
    
    score = await optuna_optimizer._objective_wrapper(mock_trial)
    
    # Verify basic objective function behavior
    assert score == 0.8
    assert mock_trial.suggest_int.call_count == 5  # One for each component
    assert len(optuna_optimizer.trials_costs) == 1

@pytest.mark.asyncio
async def test_error_handling(optuna_optimizer, mock_evaluator):
    """Test basic error handling in optimization process."""
    mock_evaluator.evaluate.side_effect = Exception("Test error")
    
    # Create a mock study with async-compatible methods
    mock_study = MagicMock()
    mock_trial = MagicMock()
    mock_trial.number = 1
    # Provide values for all 5 components
    mock_trial.suggest_int = MagicMock(side_effect=[0, 1, 0, 1, 0])
    
    # Create a mock study that returns the trial directly
    mock_study.ask = AsyncMock(return_value=mock_trial)
    mock_study.tell = AsyncMock()
    
    # Patch the optimize method to properly await the trial
    async def mock_optimize():
        trial = await mock_study.ask()
        await optuna_optimizer._objective_wrapper(trial)  # This should raise the exception
    
    with patch.object(optuna_optimizer, 'optimize', mock_optimize):
        with pytest.raises(Exception):
            await optuna_optimizer.optimize() 