"""Objective function implementation for prompt optimization."""

from typing import Dict, List, Optional, Any, Union, Callable

from ..prompt_templates import PromptTemplate
from .evaluator import Evaluator


class ObjectiveFunction:
    """Objective function for prompt optimization."""
    
    def __init__(
        self,
        evaluator: Evaluator,
        template: PromptTemplate,
        param_names: Optional[List[str]] = None
    ):
        """Initialize the objective function.
        
        Args:
            evaluator: Evaluator to use for scoring prompts
            template: Prompt template to optimize
            param_names: Names of parameters to optimize (if None, all components are optimized)
        """
        self.evaluator = evaluator
        self.template = template
        
        # If no parameter names provided, use all component names
        if param_names is None:
            self.param_names = [comp.name for comp in template.components]
        else:
            # Verify that all provided param_names exist in the template
            for name in param_names:
                if name not in [comp.name for comp in template.components]:
                    raise ValueError(f"Parameter '{name}' not found in template components")
            self.param_names = param_names
    
    async def __call__(self, **kwargs) -> float:
        """Evaluate the template with the given parameters.
        
        Args:
            **kwargs: Values for template components
            
        Returns:
            Score from the evaluator (higher is better)
        """
        # Forward only the parameters we're optimizing
        optimized_params = {name: kwargs[name] for name in self.param_names if name in kwargs}
        
        # Keep other components at their default values
        for comp in self.template.components:
            if comp.name not in optimized_params:
                optimized_params[comp.name] = comp.default_value
        
        # Evaluate the template with these parameters
        score = await self.evaluator.evaluate_template(self.template, **optimized_params)
        return score
    
    def get_param_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about the parameters being optimized.
        
        Returns:
            Dictionary mapping parameter names to info about each parameter
        """
        param_info = {}
        
        for comp in self.template.components:
            if comp.name in self.param_names:
                param_info[comp.name] = {
                    "description": comp.description,
                    "default_value": comp.default_value
                }
                
        return param_info 