"""Base classes for LLM providers."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any


class LLMProvider(ABC):
    """Base abstract class for LLM providers.
    
    All LLM provider implementations should inherit from this class.
    """
    
    @abstractmethod
    async def template_prompt(self, prompt: str, instructions: Optional[str] = None) -> Dict[str, Any]:
        """Template a prompt by breaking it down into subparts.
        
        Args:
            prompt: The original prompt to template
            instructions: Optional specific instructions for templating
            
        Returns:
            A dictionary containing the templated prompt structure
        """
        pass
    
    @abstractmethod
    async def call(self, prompt: str, temperature: float = 0.7, max_tokens: Optional[int] = None) -> str:
        """Make a direct call to the LLM with the given prompt.
        
        Args:
            prompt: The prompt to send to the LLM
            temperature: Controls randomness (0.0 = deterministic, 1.0 = creative)
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            The LLM's response as a string
        """
        pass 