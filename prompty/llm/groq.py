"""Groq LLM provider implementation."""

import json
from typing import Dict, List, Optional, Any

import groq

from .base import LLMProvider


class GroqProvider(LLMProvider):
    """Groq LLM provider for PROMPTy."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama3-70b-8192",
    ):
        """Initialize the Groq provider.
        
        Args:
            api_key: Groq API key. If None, it will use the environment variable.
            model: The Groq model to use.
        """
        self.client = groq.AsyncGroq(api_key=api_key)
        self.model = model

    async def template_prompt(self, prompt: str, instructions: Optional[str] = None) -> Dict[str, Any]:
        """Template a prompt by breaking it down into subparts using Groq.
        
        Args:
            prompt: The original prompt to template
            instructions: Optional specific instructions for templating
            
        Returns:
            A dictionary containing the templated prompt structure
        """
        system_prompt = """
        You are an expert at breaking down prompts into templated components.
        Analyze the given prompt and break it down into logical variable components.
        Return a JSON object with the following structure:
        {
            "components": [
                {
                    "name": "component_name",
                    "description": "What this component does",
                    "default_value": "The original text from the prompt"
                },
                ...
            ],
            "template": "The prompt with {component_name} placeholders"
        }
        """
        
        user_prompt = f"Here is the prompt to template:\n\n{prompt}"
        if instructions:
            user_prompt += f"\n\nAdditional instructions: {instructions}"
            
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2
        )
        
        # Parse the response as JSON
        try:
            result = json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            # If JSON parsing fails, create a simple template
            result = {
                "components": [
                    {
                        "name": "full_prompt",
                        "description": "The entire prompt",
                        "default_value": prompt
                    }
                ],
                "template": "{full_prompt}"
            }
            
        return result

    async def call(self, prompt: str, temperature: float = 0.7, max_tokens: Optional[int] = None) -> str:
        """Make a direct call to Groq with the given prompt.
        
        Args:
            prompt: The prompt to send to Groq
            temperature: Controls randomness (0.0 = deterministic, 1.0 = creative)
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Groq's response as a string
        """
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content 