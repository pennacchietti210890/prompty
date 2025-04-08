"""OpenAI LLM provider implementation."""

import json
from typing import Dict, List, Optional, Any

import openai
from openai import AsyncOpenAI

from .base import LLMProvider


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider for PROMPTy."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4-turbo",
    ):
        """Initialize the OpenAI provider.
        
        Args:
            api_key: OpenAI API key. If None, it will use the environment variable.
            model: The OpenAI model to use.
        """
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def template_prompt(self, prompt: str, instructions: Optional[str] = None) -> Dict[str, Any]:
        """Template a prompt by breaking it down into subparts using OpenAI.
        
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
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        result = json.loads(response.choices[0].message.content)
        return result

    async def call(self, prompt: str, temperature: float = 0.7, max_tokens: Optional[int] = None) -> str:
        """Make a direct call to OpenAI with the given prompt.
        
        Args:
            prompt: The prompt to send to OpenAI
            temperature: Controls randomness (0.0 = deterministic, 1.0 = creative)
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            OpenAI's response as a string
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