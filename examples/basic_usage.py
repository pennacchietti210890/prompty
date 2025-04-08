#!/usr/bin/env python3
"""Basic usage example for PROMPTy."""

import asyncio
import os
import pandas as pd
from dotenv import load_dotenv

from prompty.llm import OpenAIProvider
from prompty.prompt_templates import TemplateManager
from prompty.optimize import DatasetEvaluator, ObjectiveFunction, PromptOptimizer


# Load environment variables from .env file
load_dotenv()

# Get API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


async def main():
    """Run a simple prompt optimization example."""
    
    # Create an OpenAI provider
    llm = OpenAIProvider(api_key=OPENAI_API_KEY)
    
    # Create a template manager
    manager = TemplateManager(llm_provider=llm)
    
    # Create a template from a prompt
    print("Creating template...")
    template = await manager.create_template(
        prompt="You are a helpful assistant. Please answer the following question: {input}",
        name="question_answerer",
        instructions="Break this down into components like introduction, question prompt, etc."
    )
    
    # Print the template structure
    print("\nTemplate structure:")
    for i, comp in enumerate(template.components):
        print(f"  Component {i+1}: {comp.name}")
        print(f"    Description: {comp.description}")
        print(f"    Default value: {comp.default_value}")
    
    print(f"\nTemplate string: {template.template}")
    
    # Create a simple example dataset
    print("\nCreating sample dataset...")
    data = pd.DataFrame({
        "input": [
            "What is the capital of France?",
            "How does photosynthesis work?",
            "Who wrote 'Pride and Prejudice'?"
        ],
        "target": [
            "The capital of France is Paris.",
            "Photosynthesis is the process used by plants to convert light energy into chemical energy.",
            "Jane Austen wrote 'Pride and Prejudice'."
        ]
    })
    
    # Define a simple scoring function
    def score_response(response, target):
        """Score a response against a target.
        
        This is a very simple scoring function that checks if the target is contained
        in the response. In a real application, you'd want a more sophisticated scorer.
        """
        # Lower case both for case-insensitive comparison
        response_lower = response.lower()
        target_lower = target.lower()
        
        # Check if the target is in the response
        if target_lower in response_lower:
            return 1.0
        
        # Check if at least half the words in the target are in the response
        target_words = set(target_lower.split())
        matching_words = sum(1 for word in target_words if word in response_lower)
        if matching_words >= len(target_words) / 2:
            return 0.5
            
        return 0.0
    
    # Create an evaluator
    print("Setting up evaluator...")
    evaluator = DatasetEvaluator(
        llm_provider=llm,
        dataset=data,
        input_column="input",
        target_column="target",
        scoring_function=score_response,
        max_samples=3  # Use all 3 samples
    )
    
    # Create an objective function
    objective = ObjectiveFunction(
        evaluator=evaluator,
        template=template
    )
    
    # Create and run the optimizer
    print("Running optimizer (this may take a while)...")
    optimizer = PromptOptimizer(
        objective=objective,
        n_trials=5  # Only 5 trials for demo purposes
    )
    
    results = await optimizer.optimize()
    
    # Print the results
    print("\nOptimization results:")
    print(f"Best score: {results['best_value']}")
    print(f"Best parameters:")
    for param, value in results['best_params'].items():
        print(f"  {param}: {value}")
    
    print(f"\nBest prompt:\n{results['best_prompt']}")
    
    # Save the results
    optimizer.save_results("optimization_results.json")
    print("\nResults saved to optimization_results.json")


if __name__ == "__main__":
    asyncio.run(main()) 