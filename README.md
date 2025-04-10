<h1>
  <img src="assets/icon.png" width="32" alt="Logo" style="vertical-align: middle; margin-right: 10px;"/>
  PROMPTy
</h1>

A Python package for prompt optimization and prototyping.

PROMPTy allows users to quickly optimize or prototype prompts for large language models. It provides tools for templating prompts using LLMs and optimizing them using Bayesian optimization.

## Features

- **Prompt Templating**: Break down prompts into reusable components using LLMs
- **Prompt Optimization**: Optimize prompts using Bayesian optimization via Optuna
- **Multiple LLM Providers**: Support for OpenAI, Groq, and more
- **Dataset Evaluation**: Evaluate prompts on custom datasets
- **Flexible Framework**: Easily extend with new providers and evaluators

## Installation

```bash
pip install prompty
```

Or clone the repository and install locally:

```bash
git clone https://github.com/yourusername/prompty.git
cd prompty
pip install -e .
```

## Quick Start

```python
import asyncio
import pandas as pd
from prompty.llm import OpenAIProvider
from prompty.prompt_templates import TemplateManager
from prompty.optimize import DatasetEvaluator, ObjectiveFunction, PromptOptimizer

# Set up the LLM provider
llm = OpenAIProvider(api_key="your-api-key")

# Create a template manager
template_manager = TemplateManager(llm_provider=llm)

# Create a template from a prompt
async def main():
    template = await template_manager.create_template(
        prompt="Summarize the following text in a concise way: {input}",
        name="summarizer"
    )
    
    # Define a scoring function
    def score_summary(response, target):
        # Simple scoring based on length (just an example)
        if len(response) < 10:
            return 0.0
        if len(response) > 200:
            return 0.2
        return 0.8
    
    # Create a dataset for evaluation
    data = pd.DataFrame({
        "input": ["Long text to summarize...", "Another text..."],
        "target": ["Short summary", "Another summary"]
    })
    
    # Set up an evaluator
    evaluator = DatasetEvaluator(
        llm_provider=llm,
        dataset=data,
        input_column="input",
        target_column="target",
        scoring_function=score_summary,
        max_samples=10
    )
    
    # Create an objective function
    objective = ObjectiveFunction(
        evaluator=evaluator,
        template=template
    )
    
    # Create and run the optimizer
    optimizer = PromptOptimizer(
        objective=objective,
        n_trials=10
    )
    
    results = await optimizer.optimize()
    print(f"Best prompt: {results['best_prompt']}")
    print(f"Best score: {results['best_value']}")
    
    # Save the results
    optimizer.save_results("optimization_results.json")

if __name__ == "__main__":
    asyncio.run(main())
```

## Documentation

For detailed documentation, see the [docs](https://github.com/yourusername/prompty/docs).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
