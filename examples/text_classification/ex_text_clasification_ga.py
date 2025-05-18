#!/usr/bin/env python3
"""Basic usage example for PROMPTy."""

import asyncio
import logging
import os

import mlflow
import pandas as pd
from dotenv import load_dotenv
from jinja2 import DebugUndefined, Environment, Template
from langchain.chat_models import init_chat_model

from datasets import load_dataset
from prompty.optimize.evals.cost_aware_evaluator import CostAwareEvaluator
from prompty.optimize.evals.dataset_evaluator import DatasetEvaluator
from prompty.optimize.genetic_algorithms.ga_optimizer import (GAOptimizer,
                                                              SearchSpace)
from prompty.prompt_components.schemas import (NLPTask,
                                               PromptComponentCandidates,
                                               PromptTemplate)
from prompty.search_space.generate_prompt import PromptGenerator
from prompty.search_space.generate_training import BestShotsSelector

env = Environment(undefined=DebugUndefined)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],  # Output to console
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Get API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


async def main():
    """Run a simple prompt optimization example."""

    # Initialize LLM
    llm = init_chat_model("gpt-4.1-nano", model_provider="openai")
    mlflow.openai.autolog()
    # Load test dataset - AG News for text classification
    ag_news_dataset = load_dataset("ag_news")
    label_names = ag_news_dataset["train"].features["label"].names

    # Convert HF dataset to pandas dataframe
    df_train = ag_news_dataset["train"].to_pandas()
    df_test = ag_news_dataset["test"].to_pandas()

    # Optional: map numeric labels to strings
    df_train["label_text"] = df_train["label"].apply(lambda x: label_names[x])
    df_test["label_text"] = df_test["label"].apply(lambda x: label_names[x])

    train_sample = df_train.sample(10)
    test_sample = df_test.sample(2)

    labels_names = """
        - World
        - Sports
        - Business
        - Sci/Tech
    """

    # Initialize evaluator
    # evaluator = DatasetEvaluator(llm_provider=llm, dataset=test_sample, input_column="text", target_column="label_text")
    evaluator = CostAwareEvaluator(
        llm_provider=llm,
        dataset=test_sample,
        input_column="text",
        target_column="label_text",
        cost_weight=0,
        performance_weight=1,
    )

    # get prompt template for text classification
    logger.info("Loading text classification template...")
    text_classification_prompt_template = PromptTemplate(
        task=NLPTask.TEXT_CLASSIFICATION
    )
    prompt_template = env.from_string(
        text_classification_prompt_template.load_template_from_task()
    ).render(categories=labels_names)

    logger.info("Template:")
    logger.info(prompt_template)

    logger.info("Checking baseline accuracy on test set....")
    baseline_score = await evaluator.evaluate(prompt_template)
    logger.info(f"Baseline Score: {baseline_score}")

    logger.info("Starting Genetic Algorithms Optimization...")
    logger.info("Creating Prompt Component candidates...")

    generator = PromptGenerator(llm=llm, base_prompt=prompt_template)
    generator.get_candidate_components()

    logger.info("Extracting best training examples...")
    examples = list(train_sample["text"])
    best_shots_selector = BestShotsSelector(examples)
    diverse_shots = best_shots_selector.min_max_diverse_subset(2)

    logger.info("Constructing input/output pairs for best shots...")
    labels = list(train_sample["label_text"])
    labels = [labels[examples.index(shot)] for shot in diverse_shots]

    examples_str = "\n\n".join(
        [f"{example}. \n{label}" for example, label in zip(diverse_shots, labels)]
    )

    shots_prompt = f"Here are some examples of similar text to the one you have to classify, with their corresponding labels:\n{examples_str}"

    shots_prompt_candidates = PromptComponentCandidates(candidates=[shots_prompt])
    final_candidates = generator.candidates
    final_candidates["training_examples"] = shots_prompt_candidates

    logger.info("Search space: candidate generation complete...")
    logger.info("Running DEAP Genetic Algorithm Optimization...")

    # Create search space from candidates
    search_space = SearchSpace(component_candidates=final_candidates, other_params={})

    optimizer = GAOptimizer(evaluator=evaluator, search_space=search_space)
    results = await optimizer.optimize()

    logger.info("Found best prompt configuration:")
    logger.info(f"Best Parameters: {results['best_params']}")
    logger.info(f"Best Score: {results['best_value']}")

    # Save the results
    optimizer.save_results("optimization_results.json")
    logger.info("Results saved to optimization_results.json")


if __name__ == "__main__":
    asyncio.run(main())
