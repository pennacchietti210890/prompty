#!/usr/bin/env python3
"""Basic usage example for PROMPTy."""

import asyncio
import os
import pandas as pd
from dotenv import load_dotenv
import logging
from datasets import load_dataset

from prompty.prompt_components.schemas import PromptTemplate, NLPTask
from prompty.search_space.generate_prompt import PromptGenerator
from prompty.optimize.evaluator import DatasetEvaluator
from prompty.optimize.optimizer import Optimizer, SearchSpace
from datasets import load_dataset

from langchain.chat_models import init_chat_model
from string import Template

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Output to console
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Get API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

async def main():
    """Run a simple prompt optimization example."""

    # Initialize LLM
    llm = init_chat_model("gpt-4o-mini", model_provider="openai")

    # Load test dataset - AG News for text classification
    ag_news_dataset = load_dataset("ag_news")
    label_names = ag_news_dataset["train"].features["label"].names

    # Convert HF dataset to pandas dataframe
    df_train = ag_news_dataset["train"].to_pandas()
    df_test = ag_news_dataset["test"].to_pandas()

    # Optional: map numeric labels to strings
    df_train["label_text"] = df_train["label"].apply(lambda x: label_names[x])
    df_test["label_text"] = df_test["label"].apply(lambda x: label_names[x])

    train_sample = df_train.sample(2)
    test_sample = df_test.sample(2)

    # categories for the ag news dataset are:
    # 1: World
    # 2: Sports
    # 3: Business
    # 4: Science
    labels_names = """
        - World
        - Sports
        - Business
        - Sci/Tech
    """

    # Initialize evaluator
    evaluator = DatasetEvaluator(llm_provider=llm, dataset=test_sample, input_column="text", target_column="label_text")

    # get prompt template for text classification 
    logger.info("Loading text classification template...")
    text_classification_prompt_template = PromptTemplate(task=NLPTask.TEXT_CLASSIFICATION)
    prompt_template = Template(text_classification_prompt_template.load_template_from_task()).safe_substitute(categories=labels_names)

    logger.info("Template:")
    logger.info(prompt_template)
    
    logger.info("Checking baseline accuracy on test set....")
    baseline_score = await evaluator.evaluate(prompt_template)
    logger.info(f"Baseline Score: {baseline_score}")
    
    logger.info("Starting Optimization...")
    logger.info("Creating Prompt Component candidates...")
    generator = PromptGenerator(llm=llm, base_prompt=prompt_template)
    generator.get_candidate_components()
    
    logger.info("Search space: candidate generation complete...")
    logger.info("Running Bayesian Optimization via Optuna...")
    
    # Create search space from candidates
    search_space = SearchSpace(
        component_candidates=generator.candidates,
        other_params={}
    )
    
    optimizer = Optimizer(evaluator=evaluator, search_space=search_space, n_trials=10)
    results = await optimizer.optimize()
    
    logger.info("Found best prompt configuration:")
    logger.info(f"Best Parameters: {results['best_params']}")
    logger.info(f"Best Score: {results['best_value']}")
    
    # Save the results
    optimizer.save_results("optimization_results.json")
    logger.info("Results saved to optimization_results.json")


if __name__ == "__main__":
    asyncio.run(main())
