#!/usr/bin/env python3
"""Basic usage example for PROMPTy."""

import asyncio
import os
import pandas as pd
from dotenv import load_dotenv
import logging

from prompty.llm import OpenAIProvider
from prompty.prompt_templates import TemplateManager
from prompty.optimize import DatasetEvaluator, ObjectiveFunction, PromptOptimizer
from datasets import load_dataset

from langchain.chat_models import init_chat_model

# Load environment variables from .env file
load_dotenv()

# Get API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

logger = logging.getLogger(__name__)

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

    train_sample = df_train.sample(20)
    test_sample = df_test.sample(20)

    # categories for the ag news dataset are:
    # 1: World
    # 2: Sports
    # 3: Business
    # 4: Science

    # Initialize evaluator
    evaluator = DatasetEvaluator(llm=llm, dataset=test_sample, input_column="text", target_column="label_text")

    # get prompt template for text classification 
    logger.info("Loading text classification template...")
    text_classification_prompt_template = PromptTemplate(task=NLPTask.TEXT_CLASSIFICATION)


    logger.info("Checking baseline accuracy on test set....")
    baseline_score = await evaluator.evaluate(text_classification_prompt_template.template)
    logger.info(f"Baseline Score: {baseline_score}")
    
    # logger.info("Starting Optimization...")
    # logger.info("Creating Prompt Component candidates...")
    # generator = PromptGenerator(llm=llm, base_prompt=raw_text_classifier_prompt)
    # generator.get_candidate_components()
    
    # logger.info("Running Bayesina Optimization via Optuna...")
    
    # objective = ObjectiveFunction(evaluator=evaluator, template=text_classification_prompt_template)
    # optimizer = PromptOptimizer(objective=objective, n_trials=10)
    # results = await optimizer.optimize()
    
    # logger.info("Found best prompt configuration:")
    # best = study.best_trial
    # logger.info(f"Best Parameters: {best.params}")
    # logger.info(f"Best Score: {best.value}")


if __name__ == "__main__":
    asyncio.run(main())
