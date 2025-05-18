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
from prompty.optimize.bayesian.optuna_optimizer import OptunaOptimizer, SearchSpace
from prompty.optimize.evals.cost_aware_evaluator import CostAwareEvaluator
from prompty.optimize.evals.dataset_evaluator import DatasetEvaluator
from prompty.prompt_components.schemas import (
    NLPTask,
    PromptComponentCandidates,
    PromptTemplate,
)
from prompty.search_space.generate_prompt import PromptGenerator
from prompty.search_space.generate_training import BestShotsSelector
from prompty.tracking.wandb_tracking import WandbTracker

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

    # Load WMT14 English-French dataset
    dataset = load_dataset("wmt14", "fr-en")
    df_train = (
        dataset["train"].shuffle(seed=42).select(range(500)).to_pandas()
    )  # sample to speed up
    df_test = dataset["test"].shuffle(seed=42).select(range(50)).to_pandas()

    train_sample = df_train.sample(10)
    test_sample = df_test.sample(10)

    train_sample["en"] = train_sample["translation"].apply(lambda x: x["en"])
    train_sample["fr"] = train_sample["translation"].apply(lambda x: x["fr"])

    test_sample["en"] = test_sample["translation"].apply(lambda x: x["en"])
    test_sample["fr"] = test_sample["translation"].apply(lambda x: x["fr"])

    from evaluate import load as load_metric

    bleu_metric = load_metric("bleu")

    def bleu_score_fn(pred: str, ref: str) -> float:
        # BLEU expects lists of tokens
        result = bleu_metric.compute(predictions=[pred.content], references=[[ref]])
        return result["bleu"]

    # Construct translation evaluator
    evaluator = DatasetEvaluator(
        llm_provider=llm,
        dataset=test_sample,
        input_column="en",
        target_column="fr",
        scoring_function=bleu_score_fn,
    )

    logger.info("Loading translation prompt template...")
    translation_prompt_template = PromptTemplate(task=NLPTask.TRANSLATION)
    prompt_template = env.from_string(
        translation_prompt_template.load_template_from_task()
    ).render()

    logger.info("Template:")
    logger.info(prompt_template)

    logger.info("Checking baseline BLEU on test set....")
    baseline_score = await evaluator.evaluate(prompt_template)
    logger.info(f"Baseline BLEU: {baseline_score:.2f}")

    logger.info("Creating prompt candidates...")
    generator = PromptGenerator(llm=llm, base_prompt=prompt_template)
    generator.get_candidate_components()

    logger.info("Extracting few-shot examples...")
    examples = list(train_sample["en"])
    translations = list(train_sample["fr"])
    best_shots_selector = BestShotsSelector(examples)
    diverse_shots = best_shots_selector.min_max_diverse_subset(20)
    translations = [translations[examples.index(s)] for s in diverse_shots]

    examples_str = "\n\n".join(
        [f"English: {en}\nFrench: {fr}" for en, fr in zip(diverse_shots, translations)]
    )
    shots_prompt = (
        f"Here are some English to French translation examples:\n\n{examples_str}"
    )

    shots_prompt_candidates = PromptComponentCandidates(candidates=[shots_prompt])
    final_candidates = generator.candidates
    final_candidates["training_examples"] = shots_prompt_candidates

    search_space = SearchSpace(component_candidates=final_candidates, other_params={})

    logger.info("Running Optuna for BLEU optimization...")
    optimizer = OptunaOptimizer(
        evaluator=evaluator, search_space=search_space, n_trials=5
    )
    results = await optimizer.optimize()

    logger.info("Best Prompt Found:")
    logger.info(f"Params: {results['best_params']}")
    logger.info(f"BLEU Score: {results['best_value']:.2f}")

    optimizer.save_results("translation_optimization_results.json")


if __name__ == "__main__":
    asyncio.run(main())
