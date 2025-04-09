import json
import logging
import os
from typing import Any, Dict, List

import pandas as pd
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


class LLMService:
    """
    Service for interacting with Large Language Models and processing financial data.
    """

    def __init__(
        self,
        llm_provider: str = "openai",
        model_name: str = "gpt-4o",
        api_key: str = os.getenv("OPENAI_API_KEY"),
    ):
        if llm_provider == "openai":
            self.client = ChatOpenAI(api_key=api_key, model=model_name)
        elif llm_provider == "groq":
            self.client = ChatGroq(api_key=api_key, model=model_name)
        elif llm_provider == "anthropic":
            self.client = ChatAnthropic(api_key=api_key, model=model_name)
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")
