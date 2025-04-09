"""Prompt templates module for PROMPTy.

This module provides functionality for creating and manipulating prompt templates.
"""

from .template import PromptTemplate, PromptComponent
from .manager import TemplateManager

__all__ = ["PromptTemplate", "PromptComponent", "TemplateManager"]
