import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

from jinja2 import DebugUndefined, Environment, Template
from langchain_core.language_models.chat_models import BaseChatModel

from prompty.prompt_components.schemas import (
    PromptComponentCandidates,
    PromptComponents,
)

logger = logging.getLogger(__name__)


class PromptGenerator:
    """
    A class that generates and manages prompt components for large language models.

    This class helps break down prompts into components, generate variations of those
    components, and reconstruct structured prompts for LLM interactions.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        base_prompt: str = "",
        components: Optional["PromptComponents"] = None,
    ):
        """
        Initialize a PromptGenerator instance.

        Args:
            llm: Base chat model used for generating and processing prompts
            base_prompt: The initial prompt text to be structured into components
            components: Optional pre-defined prompt components
        """
        self.llm = llm
        self.base_prompt = base_prompt
        self.candidates: Dict[str, "PromptComponentCandidates"] = {}

        template_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "prompt_components",
            "templates",
            "components",
        )
        logger.info(
            f"Loading prompt componentgeneration templates from {template_path}"
        )

        with open(
            os.path.join(template_path, "get_component.txt"), "r", encoding="utf-8"
        ) as file:
            self.components_generate_template = file.read()

        with open(
            os.path.join(template_path, "sys_settings.txt"), "r", encoding="utf-8"
        ) as file:
            self.sys_settings_generate_template = file.read()

        with open(
            os.path.join(template_path, "task_description.txt"), "r", encoding="utf-8"
        ) as file:
            self.task_description_generate_template = file.read()

        with open(
            os.path.join(template_path, "task_instructions.txt"), "r", encoding="utf-8"
        ) as file:
            self.task_instructions_generate_template = file.read()

        with open(
            os.path.join(template_path, "user_query.txt"), "r", encoding="utf-8"
        ) as file:
            self.user_query_generate_template = file.read()

        if not components:
            self._components = self._get_components()
        else:
            self._components = components

    @property
    def components(self) -> "PromptComponents":
        """
        Get the prompt components.

        Returns:
            The prompt components object
        """
        return self._components

    def _get_components(self) -> "PromptComponents":
        """
        Break down the base prompt into structured components using the LLM.

        Returns:
            A structured representation of the prompt components
        """
        structured_llm = self.llm.with_structured_output(PromptComponents)
        components = structured_llm.invoke(
            Template(self.components_generate_template).render(
                raw_prompt=self.base_prompt
            )
        )
        return components

    @components.setter
    def components(self, value: "PromptComponents") -> None:
        """
        Set the prompt components.

        Args:
            value: The prompt components to set

        Raises:
            ValueError: If the provided value is not a PromptComponents instance
        """
        if not isinstance(value, PromptComponents):
            raise ValueError("Prompt Components must be of PromptComponents type")
        self._components = value

    def get_structured_prompt(self) -> str:
        """
        Generate a structured prompt from the components.

        Returns:
            A formatted string containing the structured prompt
        """
        if not self._components:
            self._get_components()
        full_prompt = ""
        for component in self.components:
            if component[1]:
                full_prompt += f"**{component[0]}**\n"
                full_prompt += f"{component[1]}\n\n"
        return full_prompt

    def get_candidate_components(self, num_candidates: int = 2) -> None:
        """
        Generate candidate variations for each prompt component.

        Args:
            num_candidates: Number of candidate variations to generate for each component
        """
        if not self._components:
            self._get_components()

        candidate_generation_templates = {
            "sys_settings": Template(self.sys_settings_generate_template).render(
                n=num_candidates, rewrite=self._components.sys_settings
            ),
            "task_description": Template(
                self.task_description_generate_template
            ).render(n=num_candidates, rewrite=self._components.task_description),
            "task_instructions": Template(
                self.task_instructions_generate_template
            ).render(n=num_candidates, rewrite=self._components.task_instructions),
            "user_query": Template(self.user_query_generate_template).render(
                n=num_candidates, rewrite=self._components.user_query
            ),
        }

        for component in self.components:
            if component[1]:
                logging.info(f"Generating Candidates for component: {component[0]}")
                candidate_template = candidate_generation_templates[component[0]]
                structured_llm = self.llm.with_structured_output(
                    PromptComponentCandidates
                )
                candidates = structured_llm.invoke(candidate_template)
                self.candidates[component[0]] = candidates

        logging.info(f"Candidates generation finished!")
