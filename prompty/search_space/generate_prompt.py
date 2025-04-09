from typing import Dict, List, Optional, Tuple, Any, Union
from langchain.schema.language_model import BaseChatModel
import logging
from prompty.prompt_components.schemas import PromptComponents, PromptComponentCandidates

class PromptGenerator:
    """
    A class that generates and manages prompt components for large language models.
    
    This class helps break down prompts into components, generate variations of those
    components, and reconstruct structured prompts for LLM interactions.
    """

    def __init__(self, llm: BaseChatModel, base_prompt: str = "", components: Optional["PromptComponents"] = None):
        """
        Initialize a PromptGenerator instance.
        
        Args:
            llm: Base chat model used for generating and processing prompts
            base_prompt: The initial prompt text to be structured into components
            components: Optional pre-defined prompt components
        """
        self.llm = llm
        self.base_prompt = base_prompt
        if not components:
            self._components = self._get_components()
        self.candidates: Dict[str, "PromptComponentCandidates"] = {}

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
        components = structured_llm.invoke(template_prompt.format(raw_prompt=self.base_prompt))
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

        candidate_generation_templates = PromptComponents(
            sys_settings=template_generate_sys_settings.format(n=num_candidates, rewrite=self._components.sys_settings),
            task_description=template_generate_task_description.format(n=num_candidates, rewrite=self._components.task_description),
            task_instructions=template_generate_task_instructions.format(n=num_candidates, rewrite=self._components.task_instructions),
            training_examples=[],]
            user_query=template_generate_user_query.format(n=num_candidates, rewrite=self._components.user_query),
        )

        for component in self.components:
            if component[1]:
                logging.info(f"Generating Candidates for component: {component[0]}")
                candidate_template = candidate_generation_templates.dict()[component[0]]
                structured_llm = self.llm.with_structured_output(PromptComponentCandidates)
                candidates = structured_llm.invoke(candidate_template)
                self.candidates[component[0]] = candidates
        
        logging.info(f"Candidates generation finished!")