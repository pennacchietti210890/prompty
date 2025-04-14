from pydantic import BaseModel, Field
from typing import List, Optional, TypedDict, Dict, Any
from enum import Enum
import logging
import os 

logger = logging.getLogger(__name__)

class PromptComponents(BaseModel):
    """Key components making up a prompt."""

    sys_settings: str = Field(
        description="The systema and persona settings for the LLM to adequately carry out the desired task"
    )
    task_description: str = Field(
        description="The description of the task the LLM has to perform"
    )
    task_instructions: str = Field(
        description="The instructions that need to be followed by the LLM in order to complete the task. If the instructions contain a bullet point or numbered list, return the list as it is. "
    )
    training_examples: Optional[str] = Field(
        description="A list of training examples with input/output pair demonstrations to guide the LLM towards successfully achieving the task"
    )
    user_query: str = Field(description="The final query the user asks")


class PromptComponentCandidates(BaseModel):
    """Key components making up a prompt."""

    candidates: List[str] = Field(
        description="A list of candidate prompts for a specific prompt component"
    )


class NLPTask(Enum):
    """NLP Tasks with an associated prompt template."""

    TEXT_CLASSIFICATION = "text_classification"
    QUESTION_ANSWERING = "question_answering"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    PARAPHRAZING = "paraphrasing"
    SENTIMENT_ANALYSIS = "sentiment_analysis"


class PromptTemplate(BaseModel):
    """A template for a prompt with variable components."""

    task: Optional[NLPTask] = Field(None, description="NLP Task to be corresponding to the template")
    components: Optional[List[PromptComponents]] = Field(
        None, description="List of components in this template"
    )
    name: Optional[str] = Field(None, description="Optional name for the template")
    description: Optional[str] = Field(
        None, description="Optional description of the template"
    )

    def load_template_from_task(self) -> str:
        """Load the template from the template file corresponding to the task.

        Returns:
            The template
        """
        template_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "prompt_components",
            "templates",
            "tasks",
            f"{self.task.value}.txt"
        )
        logger.info(f"Loading template from {template_path}")
        with open(template_path, "r", encoding="utf-8") as file:
            content = file.read()
        
        return content

    def load_template_from_components(self, components: List[PromptComponents]) -> str:
        """Load the template given a list of Prompt Components.

        Args:
            components: Component values

        Returns:
            The filled template prompt as a string
        """
        content = ""
        print(components)
        for component in components:
            content += f"**{component[0]}**\n"
            content += f"{component[1]}\n\n"

        return content

    def fill(self, **kwargs) -> str:
        """Fill the template with provided values.

        Args:
            **kwargs: Keyword arguments for the template

        Returns:
            The filled prompt as a string
        """
        # Fill the template
        if self.template is None:
            raise ValueError("Template is not loaded. Please load the template first.")
        return self.template.format(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the template to a dictionary.

        Returns:
            Dictionary representation of the template
        """
        return {
            "name": self.name,
            "description": self.description,
            "template": self.template,
        }
