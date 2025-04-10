from pydantic import BaseModel, Field
from typing import List, Optional, TypedDict

class PromptComponents(BaseModel):
    """Key components making up a prompt."""

    sys_settings: str = Field(description="The systema and persona settings for the LLM to adequately carry out the desired task")
    task_description: str = Field(description="The description of the task the LLM has to perform")
    task_instructions: str = Field(description="The instructions that need to be followed by the LLM in order to complete the task. If the instructions contain a bullet point or numbered list, return the list as it is. ")
    training_examples: Optional[List[str]] = Field(description="A list of training examples with input/output pair demonstrations to guide the LLM towards successfully achieving the task")
    user_query: str = Field(description="The final query the user asks")


class PromptComponentCandidates(BaseModel):
    """Key components making up a prompt."""

    candidates: List[str] = Field(description="A list of candidate prompts for a specific prompt component")


class PromptTemplate(BaseModel):
    """A template for a prompt with variable components."""

    task: str = Field(..., description="NLP Task to be corresponding to the template")
    components: Optional[List[PromptComponent]] = Field(None, default_factory=list, 
                                             description="List of components in this template")
    name: Optional[str] = Field(None, description="Optional name for the template"
    description: Optional[str] = Field(None, description="Optional description of the template")
    
    def load_template_from_task(self) -> str:
        """Load the template from the template file corresponding to the task.
                    
        Returns:
            The template
        """
        with open('your_file.txt', 'r', encoding='utf-8') as file:
            content = file.read()
        self.template = content
    
    def load_template_from_components(self, components: List[PromptComponent]) -> str:
        """Load the template given a list of Prompt Components.
        
        Args:
            components: Component values 
            
        Returns:
            The filled template prompt as a string
        """
        self.template = ""
        for component in self.components:
            self.template += f"{component}\n\n"
            
        return self.template

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