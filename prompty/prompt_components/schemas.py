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
    
    components: List[PromptComponent] = Field(default_factory=list, 
                                             description="List of components in this template")
    template: str = Field(..., description="Template string with {component_name} placeholders")
    name: Optional[str] = Field(None, description="Optional name for the template")
    description: Optional[str] = Field(None, description="Optional description of the template")
    
    def fill(self, **kwargs) -> str:
        """Fill the template with provided values.
        
        Args:
            **kwargs: Component values keyed by component name
            
        Returns:
            The filled prompt as a string
        """
        # Create a dictionary of component values, using defaults for missing components
        values = {}
        for component in self.components:
            values[component.name] = kwargs.get(component.name, component.default_value)
            
        # Fill the template
        return self.template.format(**values)
    
    @classmethod
    def from_llm_response(cls, response: Dict[str, Any], name: Optional[str] = None) -> "PromptTemplate":
        """Create a PromptTemplate from an LLM response.
        
        Args:
            response: Dictionary containing components and template from LLM
            name: Optional name for the template
            
        Returns:
            A new PromptTemplate instance
        """
        components = [
            PromptComponent(
                name=comp["name"],
                description=comp["description"],
                default_value=comp["default_value"]
            )
            for comp in response["components"]
        ]
        
        return cls(
            components=components,
            template=response["template"],
            name=name
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the template to a dictionary.
        
        Returns:
            Dictionary representation of the template
        """
        return {
            "name": self.name,
            "description": self.description,
            "template": self.template,
            "components": [
                {
                    "name": comp.name,
                    "description": comp.description,
                    "default_value": comp.default_value
                }
                for comp in self.components
            ]
        } 