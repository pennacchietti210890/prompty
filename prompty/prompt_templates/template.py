"""Prompt template implementation."""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field


class PromptComponent(BaseModel):
    """A component of a prompt template."""
    
    name: str = Field(..., description="Unique name for this component")
    description: str = Field(..., description="Description of what this component does")
    default_value: str = Field("", description="Default value for this component")
    
    def __str__(self) -> str:
        """Return string representation of the component."""
        return f"{{{self.name}}}"


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