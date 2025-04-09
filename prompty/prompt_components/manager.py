"""Template manager implementation."""

import json
import os
from typing import Dict, List, Optional, Any, Union

from ..llm import LLMProvider
from .template import PromptTemplate


class TemplateManager:
    """Manager for creating and handling prompt templates."""
    
    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        """Initialize the template manager.
        
        Args:
            llm_provider: Optional LLM provider for generating templates
        """
        self.llm_provider = llm_provider
        self.templates: Dict[str, PromptTemplate] = {}
        
    async def create_template(
        self, 
        prompt: str, 
        name: str, 
        instructions: Optional[str] = None,
        description: Optional[str] = None
    ) -> PromptTemplate:
        """Create a new template from a prompt using an LLM.
        
        Args:
            prompt: The prompt to template
            name: Name for the template
            instructions: Optional instructions for the LLM
            description: Optional description for the template
            
        Returns:
            The created PromptTemplate
            
        Raises:
            ValueError: If no LLM provider is set
        """
        if not self.llm_provider:
            raise ValueError("No LLM provider set for template creation")
            
        llm_response = await self.llm_provider.template_prompt(prompt, instructions)
        template = PromptTemplate.from_llm_response(llm_response, name=name)
        
        if description:
            template.description = description
            
        self.templates[name] = template
        return template
    
    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Get a template by name.
        
        Args:
            name: Name of the template
            
        Returns:
            The template if found, None otherwise
        """
        return self.templates.get(name)
    
    def add_template(self, template: PromptTemplate) -> None:
        """Add a template to the manager.
        
        Args:
            template: The template to add
            
        Raises:
            ValueError: If the template has no name
        """
        if not template.name:
            raise ValueError("Template must have a name to be added to the manager")
            
        self.templates[template.name] = template
    
    def remove_template(self, name: str) -> bool:
        """Remove a template by name.
        
        Args:
            name: Name of the template to remove
            
        Returns:
            True if the template was removed, False if it wasn't found
        """
        if name in self.templates:
            del self.templates[name]
            return True
        return False
    
    def list_templates(self) -> List[str]:
        """List all template names.
        
        Returns:
            List of template names
        """
        return list(self.templates.keys())
    
    def save_templates(self, file_path: str) -> None:
        """Save all templates to a JSON file.
        
        Args:
            file_path: Path to save the templates to
        """
        data = {name: template.to_dict() for name, template in self.templates.items()}
        
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load_templates(cls, file_path: str, llm_provider: Optional[LLMProvider] = None) -> "TemplateManager":
        """Load templates from a JSON file.
        
        Args:
            file_path: Path to load templates from
            llm_provider: Optional LLM provider for the new manager
            
        Returns:
            A new TemplateManager with loaded templates
        """
        manager = cls(llm_provider=llm_provider)
        
        if not os.path.exists(file_path):
            return manager
            
        with open(file_path, "r") as f:
            data = json.load(f)
            
        for name, template_data in data.items():
            # Convert components data to PromptComponent objects
            template = PromptTemplate(
                name=template_data.get("name", name),
                description=template_data.get("description"),
                template=template_data["template"],
                components=[
                    {
                        "name": comp["name"],
                        "description": comp["description"],
                        "default_value": comp.get("default_value", "")
                    }
                    for comp in template_data["components"]
                ]
            )
            manager.templates[name] = template
            
        return manager 