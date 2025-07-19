"""
Template Loader for Modularized Dashboard

This module provides utilities for loading and managing the modularized
dashboard templates.
"""

import os
from typing import Dict, Optional
from pathlib import Path

class DashboardTemplateLoader:
    """Manages loading of modularized dashboard templates"""
    
    def __init__(self, template_dir: str = None):
        """
        Initialize the template loader
        
        Args:
            template_dir: Path to templates directory
        """
        if template_dir is None:
            # Get the path relative to this file
            template_dir = Path(__file__).parent
        
        self.template_dir = Path(template_dir)
        self.components_dir = self.template_dir / 'components'
        
    def get_component_path(self, component_name: str) -> Path:
        """
        Get the full path to a component file
        
        Args:
            component_name: Name of the component (without .html extension)
            
        Returns:
            Path to the component file
        """
        return self.components_dir / f"{component_name}.html"
    
    def load_component(self, component_name: str) -> Optional[str]:
        """
        Load a component's content
        
        Args:
            component_name: Name of the component to load
            
        Returns:
            Component content as string, or None if not found
        """
        component_path = self.get_component_path(component_name)
        
        if component_path.exists():
            with open(component_path, 'r', encoding='utf-8') as f:
                return f.read()
        return None
    
    def get_all_components(self) -> Dict[str, str]:
        """
        Load all available components
        
        Returns:
            Dictionary mapping component names to their content
        """
        components = {}
        
        if self.components_dir.exists():
            for component_file in self.components_dir.glob('*.html'):
                component_name = component_file.stem
                components[component_name] = self.load_component(component_name)
        
        return components
    
    def validate_template_structure(self) -> Dict[str, bool]:
        """
        Validate that all expected components exist
        
        Returns:
            Dictionary mapping component names to existence status
        """
        expected_components = [
            'head',
            'styles', 
            'navbar',
            'overview_metrics',
            'charts',
            'positions_table',
            'alerts',
            'rebalancing_table'
        ]
        
        validation_results = {}
        for component in expected_components:
            validation_results[component] = self.get_component_path(component).exists()
        
        return validation_results
    
    def get_template_info(self) -> Dict[str, any]:
        """
        Get information about the template structure
        
        Returns:
            Dictionary with template metadata
        """
        info = {
            'template_dir': str(self.template_dir),
            'components_dir': str(self.components_dir),
            'components': [],
            'total_size': 0
        }
        
        if self.components_dir.exists():
            for component_file in self.components_dir.glob('*'):
                if component_file.is_file():
                    size = component_file.stat().st_size
                    info['components'].append({
                        'name': component_file.name,
                        'size': size,
                        'type': component_file.suffix
                    })
                    info['total_size'] += size
        
        return info

# Singleton instance
template_loader = DashboardTemplateLoader()
