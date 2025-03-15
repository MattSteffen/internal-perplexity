import os
import yaml
from typing import Dict, Any, Optional, List
import logging
from pathlib import Path

class ConfigManager:
    """
    Configuration manager for the crawler application.
    Handles loading and merging of configuration files.
    """
    
    def __init__(self, config_dir: str = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_dir: Directory containing configuration files. If None, uses the default.
        """
        if config_dir is None:
            # Default to the config directory relative to this file
            self.config_dir = os.path.dirname(os.path.abspath(__file__))
        else:
            self.config_dir = config_dir
            
        self.base_config = {}
        self.collections = {}
        self.directory_configs = {}
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("ConfigManager")
        
        # Load base configuration
        self._load_base_config()
        
        # Load collection templates
        self._load_collection_templates()
        
        # Load directory configurations
        self._load_directory_configs()
        
    def _load_base_config(self):
        """Load the base configuration file."""
        base_config_path = os.path.join(self.config_dir, "base_config.yaml")
        try:
            with open(base_config_path, 'r') as f:
                self.base_config = yaml.safe_load(f)
            self.logger.info(f"Loaded base configuration from {base_config_path}")
        except Exception as e:
            self.logger.error(f"Failed to load base configuration: {e}")
            raise
            
    def _load_collection_templates(self):
        """Load collection template configurations."""
        # Load the default template
        template_path = os.path.join(self.config_dir, "collection_template.yaml")
        try:
            with open(template_path, 'r') as f:
                template = yaml.safe_load(f)
                self.collections[template.get('name', 'default')] = template
            self.logger.info(f"Loaded collection template from {template_path}")
        except Exception as e:
            self.logger.error(f"Failed to load collection template: {e}")
            
        # Load any additional collection configurations from collections directory
        collections_dir = os.path.join(self.config_dir, "collections")
        if os.path.exists(collections_dir):
            for filename in os.listdir(collections_dir):
                if filename.endswith('.yaml'):
                    try:
                        with open(os.path.join(collections_dir, filename), 'r') as f:
                            collection = yaml.safe_load(f)
                            name = collection.get('name', os.path.splitext(filename)[0])
                            self.collections[name] = collection
                        self.logger.info(f"Loaded collection config from {filename}")
                    except Exception as e:
                        self.logger.error(f"Failed to load collection config {filename}: {e}")
    
    def _load_directory_configs(self):
        """Load directory-specific configurations."""
        directories_dir = os.path.join(self.config_dir, "directories")
        if os.path.exists(directories_dir):
            for filename in os.listdir(directories_dir):
                if filename.endswith('.yaml'):
                    try:
                        with open(os.path.join(directories_dir, filename), 'r') as f:
                            dir_config = yaml.safe_load(f)
                            dir_name = os.path.splitext(filename)[0]
                            self.directory_configs[dir_name] = dir_config
                        self.logger.info(f"Loaded directory config from {filename}")
                    except Exception as e:
                        self.logger.error(f"Failed to load directory config {filename}: {e}")
    
    def _deep_merge(self, source: Dict[str, Any], destination: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries. Values from source override destination.
        Lists are appended (not replaced).
        
        Args:
            source: Source dictionary with values to merge.
            destination: Destination dictionary to merge into.
            
        Returns:
            Merged dictionary.
        """
        for key, value in source.items():
            if isinstance(value, dict):
                # Get node or create one
                node = destination.setdefault(key, {})
                if isinstance(node, dict):
                    self._deep_merge(value, node)
                else:
                    destination[key] = value
            elif isinstance(value, list) and key in destination and isinstance(destination[key], list):
                # Append lists instead of replacing
                destination[key].extend(value)
            else:
                destination[key] = value
        return destination
    
    def get_config_for_directory(self, directory_name: str) -> Dict[str, Any]:
        """
        Get the complete configuration for a specific directory.
        
        Args:
            directory_name: Name of the directory configuration to use.
            
        Returns:
            Complete configuration with base settings and directory-specific overrides.
        """
        if directory_name not in self.directory_configs:
            self.logger.warning(f"No configuration found for directory {directory_name}, using base config only")
            return self.base_config.copy()
            
        dir_config = self.directory_configs[directory_name]
        
        # Start with the base configuration
        config = self.base_config.copy()
        
        # Get the collection configuration
        collection_name = dir_config.get('collection', 'default')
        if collection_name in self.collections:
            collection_config = self.collections[collection_name].copy()
            
            # Apply collection overrides from directory config
            if 'collection_overrides' in dir_config:
                self._deep_merge(dir_config['collection_overrides'], collection_config)
                
            config['collection'] = collection_config
        else:
            self.logger.warning(f"Collection {collection_name} not found")
            
        # Apply directory-specific overrides to the base config
        for key, value in dir_config.items():
            if key not in ['collection', 'collection_overrides']:
                if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                    self._deep_merge(value, config[key])
                else:
                    config[key] = value
                    
        return config
    
    def get_all_directory_names(self) -> List[str]:
        """
        Get a list of all configured directory names.
        
        Returns:
            List of directory names.
        """
        return list(self.directory_configs.keys())
    
    def save_config(self, config: Dict[str, Any], config_path: str):
        """
        Save a configuration to a file.
        
        Args:
            config: Configuration dictionary to save.
            config_path: Path to save the configuration to.
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            self.logger.info(f"Saved configuration to {config_path}")
        except Exception as e:
            self.logger.error(f"Failed to save configuration to {config_path}: {e}")
            raise 