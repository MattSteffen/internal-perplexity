import os
import yaml
from typing import Dict, Any, Optional, List

# TODO: Add a validator for the config files
# TODO: Add ability to load a config file and place it in the directories section, this way the user can pass in an override config file (add_config(path, save=False))
# and when save = true it'll save it to the collections_config_dir
class ConfigManager:
    """
    Configuration manager for file indexer application that handles multi-level YAML configurations.
    Handles base config, collection template config, and directory-specific collection configs.
    """

    def __init__(
        self,
        base_config_path: str,
        collection_template_path: str,
        collections_config_dir: str
    ):
        """
        Initialize the configuration manager.

        Args:
            base_config_path: Path to the base configuration file
            collection_template_path: Path to the collection template configuration file
            collections_config_dir: Directory containing collection-specific configuration files
        """
        self.base_config = self._load_yaml(base_config_path)
        self.collection_template = self._load_yaml(collection_template_path)
        self.collections_config_dir = collections_config_dir
        self.collection_configs = {}
        
        # Load all collection configuration files
        self._load_collection_configs()

    def _load_yaml(self, file_path: str) -> Dict[str, Any]:
        """
        Load a YAML file and return the parsed content.

        Args:
            file_path: Path to the YAML file

        Returns:
            Dictionary containing the YAML file content
        """
        try:
            with open(file_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Error loading YAML file {file_path}: {e}")
            return {}

    def _load_collection_configs(self) -> None:
        """
        Load all collection configuration files from the specified directory.
        Organizes them by directory path for quick lookup.
        """
        try:
            for filename in os.listdir(self.collections_config_dir):
                if filename.endswith('.yaml') or filename.endswith('.yml'):
                    file_path = os.path.join(self.collections_config_dir, filename)
                    config = self._load_yaml(file_path)
                    
                    # Only add configs that have a valid path
                    if config and 'path' in config:
                        dir_path = config['path']
                        self.collection_configs[dir_path] = config
        except Exception as e:
            print(f"Error loading collection configs: {e}")

    def deep_merge_dicts(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merge two dictionaries, with values from dict2 taking precedence.
        
        Args:
            dict1: Base dictionary
            dict2: Dictionary with values that will override dict1

        Returns:
            A new dictionary with merged values
        """
        result = dict1.copy()
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                result[key] = self.deep_merge_dicts(result[key], value)
            else:
                # Override or add the value
                result[key] = value
                
        return result

    def get_config(self, dir_path: str) -> Dict[str, Any]:
        """
        Get the merged configuration for a specific directory path.
        
        Args:
            dir_path: The directory path to get configuration for

        Returns:
            A merged configuration dictionary
        """
        # Start with the base configuration
        result = self.base_config.copy()
        
        # Merge with collection template
        result = self.deep_merge_dicts(result, self.collection_template)
        
        # If we have a specific configuration for this directory, merge it in
        if dir_path in self.collection_configs:
            result = self.deep_merge_dicts(result, self.collection_configs[dir_path])
        
        return result

    def get_all_collection_paths(self) -> List[str]:
        """
        Get a list of all configured collection paths.
        
        Returns:
            List of directory paths
        """
        return list(self.collection_configs.keys())
    
    def reload_configs(self) -> None:
        """
        Reload all configuration files.
        Useful when configurations have been updated.
        """
        self.collection_configs = {}
        self._load_collection_configs()


# Example usage
if __name__ == "__main__":
    # Example paths
    base_config_path = "base_config.yaml"
    collection_template_path = "collection_template.yaml"
    collections_config_dir = "directories"
    
    # Initialize the configuration manager
    config_manager = ConfigManager(
        base_config_path,
        collection_template_path,
        collections_config_dir
    )

    print(config_manager.get_config("../../data/conference"))