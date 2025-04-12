import yaml
from typing import Dict, Any, Optional, List

# TODO: Add a validator for the config files
# Validation includes: If milvus is enabled, it should have the username and password
# If milvus is enabled, it should test the connection, then check the schema with the collection name and schema, embedding shape should match too
# Warning if in schema there are properties text and embedding saying they will not be used.
# If embeddings are enabled, it should have the model name etc 
# extra embeddings should be fields in the schema
class ConfigManager:
    """
    Configuration manager that handles loading and accessing configuration settings.
    Can be initialized with either a config file path or a config dictionary.
    """

    def __init__(
        self,
        config_source: str|Dict[str, any] = "",
        base_config_path: str = "",
        collection_template_path: str = "",
    ):
        """
        Initialize the configuration manager.

        Args:
            base_config_path: Path to the base configuration file
            collection_template_path: Path to the collection template configuration file
            config_source: Either a path to a YAML config file or a config dictionary
        """
        import os

        # Get the absolute path to the directory where the current file is located
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Define your config paths relative to this directory
        base_config_path = os.path.join(base_dir, 'base_config.yaml') if base_config_path == "" else base_config_path
        collection_template_path = os.path.join(base_dir, 'collection_template.yaml') if collection_template_path == "" else collection_template_path

        self.base_config = self._load_yaml(base_config_path)
        self.collection_template = self._load_yaml(collection_template_path)
        
        if isinstance(config_source, str):
            self.config = self._load_yaml(config_source)
        else:
            self.config = config_source
        
        # Merge the base config, collection template, and config
        self.config = self.deep_merge_dicts(
            self.deep_merge_dicts(self.base_config, self.collection_template),
            self.config,
        )
        
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


# # Example usage
# if __name__ == "__main__":
#     # Example paths
#     base_config_path = "base_config.yaml"
#     collection_template_path = "collection_template.yaml"
#     collections_config_dir = "directories/sample.yaml"
    
#     # Initialize the configuration manager
#     config_manager = ConfigManager(
#         base_config_path,
#         collection_template_path,
#         collections_config_dir
#     )

#     import json
#     print(json.dumps(config_manager.config))