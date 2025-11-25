"""Pipeline registry for managing predefined crawler configurations."""

from collections.abc import Callable

from crawler import CrawlerConfig


class PipelineRegistry:
    """Registry for managing predefined document processing pipelines.

    Pipelines are registered by name with factory functions that return
    CrawlerConfig instances. This allows for easy retrieval and extension
    of predefined processing configurations.
    """

    def __init__(self) -> None:
        """Initialize an empty pipeline registry."""
        self._pipelines: dict[str, Callable[[], CrawlerConfig]] = {}

    def register(self, name: str, config_factory: Callable[[], CrawlerConfig]) -> None:
        """Register a pipeline configuration factory.

        Args:
            name: Unique name identifier for the pipeline
            config_factory: Function that returns a CrawlerConfig instance

        Raises:
            ValueError: If pipeline name already exists
        """
        if name in self._pipelines:
            raise ValueError(f"Pipeline '{name}' is already registered")
        self._pipelines[name] = config_factory

    def get_config(self, name: str) -> CrawlerConfig:
        """Get a pipeline configuration by name.

        Args:
            name: Pipeline name identifier

        Returns:
            CrawlerConfig instance for the pipeline

        Raises:
            KeyError: If pipeline name is not found
        """
        if name not in self._pipelines:
            raise KeyError(f"Pipeline '{name}' not found")
        return self._pipelines[name]()

    def list_pipelines(self) -> list[str]:
        """List all registered pipeline names.

        Returns:
            List of pipeline name strings
        """
        return list[str](self._pipelines.keys())

    def has_pipeline(self, name: str) -> bool:
        """Check if a pipeline is registered.

        Args:
            name: Pipeline name identifier

        Returns:
            True if pipeline exists, False otherwise
        """
        return name in self._pipelines


# Global registry instance
_pipeline_registry = PipelineRegistry()


def get_registry() -> PipelineRegistry:
    """Get the global pipeline registry instance.

    Returns:
        The global PipelineRegistry instance
    """
    return _pipeline_registry
