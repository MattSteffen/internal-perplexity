"""
Plugin registry for converter backends.

This module provides a registry system for dynamically registering and creating
converter instances by name, useful for plugin systems or dynamic configuration.
"""

from __future__ import annotations

from typing import Callable, Dict, Any
from .base import Converter


class _Registry:
    """Internal registry for converter builders."""
    
    def __init__(self) -> None:
        self._builders: Dict[str, Callable[..., Converter]] = {}

    def register(self, name: str, builder: Callable[..., Converter]) -> None:
        """
        Register a converter builder function.

        Args:
            name: Name to register the converter under
            builder: Function that creates a converter instance
        """
        self._builders[name.lower()] = builder

    def create(self, name: str, **kwargs: Any) -> Converter:
        """
        Create a converter instance by name.

        Args:
            name: Name of the converter to create
            **kwargs: Arguments to pass to the builder function

        Returns:
            Converter instance

        Raises:
            ValueError: If the converter name is not registered
        """
        key = name.lower()
        if key not in self._builders:
            raise ValueError(f"Unknown converter: {name}")
        return self._builders[key](**kwargs)

    def list_available(self) -> list[str]:
        """List all available converter names."""
        return list(self._builders.keys())


# Global registry instance
registry = _Registry()
