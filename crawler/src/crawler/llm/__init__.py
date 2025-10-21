"""
LLM (Large Language Model) package.

This package provides unified interfaces and implementations for various LLM providers
including Ollama and vLLM, with support for structured output and tool calling.
"""

from .llm import (
    LLM,
    LLMConfig,
    OllamaLLM,
    VllmLLM,
    get_llm,
    schema_to_openai_tools,
)

__all__ = [
    "LLM",
    "LLMConfig",
    "OllamaLLM",
    "VllmLLM",
    "get_llm",
    "schema_to_openai_tools",
]
