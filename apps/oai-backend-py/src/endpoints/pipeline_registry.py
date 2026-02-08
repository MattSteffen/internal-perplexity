"""Pipeline registry for predefined crawler-config JSONs.

Pipelines are name-to-JSON maps only. No CrawlerConfig creation or schema
validation in this module. Call sites get JSON via get(name) and build
a CrawlerConfig with CrawlerConfig.from_dict(json).
"""

import copy
import json
from typing import Any

# --- Pipeline JSON (crawler-config shape, one-time generated from template model_dump) ---

_STANDARD_JSON = """
{
  "name": "standard",
  "embeddings": {
    "model": "qwen3-embedding:0.6b",
    "base_url": "http://localhost:11434",
    "api_key": "",
    "provider": "ollama",
    "dimension": null
  },
  "llm": {
    "model_name": "gpt-oss:20b",
    "base_url": "http://localhost:11434",
    "system_prompt": null,
    "ctx_length": 32000,
    "default_timeout": 300.0,
    "provider": "ollama",
    "api_key": "",
    "structured_output": "tools"
  },
  "vision_llm": {
    "model_name": "qwen3-vl:2b",
    "base_url": "http://localhost:11434",
    "system_prompt": null,
    "ctx_length": 32000,
    "default_timeout": 300.0,
    "provider": "ollama",
    "api_key": "",
    "structured_output": "response_format"
  },
  "database": {
    "provider": "milvus",
    "collection": "placeholder",
    "partition": null,
    "access_level": "public",
    "recreate": false,
    "collection_description": "Standard document collection",
    "host": "localhost",
    "port": 19530,
    "username": "placeholder",
    "password": "placeholder"
  },
  "converter": {
    "type": "pymupdf4llm",
    "vlm_config": {
      "model_name": "qwen3-vl:2b",
      "base_url": "http://localhost:11434",
      "system_prompt": null,
      "ctx_length": 32000,
      "default_timeout": 300.0,
      "provider": "ollama",
      "api_key": "",
      "structured_output": "response_format"
    },
    "image_prompt": "Describe this image in detail. Focus on the main content, objects, text, and any relevant information useful in a document context.",
    "max_workers": 2,
    "to_markdown_kwargs": {}
  },
  "extractor": {
    "json_schema": {
      "type": "object",
      "required": ["title", "authors", "year", "keywords"],
      "properties": {
        "title": {"type": "string", "maxLength": 500, "description": "Document title."},
        "authors": {"type": "array", "description": "List of authors or contributors.", "items": {"type": "string", "maxLength": 255}, "minItems": 1},
        "year": {"type": "integer", "description": "Publication year for filtering and sorting.", "minimum": 1900, "maximum": 2100},
        "document_type": {"type": "string", "enum": ["report", "article", "book", "whitepaper", "manual", "presentation", "other"], "description": "Broad document category for filtering."},
        "categories": {"type": "array", "description": "High-level subject categories.", "items": {"type": "string", "maxLength": 100}},
        "keywords": {"type": "array", "description": "Searchable keywords describing content.", "items": {"type": "string", "maxLength": 100}},
        "description": {"type": "string", "maxLength": 5000, "description": "Brief summary or abstract."}
      }
    },
    "context": "General document collection",
    "structured_output": "json_schema",
    "include_benchmark_questions": false,
    "num_benchmark_questions": 3,
    "truncate_document_chars": 4000,
    "strict": true
  },
  "chunking": {
    "chunk_size": 2000,
    "overlap": 200,
    "strategy": "naive",
    "preserve_paragraphs": true,
    "min_chunk_size": 100
  },
  "metadata_schema": {
    "type": "object",
    "required": ["title", "authors", "year", "keywords"],
    "properties": {
      "title": {"type": "string", "maxLength": 500, "description": "Document title."},
      "authors": {"type": "array", "description": "List of authors or contributors.", "items": {"type": "string", "maxLength": 255}, "minItems": 1},
      "year": {"type": "integer", "description": "Publication year for filtering and sorting.", "minimum": 1900, "maximum": 2100},
      "document_type": {"type": "string", "enum": ["report", "article", "book", "whitepaper", "manual", "presentation", "other"], "description": "Broad document category for filtering."},
      "categories": {"type": "array", "description": "High-level subject categories.", "items": {"type": "string", "maxLength": 100}},
      "keywords": {"type": "array", "description": "Searchable keywords describing content.", "items": {"type": "string", "maxLength": 100}},
      "description": {"type": "string", "maxLength": 5000, "description": "Brief summary or abstract."}
    }
  },
  "temp_dir": "tmp/",
  "use_cache": true,
  "benchmark": false,
  "generate_benchmark_questions": false,
  "num_benchmark_questions": 3,
  "security_groups": ["public"]
}
"""

_ACADEMIC_JSON = """
{
  "name": "academic",
  "embeddings": {
    "model": "qwen3-embedding:0.6b",
    "base_url": "http://localhost:11434",
    "api_key": "",
    "provider": "ollama",
    "dimension": null
  },
  "llm": {
    "model_name": "gpt-oss:20b",
    "base_url": "http://localhost:11434",
    "system_prompt": null,
    "ctx_length": 32000,
    "default_timeout": 300.0,
    "provider": "ollama",
    "api_key": "",
    "structured_output": "tools"
  },
  "vision_llm": {
    "model_name": "qwen3-vl:2b",
    "base_url": "http://localhost:11434",
    "system_prompt": null,
    "ctx_length": 32000,
    "default_timeout": 300.0,
    "provider": "ollama",
    "api_key": "",
    "structured_output": "response_format"
  },
  "database": {
    "provider": "milvus",
    "collection": "placeholder",
    "partition": null,
    "access_level": "public",
    "recreate": false,
    "collection_description": "Academic research paper collection",
    "host": "localhost",
    "port": 19530,
    "username": "placeholder",
    "password": "placeholder"
  },
  "converter": {
    "type": "pymupdf4llm",
    "vlm_config": {
      "model_name": "qwen3-vl:2b",
      "base_url": "http://localhost:11434",
      "system_prompt": null,
      "ctx_length": 32000,
      "default_timeout": 300.0,
      "provider": "ollama",
      "api_key": "",
      "structured_output": "response_format"
    },
    "image_prompt": "Describe this image in detail. Focus on the main content, objects, text, and any relevant information useful in a document context.",
    "max_workers": 2,
    "to_markdown_kwargs": {}
  },
  "extractor": {
    "json_schema": {
      "type": "object",
      "required": ["title", "authors", "year", "keywords", "domain_terms"],
      "properties": {
        "title": {"type": "string", "maxLength": 500, "description": "Document title."},
        "authors": {"type": "array", "description": "List of authors.", "items": {"type": "string", "maxLength": 255}, "minItems": 1},
        "year": {"type": "integer", "description": "Publication year.", "minimum": 1900, "maximum": 2100},
        "publication_date": {"type": "string", "format": "date", "description": "Full publication date if available (ISO 8601)."},
        "doi": {"type": "string", "pattern": "^10\\\\.[0-9]{4,}/.+$", "description": "Digital Object Identifier."},
        "document_type": {
          "type": "string",
          "enum": ["journal_article", "conference_paper", "preprint", "thesis", "book_chapter", "technical_report", "review"],
          "description": "Academic document type for filtering."
        },
        "venue": {"type": "string", "maxLength": 500, "description": "Journal name, conference name, or publisher."},
        "language": {"type": "string", "pattern": "^[a-z]{2}$", "description": "ISO 639-1 language code."},
        "subject_areas": {"type": "array", "description": "Academic disciplines.", "items": {"type": "string", "maxLength": 100}},
        "keywords": {"type": "array", "description": "General academic keywords.", "items": {"type": "string", "maxLength": 100}},
        "domain_terms": {"type": "array", "description": "Technical or domain-specific terminology for precise search.", "items": {"type": "string", "maxLength": 100}},
        "abstract": {"type": "string", "maxLength": 15000, "description": "Full abstract."},
        "key_findings": {"type": "array", "description": "Specific findings or contributions (optional).", "items": {"type": "string", "maxLength": 2000}, "maxItems": 5}
      }
    },
    "context": "Academic research papers and publications",
    "structured_output": "json_schema",
    "include_benchmark_questions": false,
    "num_benchmark_questions": 3,
    "truncate_document_chars": 4000,
    "strict": true
  },
  "chunking": {
    "chunk_size": 10000,
    "overlap": 500,
    "strategy": "naive",
    "preserve_paragraphs": true,
    "min_chunk_size": 100
  },
  "metadata_schema": {
    "type": "object",
    "required": ["title", "authors", "year", "keywords", "domain_terms"],
    "properties": {
      "title": {"type": "string", "maxLength": 500, "description": "Document title."},
      "authors": {"type": "array", "description": "List of authors.", "items": {"type": "string", "maxLength": 255}, "minItems": 1},
      "year": {"type": "integer", "description": "Publication year.", "minimum": 1900, "maximum": 2100},
      "publication_date": {"type": "string", "format": "date", "description": "Full publication date if available (ISO 8601)."},
      "document_type": {
        "type": "string",
        "enum": ["journal_article", "conference_paper", "preprint", "thesis", "book_chapter", "technical_report", "review"],
        "description": "Academic document type for filtering."
      },
      "venue": {"type": "string", "maxLength": 500, "description": "Journal name, conference name, or publisher."},
      "language": {"type": "string", "pattern": "^[a-z]{2}$", "description": "ISO 639-1 language code."},
      "subject_areas": {"type": "array", "description": "Academic disciplines.", "items": {"type": "string", "maxLength": 100}},
      "keywords": {"type": "array", "description": "General academic keywords.", "items": {"type": "string", "maxLength": 100}},
      "domain_terms": {"type": "array", "description": "Technical or domain-specific terminology for precise search.", "items": {"type": "string", "maxLength": 100}},
      "abstract": {"type": "string", "maxLength": 15000, "description": "Full abstract."},
      "key_findings": {"type": "array", "description": "Specific findings or contributions (optional).", "items": {"type": "string", "maxLength": 2000}, "maxItems": 5}
    }
  },
  "temp_dir": "tmp/",
  "use_cache": true,
  "benchmark": false,
  "generate_benchmark_questions": true,
  "num_benchmark_questions": 3,
  "security_groups": ["public"]
}
"""

STANDARD_PIPELINE_JSON: dict[str, Any] = json.loads(_STANDARD_JSON)
ACADEMIC_PIPELINE_JSON: dict[str, Any] = json.loads(_ACADEMIC_JSON)


# --- Pipeline Registry ---


class PipelineRegistry:
    """Registry of pipeline names to crawler-config-shaped JSON. No validation."""

    def __init__(self) -> None:
        self._pipelines: dict[str, dict[str, Any]] = {}

    def register(self, name: str, config_dict: dict[str, Any]) -> None:
        """Register a pipeline by name with its full config JSON (no validation)."""
        if name in self._pipelines:
            raise ValueError(f"Pipeline '{name}' is already registered")
        self._pipelines[name] = config_dict

    def get(self, name: str) -> dict[str, Any]:
        """Return a copy of the pipeline JSON for the given name.

        Raises:
            KeyError: If pipeline name is not found.
        """
        if name not in self._pipelines:
            raise KeyError(f"Pipeline '{name}' not found. Available: {list(self._pipelines.keys())}")
        return copy.deepcopy(self._pipelines[name])

    def list_pipelines(self) -> list[str]:
        """List all registered pipeline names."""
        return list(self._pipelines.keys())

    def has_pipeline(self, name: str) -> bool:
        """Return True if the pipeline is registered."""
        return name in self._pipelines

    def get_pipeline_info(self) -> list[dict[str, Any]]:
        """Best-effort pipeline info from stored JSON (no validation). Same shape as before for UI."""
        info = []
        for name in self._pipelines:
            d = self._pipelines[name]
            db = d.get("database") or {}
            chunking = d.get("chunking") or {}
            embeddings = d.get("embeddings") or {}
            llm = d.get("llm") or {}
            info.append(
                {
                    "name": d.get("name", name),
                    "description": db.get("collection_description", "") or "",
                    "metadata_schema": d.get("metadata_schema", {}),
                    "chunk_size": chunking.get("chunk_size", 0),
                    "embedding_model": embeddings.get("model", "") or "",
                    "llm_model": llm.get("model_name", "") or "",
                }
            )
        info.sort(key=lambda x: x["name"])
        return info


# --- Global registry ---

_pipeline_registry = PipelineRegistry()
_pipeline_registry.register("standard", STANDARD_PIPELINE_JSON)
_pipeline_registry.register("academic", ACADEMIC_PIPELINE_JSON)


def get_registry() -> PipelineRegistry:
    """Return the global pipeline registry instance."""
    return _pipeline_registry
