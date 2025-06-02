# Document Processing System

## TODO

- [ ] Make it so that non-required parts of the schema don't cause it to fail as metadata extracted from llm

This module provides a framework for extracting text and metadata from various document formats, processing the content with LLMs, and generating embeddings for search and retrieval.

## Overview

The system consists of three main components:

1. **Processors** - Extract content from various file formats (PDF, TXT, HTML, etc.)
2. **Extractors** - Process document content with LLMs to extract structured metadata
3. **Embeddings** - Generate vector embeddings for document content
4. **LLM** - An interface allowing rest requests to a provided api (supporting only ollama currently)

## Convert

- Take a document and convert to markdown

## Metadata

- Extract metadata from markdown

## Embeddings

- Generate embeddings for each chunk

## The rest

- Chunk the markdown into smaller chunks
- Generate embeddings for each chunk
- Return list dict of entities for milvus
