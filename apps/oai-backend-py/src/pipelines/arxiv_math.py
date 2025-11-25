"""ArXiv mathematics paper processing pipeline configuration."""

from crawler import CrawlerConfig
from crawler.chunker import ChunkingConfig
from crawler.converter import PyMuPDF4LLMConfig
from crawler.extractor import MetadataExtractorConfig
from crawler.llm.embeddings import EmbedderConfig
from crawler.llm.llm import LLMConfig
from crawler.vector_db import DatabaseClientConfig

arxiv_math_library_description = (
    "You are processing a collection of academic mathematics papers from arXiv. "
    "These papers contain formal mathematical proofs, theorems, definitions, and research "
    "contributions in various areas of pure and applied mathematics. The documents include "
    "mathematical notation, equations, proofs, and technical content that requires precise "
    "understanding. Each paper represents original research contributions, theoretical "
    "developments, or applications of mathematical methods to solve problems in mathematics "
    "and related fields."
)

# Schema for ArXiv mathematics papers
arxiv_math_schema = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "ArXiv Mathematics Paper Metadata",
    "description": "Schema defining metadata for academic mathematics papers from arXiv.",
    "type": "object",
    "required": ["title", "authors", "year", "subject_class"],
    "properties": {
        "title": {
            "type": "string",
            "maxLength": 2000,
            "description": "The full title of the mathematics paper as it appears in the document.",
        },
        "authors": {
            "type": "array",
            "description": "List of all authors of the paper in the format 'Last, First' or 'First Last'.",
            "items": {
                "type": "string",
                "maxLength": 200,
                "description": "Author name in standard academic format.",
            },
            "minItems": 1,
        },
        "year": {
            "type": "integer",
            "description": "The year the paper was published or submitted to arXiv (four-digit year YYYY).",
            "minimum": 1990,
            "maximum": 2100,
        },
        "subject_class": {
            "type": "array",
            "description": "Mathematics Subject Classification (MSC) codes or arXiv subject classes (e.g., 'math.AT', 'math.CO', 'math.DG').",
            "items": {
                "type": "string",
                "maxLength": 50,
                "description": "A single subject classification code or category.",
            },
            "minItems": 1,
        },
        "keywords": {
            "type": "array",
            "description": "Key mathematical terms, concepts, or topics covered in the paper.",
            "items": {
                "type": "string",
                "maxLength": 200,
                "description": "A keyword or key phrase describing mathematical content.",
            },
            "minItems": 0,
        },
        "abstract_summary": {
            "type": "string",
            "maxLength": 5000,
            "description": "A concise summary of the paper's main contributions, key results, and mathematical content. Should capture the primary theorems, methods, or applications discussed.",
        },
    },
}


def create_arxiv_math_config() -> CrawlerConfig:
    """Create type-safe configuration for ArXiv mathematics paper processing.

    This configuration is optimized for processing academic mathematics papers
    with proper handling of mathematical notation and formal proofs.
    """
    # Embeddings configuration
    embeddings = EmbedderConfig.ollama(model="all-minilm:v2", base_url="http://localhost:11434")

    # Vision LLM for image processing (mathematical diagrams and figures)
    vision_llm = LLMConfig.ollama(model_name="gpt-oss:20b", base_url="http://localhost:11434")

    # Main LLM for metadata extraction
    llm = LLMConfig.ollama(
        model_name="gpt-oss:20b",
        base_url="http://localhost:11434",
        structured_output="tools",
    )

    # Database configuration
    database = DatabaseClientConfig.milvus(
        collection="arxiv_math_documents",
        host="localhost",
        port=19530,
        username="root",
        password="Milvus",
        recreate=False,  # Don't recreate collection on each upload
    )

    # Metadata extractor configuration
    extractor = MetadataExtractorConfig(json_schema=arxiv_math_schema, context=arxiv_math_library_description)

    # PyMuPDF4LLM converter configuration with image processing
    converter = PyMuPDF4LLMConfig(
        type="pymupdf4llm",
        vlm_config=vision_llm,
        image_prompt="Describe this mathematical diagram, figure, or graph in detail, including all mathematical notation, labels, and relationships shown.",
        max_workers=4,
        to_markdown_kwargs={
            "page_chunks": False,
            "embed_images": True,
        },
    )

    # Chunking configuration - larger chunks for mathematical papers
    chunking = ChunkingConfig.create(chunk_size=1500)

    # Create the complete crawler configuration
    config = CrawlerConfig.create(
        embeddings=embeddings,
        llm=llm,
        vision_llm=vision_llm,
        database=database,
        converter=converter,
        extractor=extractor,
        chunking=chunking,
        metadata_schema=arxiv_math_schema,
        temp_dir="/tmp/arxiv_math",
        benchmark=False,
    )

    return config

