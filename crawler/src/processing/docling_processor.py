import logging
import os
from pathlib import Path

import requests
from dotenv import load_dotenv
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    ApiVlmOptions,
    ResponseFormat,
    VlmPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline

def ollama_vlm_options(model: str, prompt: str):
    options = ApiVlmOptions(
        params=dict(
            model=model,
        ),
        prompt=prompt,
        timeout=90,
        scale=1.0,
        response_format=ResponseFormat.MARKDOWN,
    )
    return options

def main():
    logging.basicConfig(level=logging.INFO)

    # input_doc_path = Path("./tests/data/pdf/2206.01062.pdf")
    input_doc_path = Path("/Users/mattsteffen/projects/llm/internal-perplexity/data/arxiv/2504.08710v1.pdf")

    pipeline_options = VlmPipelineOptions(
        enable_remote_services=True  # <-- this is required!
    )

    # The ApiVlmOptions() allows to interface with APIs supporting
    # the multi-modal chat interface. Here follow a few example on how to configure those.

    # One possibility is self-hosting model, e.g. via Ollama.
    # Example using the Granite Vision  model: (uncomment the following lines)
    pipeline_options.vlm_options = ollama_vlm_options(
        model="granite3.2-vision:2b",
        prompt="OCR the full page to markdown.",
    )

    # Create the DocumentConverter and launch the conversion.
    doc_converter = DocumentConverter(
        # format_options={
        #     InputFormat.PDF: PdfFormatOption(
        #         pipeline_options=pipeline_options,
        #         pipeline_cls=VlmPipeline,
        #     )
        # }
    )
    result = doc_converter.convert(input_doc_path)
    print(result.document.export_to_markdown())

if __name__ == '__main__':
    main()