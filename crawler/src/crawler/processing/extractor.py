import logging
import time
from typing import Dict, Any, List
from abc import ABC, abstractmethod
from .llm import LLM


class Extractor(ABC):
    """
    Abstract base class for document extractors.

    This class defines the interface for extracting metadata and chunking text
    from documents. All extractor implementations should inherit from this class.
    """

    def __init__(self):
        """
        Initialize the extractor with configuration.

        Args:
            config: Dictionary containing configuration options
        """
        pass

    @abstractmethod
    def extract_metadata(self, text: str) -> Dict[str, Any]:
        """
        Extract metadata from the given text.

        Args:
            text: Text to extract metadata from

        Returns:
            Dictionary containing extracted metadata
        """
        pass

    def chunk_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        """
        Chunk the text into smaller pieces with logging.

        Args:
            text: Text to chunk
            chunk_size: Size of each chunk

        Returns:
            List of text chunks
        """
        chunk_start_time = time.time()

        self.logger.info("✂️  Starting text chunking...")
        self.logger.debug(f"Input text length: {len(text)} characters")

        chunks = []
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i : i + chunk_size])

        chunk_time = time.time() - chunk_start_time

        # Calculate chunk statistics
        avg_chunk_size = sum(len(chunk) for chunk in chunks) / len(chunks) if chunks else 0
        total_chars = sum(len(chunk) for chunk in chunks)

        self.logger.info("✅ Text chunking completed successfully")
        self.logger.info("📊 Chunking Statistics:")
        self.logger.info(f"   • Total chunks created: {len(chunks)}")
        self.logger.info(f"   • Average chunk size: {avg_chunk_size:.0f} characters")
        self.logger.info(f"   • Total characters processed: {total_chars}")
        self.logger.info(f"   • Processing time: {chunk_time:.3f}s")
        self.logger.info(f"   • Chunking rate: {len(chunks)/chunk_time:.1f} chunks/sec")

        return chunks


class BasicExtractor(Extractor):
    def __init__(
        self,
        metadata_schema: Dict[str, Any],
        llm: LLM,
        document_library_context: str = "",
    ):
        super().__init__()
        self.metadata_schema = metadata_schema
        self.llm = llm
        self.document_library_context = document_library_context

        # Setup logging
        self.logger = logging.getLogger('Extractor')
        self.logger.propagate = False  # Prevent duplicate messages
        self.logger.info("Initialized BasicExtractor with schema and LLM")

    def extract_metadata(self, text: str) -> Dict[str, Any]:
        """Extract metadata from the text with comprehensive logging."""
        extract_start_time = time.time()

        if self.metadata_schema is not None:
            self.logger.info("🧠 Starting metadata extraction...")
            self.logger.debug(f"Input text length: {len(text)} characters")

            try:
                # Generate the prompt
                prompt = self._get_prompt(text)
                self.logger.debug(f"Generated prompt length: {len(prompt)} characters")

                # Log schema information
                required_fields = self.metadata_schema.get("required", [])
                self.logger.info(f"📋 Extracting {len(required_fields)} required fields: {required_fields}")

                # Make LLM call with timing
                llm_start_time = time.time()
                self.logger.info("🤖 Calling LLM for metadata extraction...")

                result = self.llm.invoke(
                    prompt, response_format=self.metadata_schema
                )

                llm_time = time.time() - llm_start_time
                total_time = time.time() - extract_start_time

                # Log results
                if isinstance(result, dict):
                    extracted_fields = list(result.keys())
                    self.logger.info("✅ Metadata extraction completed successfully")
                    self.logger.info("📊 Extraction Statistics:")
                    self.logger.info(f"   • LLM processing time: {llm_time:.2f}s")
                    self.logger.info(f"   • Total extraction time: {total_time:.2f}s")
                    self.logger.info(f"   • Fields extracted: {len(extracted_fields)}")
                    self.logger.info(f"   • Fields: {extracted_fields}")

                    # Check for missing required fields
                    missing_fields = [field for field in required_fields if field not in result or result[field] == "Unknown"]
                    if missing_fields:
                        self.logger.warning(f"⚠️  Missing required fields: {missing_fields}")

                    self.logger.debug(f"Extracted metadata: {result}")
                else:
                    self.logger.warning(f"⚠️  Unexpected result type: {type(result)}")

                return result

            except Exception as e:
                total_time = time.time() - extract_start_time
                self.logger.error(f"❌ Metadata extraction failed after {total_time:.2f}s: {e}")
                raise
        else:
            self.logger.info("ℹ️  No metadata schema provided, skipping extraction")
            return {}

    def _get_prompt(self, text: str) -> str:
        # replace the document context and document text in the prompt template
        return extract_metadata_prompt.replace(
            "{{document_library_context}}", self.document_library_context
        ).replace("{{document_text}}", text)


class MultiSchemaExtractor(Extractor):
    """
    Custom extractor class for processing documents with multiple schemas and timeout capability.
    """

    def __init__(
        self, schemas: list[dict], llm: LLM, library_description: str = ""
    ) -> None:
        super().__init__()
        self.extractors = []
        self.schemas = schemas
        self.llm = llm
        self.library_description = library_description

        # Setup logging
        self.logger = logging.getLogger('MultiSchemaExtractor')
        self.logger.propagate = False  # Prevent duplicate messages
        self.logger.info(f"Initialized MultiSchemaExtractor with {len(schemas)} schemas")

        # Create extractors
        for i, schema in enumerate(schemas):
            self.logger.debug(f"Creating extractor {i+1}/{len(schemas)}")
            extractor = BasicExtractor(schema, llm, library_description)
            self.extractors.append(extractor)

        self.logger.info("✅ All extractors initialized successfully")

    def extract_metadata(self, text: str) -> Dict[str, Any]:
        """Extract metadata using multiple schemas with comprehensive logging."""
        extract_start_time = time.time()

        self.logger.info(f"🧠 Starting multi-schema metadata extraction with {len(self.extractors)} extractors...")
        self.logger.debug(f"Input text length: {len(text)} characters")

        metadata = {}
        stats = {
            'total_extractors': len(self.extractors),
            'successful_extractions': 0,
            'failed_extractions': 0,
            'total_fields_extracted': 0
        }

        for i, extractor in enumerate(self.extractors):
            schema_name = f"Schema_{i+1}"
            self.logger.info(f"🔄 Processing {schema_name} ({i+1}/{len(self.extractors)})...")

            try:
                extractor_start = time.time()
                extractor_metadata = extractor.extract_metadata(text)
                extractor_time = time.time() - extractor_start

                if extractor_metadata:
                    fields_extracted = len(extractor_metadata)
                    metadata.update(extractor_metadata)
                    stats['successful_extractions'] += 1
                    stats['total_fields_extracted'] += fields_extracted

                    self.logger.info(f"✅ {schema_name} completed in {extractor_time:.2f}s - {fields_extracted} fields extracted")
                else:
                    self.logger.warning(f"⚠️  {schema_name} returned no metadata")

            except Exception as e:
                stats['failed_extractions'] += 1
                self.logger.error(f"❌ {schema_name} failed: {e}")

        total_time = time.time() - extract_start_time
        total_fields = len(metadata)

        self.logger.info("=== Multi-schema extraction completed ===")
        self.logger.info("📊 Multi-Schema Extraction Statistics:")
        self.logger.info(f"   • Total schemas processed: {stats['total_extractors']}")
        self.logger.info(f"   • Successful extractions: {stats['successful_extractions']}")
        self.logger.info(f"   • Failed extractions: {stats['failed_extractions']}")
        self.logger.info(f"   • Total unique fields extracted: {total_fields}")
        self.logger.info(f"   • Total processing time: {total_time:.2f}s")
        self.logger.info(f"   • Average time per schema: {total_time/stats['total_extractors']:.2f}s")

        if stats['failed_extractions'] > 0:
            self.logger.warning(f"⚠️  {stats['failed_extractions']} schema(s) failed to extract metadata")

        return metadata


extract_metadata_prompt = """
You are an expert metadata extraction engine. Your job is to read a Markdown document (converted from PDF, so formatting may vary), identify the required metadata fields, and output a JSON object conforming exactly to the JSON schema provided at runtime.

---

## How It Works

1. **Schema Injection**  
   Before processing, you will receive a JSON schema defining the exact fields, types, formats, and requirements.

2. **Document Context**  
   You may also receive background context about the document collection to help with ambiguous cases—but never output it. This describes the type of information present in the document.

3. **Extraction Process**  
   - **Scan the entire document** for metadata: author, title, dates, identifiers, etc.  
   - **Normalize values** (e.g. convert dates to `YYYY-MM-DD`, strip extra markup or artifacts).  
   - **Handle missing required fields** by setting their value to `"Unknown"`.  
   - **Validate** every extracted value against the schema: correct type, format, and presence of all `required` fields.  

4. **Output**  
   Emit **only** a JSON object (no commentary, no Markdown fences), matching the schema exactly.

---

## Example

> **Injected Schema:**  
> ```json
> {
>   "type": "object",
>   "properties": {
>     "author":      { "type": "string" },
>     "title":       { "type": "string" },
>     "pub_date":    { "type": "string", "format": "date" }
>   },
>   "required": ["author","title","pub_date"]
> }
> ```

> **Input Document:**  
> ```
> The Future of AI in Healthcare
> By Dr. Sarah Chen
> Published March 15, 2024
>
> Artificial intelligence is transforming medical diagnosis and treatment...
> ```

> **Expected Output:**  
> {
>   "author": "Dr. Sarah Chen",
>   "title":  "The Future of AI in Healthcare",
>   "pub_date": "2024-03-15"
> }

---

## Your Task

1. You will be given:
   - json_schema  
   - (optional) document_library_context  
   - document  

2. Extract and normalize metadata exactly as the schema demands.  
3. If any required field is missing, set it to `"Unknown"`.  
4. Output **only** the JSON object.  

Begin now.

**Document Library Context:**
{{document_library_context}}


**Document:**
{{document_text}}

"""
