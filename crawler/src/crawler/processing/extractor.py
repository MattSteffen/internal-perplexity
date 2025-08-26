import logging
import time
import json
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
        generate_benchmark_questions: bool = False,
        num_benchmark_questions: int = 3,
    ):
        super().__init__()
        self.metadata_schema = metadata_schema
        self.llm = llm
        self.document_library_context = document_library_context
        self.generate_benchmark_questions = generate_benchmark_questions
        self.num_benchmark_questions = num_benchmark_questions

        # Get logger (already configured by main crawler)
        self.logger = logging.getLogger('Extractor')
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

                # Generate benchmark questions if enabled
                if self.generate_benchmark_questions:
                    self.logger.info(f"📝 Generating {self.num_benchmark_questions} benchmark questions...")
                    try:
                        questions = generate_benchmark_questions(
                            self.llm, text, self.num_benchmark_questions
                        )
                        if questions:
                            result["benchmark_questions"] = questions
                            self.logger.info(f"✅ Generated {len(questions)} benchmark questions")
                        else:
                            self.logger.warning("⚠️  No benchmark questions generated")
                    except Exception as e:
                        self.logger.error(f"❌ Failed to generate benchmark questions: {e}")

                return result

            except Exception as e:
                total_time = time.time() - extract_start_time
                self.logger.error(f"❌ Metadata extraction failed after {total_time:.2f}s: {e}")
                raise
        else:
            self.logger.info("ℹ️  No metadata schema provided, skipping extraction")
            return {}

    def _get_prompt(self, text: str) -> str:
        # Serialize schema to JSON string
        schema_json = json.dumps(self.metadata_schema, ensure_ascii=False, indent=2)

        # Inject schema, context, and document into the prompt template
        return extract_metadata_prompt.replace(
            "{{json_schema}}", schema_json
        ).replace(
            "{{document_library_context}}", self.document_library_context or ""
        ).replace(
            "{{document_text}}", text
        )


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

        # Get logger (already configured by main crawler)
        self.logger = logging.getLogger('MultiSchemaExtractor')
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


def generate_benchmark_questions(llm: LLM, text: str, n: int) -> List[str]:
    """
    Generate benchmark questions for a document.

    Args:
        llm: LLM instance to use for generation
        text: Document text to generate questions from
        n: Number of questions to generate

    Returns:
        List of benchmark questions as strings
    """
    prompt = f"""You are an expert at creating benchmark questions for document retrieval systems.

Given the following document text, generate exactly {n} diverse questions that could be answered by this document. Each question should:
- Be answerable using information from the document
- Cover different aspects of the document content
- Be specific and unambiguous
- Vary in complexity and topic coverage

Respond with a JSON array of exactly {n} strings, containing only the questions.

Document text:
{text[:4000]}...  # Truncate for token limits

Questions:"""

    try:
        response = llm.invoke(prompt)
        if isinstance(response, str):
            # Try to parse as JSON
            import json
            try:
                questions = json.loads(response.strip())
                if isinstance(questions, list) and len(questions) == n:
                    return questions
                else:
                    logging.warning(f"Expected {n} questions, got {len(questions) if isinstance(questions, list) else 'non-list'}")
                    return []
            except json.JSONDecodeError:
                # Fallback: extract questions from text response
                lines = [line.strip() for line in response.split('\n') if line.strip()]
                questions = [line for line in lines if line.endswith('?')]
                return questions[:n]
        else:
            logging.warning("Unexpected response type from LLM")
            return []
    except Exception as e:
        logging.error(f"Error generating benchmark questions: {e}")
        return []

extract_metadata_prompt = """
You are an expert metadata extraction engine. Read the document and output a
single JSON object that conforms EXACTLY to the injected JSON Schema.

STRICT RULES:
- Use ONLY the keys defined in the injected schema's properties.
- Include ALL fields required by the schema.
- Do NOT invent or copy fields that are not in the schema.
- Normalize values (e.g., convert dates to YYYY-MM-DD, strip markup/artifacts).
- If a required field is missing or cannot be inferred, set its value to
  "Unknown".
- Validate each value against the schema types/formats.
- Output must be ONLY a JSON object (no extra text, no markdown fences).

JSON Schema:
[[SCHEMA_START]]
{{json_schema}}
[[SCHEMA_END]]

Document Library Context (do not echo in output; use only for disambiguation):
[[CONTEXT_START]]
{{document_library_context}}
[[CONTEXT_END]]

Document:
[[DOCUMENT_START]]
{{document_text}}
[[DOCUMENT_END]]
"""