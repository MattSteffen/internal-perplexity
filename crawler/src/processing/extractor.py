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
        Chunk the text into smaller pieces.

        Args:
            text: Text to chunk
            chunk_size: Size of each chunk

        Returns:
            List of text chunks
        """
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i : i + chunk_size])
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

    def extract_metadata(self, text: str) -> Dict[str, Any]:
        """Extract metadata and chunks from the text."""
        if self.metadata_schema is not None:
            return self.llm.invoke(
                self._get_prompt(text), response_format=self.metadata_schema
            )
        else:
            return {}

    def _get_prompt(self, text: str) -> str:
        # replace the document context and document text in the prompt template
        return extract_metadata_prompt.replace(
            "{{document_library_context}}", self.document_library_context
        ).replace("{{document_text}}", text)


class MultiSchemaExtractor(Extractor):
    """
    Custom extractor class for processing documents with timeout capability.
    """

    def __init__(
        self, schemas: list[dict], llm: LLM, library_description: str = ""
    ) -> None:
        super().__init__()
        for schema in schemas:
            self.extractors.append(BasicExtractor(schema, llm, library_description))

    def extract_metadata(self, text: str) -> Dict[str, Any]:
        metadata = {}
        for extractor in self.extractors:
            metadata.update(extractor.extract_metadata(text))
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
