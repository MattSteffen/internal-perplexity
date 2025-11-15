import json
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from ..document import Document
from ..llm import LLM, schema_to_openai_tools


class MetadataExtractorConfig(BaseModel):
    """
    Configuration for a single-schema metadata extractor.

    Fields:
      - schema: JSON Schema (type=object) defining the metadata to extract.
      - context: Optional library/project context for disambiguation.
      - structured_output:
          "json_schema" (default) -> send schema via response_format
          "tools" -> send schema as OpenAI-style tools/functions
      - include_benchmark_questions: If True, also generate benchmark questions.
      - num_benchmark_questions: Number of questions to generate when enabled.
      - truncate_document_chars: Hard cap on document size for prompts.
      - strict:
          If True, drop extra keys and backfill missing required as "Unknown".
          When False, leave model output intact (still validated if possible).
    """

    json_schema: dict[str, Any] = Field(
        ...,
        description="JSON Schema (object) for the metadata to extract",
    )
    context: str = Field(
        default="",
        description="Optional library/project context for disambiguation",
    )
    structured_output: Literal["json_schema", "tools"] = Field(
        default="json_schema",
        description="How to request structured output from the LLM",
    )
    include_benchmark_questions: bool = Field(default=False, description="Also generate benchmark questions")
    num_benchmark_questions: int = Field(default=3, ge=1, le=20, description="Number of benchmark questions")
    truncate_document_chars: int = Field(
        default=4000,
        ge=512,
        le=32000,
        description="Max characters of document text injected into prompts",
    )
    strict: bool = Field(
        default=True,
        description=("If True, drop extra keys and fill missing required as 'Unknown'. " "Attempts jsonschema validation when available."),
    )

    model_config = {"validate_assignment": True}

    @field_validator("json_schema")
    @classmethod
    def validate_json_schema(cls, v: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(v, dict):
            raise ValueError("schema must be a JSON object")
        if v.get("type") != "object":
            raise ValueError("schema.type must be 'object'")
        if "properties" not in v or not isinstance(v["properties"], dict):
            raise ValueError("schema.properties must be a JSON object")
        # Normalize required
        if "required" in v and not isinstance(v["required"], list):
            raise ValueError("schema.required must be a list when present")
        v.setdefault("required", [])
        return v

    @property
    def required_fields(self) -> list[str]:
        return list(self.json_schema.get("required", []))

    @property
    def allowed_keys(self) -> list[str]:
        return list(self.json_schema["properties"].keys())


class MetadataExtractionResult(BaseModel):
    metadata: dict[str, Any]
    benchmark_questions: list[str] | None = None


class MetadataExtractor:
    """
    Single-schema metadata extractor.

    Primary operations:
      - extract(markdown) -> Dict[str, Any] (schema-conformant metadata)
      - run(markdown) -> MetadataExtractionResult
        (metadata + optional benchmark questions)
    """

    def __init__(self, llm: LLM, config: MetadataExtractorConfig) -> None:
        self.llm = llm
        self.config = config

    def run(self, document: Document) -> MetadataExtractionResult:
        metadata = self.extract(document.markdown)

        questions: list[str] | None = None
        if self.config.include_benchmark_questions:
            questions = self.generate_benchmark_questions(document.markdown, self.config.num_benchmark_questions)

        return MetadataExtractionResult(metadata=metadata, benchmark_questions=questions)

    def extract(self, markdown: str) -> dict[str, Any]:
        """
        Returns a dict valid under the configured JSON Schema.

        - Uses either response_format (json_schema) or tools structured output.
        - Strict mode post-processes:
            - Drop extra keys not in schema.properties
            - Fill missing required keys with 'Unknown'
        - Attempts jsonschema validation if installed (optional).
        """
        doc = markdown[: self.config.truncate_document_chars]
        prompt = self._build_metadata_prompt(
            schema=self.config.json_schema,
            context=self.config.context,
            document_text=doc,
        )

        result: Any
        if self.config.structured_output == "tools":
            tools = schema_to_openai_tools(self.config.json_schema)
            result = self.llm.invoke(prompt, tools=tools)
        else:
            result = self.llm.invoke(prompt, response_format=self.config.json_schema)

        metadata = self._coerce_to_dict(result)

        if self.config.strict:
            metadata = self._enforce_schema(metadata, self.config.json_schema)

        self._maybe_validate_with_jsonschema(metadata, self.config.json_schema)
        return metadata

    def generate_benchmark_questions(self, markdown: str, n: int) -> list[str]:
        """
        Generate n benchmark questions grounded in the provided document.
        Returns exactly n questions when possible; may return fewer if the
        model response is not parseable.
        """
        doc = markdown[: self.config.truncate_document_chars]
        prompt = (
            "You are an expert at creating benchmark questions for document "
            "retrieval systems.\n\n"
            f"Given the following document text, generate exactly {n} diverse "
            "questions that could be answered by this document. Each question "
            "should:\n"
            "- Be answerable using information from the document\n"
            "- Cover different aspects of the document content\n"
            "- Be specific and unambiguous\n"
            "- Vary in complexity and topic coverage\n\n"
            f"Respond with a JSON array of exactly {n} strings, containing "
            "only the questions.\n\n"
            "Document text:\n"
            f"{doc}\n\n"
            "Questions:"
        )

        try:
            response = self.llm.invoke(prompt)
            if isinstance(response, str):
                try:
                    arr = json.loads(response.strip())
                    if isinstance(arr, list):
                        return [str(q) for q in arr][:n]
                except json.JSONDecodeError:
                    pass
                # Fallback: split lines and pick questions
                lines = [ln.strip() for ln in response.split("\n") if ln.strip()]
                qs = [ln for ln in lines if ln.endswith("?")]
                return qs[:n]
            elif isinstance(response, list):
                return [str(q) for q in response][:n]
            else:
                # Some LLM wrappers may return dict with 'text'
                if isinstance(response, dict) and "text" in response:
                    try:
                        arr = json.loads(str(response["text"]).strip())
                        if isinstance(arr, list):
                            return [str(q) for q in arr][:n]
                    except Exception:
                        pass
                return []
        except Exception:
            return []

    def _build_metadata_prompt(self, schema: dict[str, Any], context: str, document_text: str) -> str:
        schema_json = json.dumps(schema, ensure_ascii=False, indent=2)
        return EXTRACT_METADATA_PROMPT.replace("{{json_schema}}", schema_json).replace("{{document_library_context}}", context or "").replace("{{document_text}}", document_text)

    def _coerce_to_dict(self, result: Any) -> dict[str, Any]:
        if isinstance(result, dict):
            return result
        if isinstance(result, str):
            try:
                return json.loads(result.strip())
            except json.JSONDecodeError:
                # Try to recover JSON object from text
                start = result.find("{")
                end = result.rfind("}")
                if start != -1 and end != -1 and end > start:
                    try:
                        return json.loads(result[start : end + 1])
                    except Exception:
                        pass
                raise ValueError("LLM did not return parseable JSON object")
        # Some wrappers may return objects with .content or .text
        if hasattr(result, "content"):
            try:
                return json.loads(str(result.content).strip())
            except Exception:
                pass
        if hasattr(result, "text"):
            try:
                return json.loads(str(result.text).strip())
            except Exception:
                pass
        raise ValueError("Unsupported LLM response type for metadata extraction")

    def _enforce_schema(self, data: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
        props = schema.get("properties", {})
        allowed = set(props.keys())
        required = schema.get("required", [])

        # Drop extra keys
        cleaned = {k: v for k, v in data.items() if k in allowed}

        # Backfill required
        for key in required:
            if key not in cleaned or cleaned[key] in (None, "", []):
                cleaned[key] = "Unknown"

        # Optionally coerce trivial types (lightweight safety)
        for key, spec in props.items():
            if key not in cleaned:
                continue
            val = cleaned[key]
            if spec.get("type") == "array":
                if not isinstance(val, list):
                    cleaned[key] = [val] if val not in (None, "Unknown") else []
            elif spec.get("type") == "string":
                if val is None:
                    cleaned[key] = "Unknown"
                elif not isinstance(val, str):
                    cleaned[key] = str(val)

        return cleaned

    def _maybe_validate_with_jsonschema(self, data: dict[str, Any], schema: dict[str, Any]) -> None:
        try:
            import jsonschema  # type: ignore

            jsonschema.validate(instance=data, schema=schema)
        except ModuleNotFoundError:
            # Optional dependency; skip if not installed
            pass
        except Exception as e:
            # If strict is on, we already tried to fix shape. Still raise to
            # make contract explicit: return should be schema-valid.
            raise ValueError(f"Schema validation failed: {e}") from e


EXTRACT_METADATA_PROMPT = """
You are an expert metadata extraction engine. Read the document and output a
single JSON object that conforms EXACTLY to the injected JSON Schema.

STRICT OUTPUT CONTRACT:
- Output MUST be a single valid JSON object. Do not include explanations,
  preambles, markdown code fences, or trailing text.
- Use ONLY the keys defined in the schema's properties. Do not add extra keys.
- Include ALL fields listed in the schema's required list.
- If a required field is missing or not inferable, set its value to "Unknown".
- Normalize common types:
  - Dates: use YYYY-MM-DD where a full date is available, otherwise "Unknown".
  - Arrays: ensure each item matches the item type in the schema.
  - Strings: strip markup and artifacts.
- Ensure the final object validates against the schema types and formats.

JSON Schema:
<schema>
{{json_schema}}
</schema>

Document Library Context (do not echo; use only for disambiguation):
<context>
{{document_library_context}}
</context>

Document:
<document>
{{document_text}}
</document>
""".strip()
