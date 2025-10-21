# TODO

- [ ] Document should be a class that is imported in converter, chunker, extractor, and vector db.
  - [ ] It should have functions: convert, chunk, extract, and store which export the necessary object to the other modules.
- [ ] Make sure configs for each module are properly initiated and used

Desired workflow:

- Server for Crawler
  - Endpoints:
    - PUT /upload {documents: [bytes or something], collection: collection_name, authorized_roles: [], config: {extra config for stuff that can be configurable even after collection is created}}
      - Auth header requried
    - POST/PATCH /collection {collection: collection_name, config: {}} creates the collection, indexes, etc
- Server for admin

  - ## Endpoints:

- Security:

  - How to enforce read access to search functions that are guaranteed to filter by permission?
    - Only user BD has read access. No one except that exact search function can log in as BD and admin type users. Kind of like an NPE cert.

- [x] refactor configs and all types to using pydantic
  - [x] Test to make sure it runs
- [x] Enable permissions, create users, test row level RBAC
- [ ] Config validation:
  - before running anything it should check:
    - Do the LLMs exist
    - Can the user connect to milvus and is authorized to write to the collection in quesiton
- Refactor

  - director should not have so many nested folders. Inside src/crawler should have:
    - config
    - vector_db
    - converter
    - llm (instantiates vision, llm, embedding)
    - extractor

- [ ] make a rest server

  - [ ] Registry of collections, be able to load and know what metadata to extract. Essentially save the configs used.
  - [ ] Must authenticate user, be able to correctly assign groups to the document
  - api
    - PUT /upload {documents: [bytes or something], collection: collection_name, authorized_roles: [], config: {extra config for stuff that can be configurable even after collection is created}}
      - Auth header requried
    - POST /collection {collection: collection_name, config: {}} creates the collection, indexes, etc

- [ ] make a cli
- [ ] make a local-minified version
  - [ ] single binary (ish) that runs, indexes the directories you tell it, then you can chat with it.
  - [ ] files and indexes stored .rrccp
- [ ] update the docs
- [ ] improve logging, image describer log doesn't mix with the tqdm
  - [ ] can the logs under the main file tqdm be overwitten or stay in place
  - [ ] remove the emojis from logs
  - [ ] scoped logging
- [ ] move config into package/directory
  - [ ] Add config validation where every aspect is tested
    - [ ] image describe can be sent an image and get a description, no 404
    - [ ] model is tested
    - [ ] embedding model is tested
    - [ ] milvus connection is tested
    - [ ] metadata schema is valid
    - [ ] etc
- [ ] Semantic Chunking
  - [ ] graph db and relationships too
    - **Not yet, must design memory model first**
- [ ] Do I migrate from milvus to qdrant or other?

- Design new architecture
  - [ ] Upload -> process -> insert into db
    - [ ] upload via some constructed UI with drag/drop or filepath
    - [ ] process via docling
      - [ ] docling: pdf/doc -> md, perform chunking too
      - [ ] metadata via extractor (llm)
    - [ ] insert into db
      - [ ] do i stick with milvus?

How to refactor the converter into a standalone package
Below is a clean, extensible API design for your converter layer that’s easy to use, test, and extend. It standardizes creation and invocation, introduces a strong interface, typed results, and a plugin-style registry.

Everyday usage (TL;DR)

- Create a converter:

  - From a discriminated Pydantic config union via factory
  - Or by name via the registry

- Use the converter:
  - Convert a single document (sync or async) with optional progress callbacks
  - Get a rich result object (markdown + assets + stats)
  - Optionally batch convert

1. Best API to create a converter

- Option A (preferred): Discriminated union config + factory

  - You define a Pydantic config per backend. The factory returns the correct implementation based on the type discriminator.

- Option B: Registry by name
  - Useful for plugin/provider injection or loading from dynamic config (e.g., YAML/JSON).

Example

```python
from processing.converters import (
  create_converter,
  PyMuPDFConfig,
  DocumentInput,
  ConvertOptions,
)

config = PyMuPDFConfig(
  type="pymupdf",
  image_describer={"type": "ollama", "model": "granite3.2-vision:latest"},
)

converter = create_converter(config)
result = converter.convert(
  DocumentInput.from_path("document.pdf"),
  options=ConvertOptions(describe_images=True),
)
print(result.markdown)
```

Or via registry

```python
from processing.converters import (
  registry,
  DocumentInput,
  ConvertOptions,
)

converter = registry.create("pymupdf", image_describer={"type": "dummy"})
result = converter.convert(
  DocumentInput.from_path("document.pdf"),
  options=ConvertOptions(extract_tables=True),
)
```

2. Best API to use a converter (convert a document)

- Single convert (sync/async)
- Optional progress callback
- Optional batch

Example with progress callback

```python
from processing.converters import (
  create_converter,
  PyMuPDFConfig,
  DocumentInput,
  ConvertOptions,
  ProgressEvent,
)

def on_progress(evt: ProgressEvent) -> None:
    print(f"[{evt.stage}] page={evt.page}/{evt.total_pages}: {evt.message}")

converter = create_converter(
  PyMuPDFConfig(type="pymupdf", image_describer={"type": "ollama"})
)

result = converter.convert(
  DocumentInput.from_path("file.pdf"),
  options=ConvertOptions(
    extract_tables=True,
    describe_images=True,
    page_range=(1, 10),
  ),
  on_progress=on_progress,
)

print(result.stats)
print(result.markdown[:1000])
```

Async + batch

```python
import asyncio
from processing.converters import create_converter, MarkItDownConfig, DocumentInput

async def main():
    conv = create_converter(
        MarkItDownConfig(type="markitdown", llm_base_url="http://localhost:11434", llm_model="llava")
    )
    docs = [
        DocumentInput.from_path("a.pdf"),
        DocumentInput.from_path("b.docx"),
    ]
    results = await conv.aconvert_many(docs, concurrency=2)
    print([r.stats for r in results])

asyncio.run(main())
```

Proposed file/folder structure

- Keep the converters as a subpackage with clear, focused modules.

processing/
converters/
**init**.py # Public API exports
base.py # Converter interface, exceptions
types.py # DocumentInput, ConvertOptions, results, events
configs.py # Discriminated union config models
factory.py # create_converter(config: ConverterConfig)
registry.py # Plugin registry for converters by name
markitdown.py # MarkItDownConverter
docling.py # DoclingConverter
docling_vlm.py # DoclingVLMConverter
pymupdf.py # PyMuPDFConverter and ImageDescriber impls

# other modules: llm/, extractor/, embeddings/

Types and functions you should have

- Core interface and lifecycle

  - Converter (abstract): convert, aconvert, convert_many, aconvert_many, close
  - Capabilities introspection
  - Support checks for input types/mime

- Input and options

  - DocumentInput: supports path, bytes, file-like; plus filename/mime hints
  - ConvertOptions: toggles for metadata/images/tables/page_range/etc.

- Results

  - ConvertedDocument: markdown + assets (images, tables) + metadata + stats
  - Asset models: ImageAsset, TableAsset
  - ConversionStats: timings, counts

- Events

  - ProgressEvent: stage, page, total_pages, message, metrics
  - ProgressCallback: Callable[[ProgressEvent], None]

- Configs (discriminated union)

  - MarkItDownConfig
  - DoclingConfig
  - DoclingVLMConfig
  - PyMuPDFConfig

- Factories/registry

  - create_converter(config: ConverterConfig) -> Converter
  - registry.register(name: str, builder: Callable[..., Converter])
  - registry.create(name: str, \*\*kwargs) -> Converter

- Exceptions

  - ConversionError
  - UnsupportedFormatError
  - DependencyMissingError
  - TimeoutError

- Utilities
  - mime detection helper (optional)
  - page range parsing (optional)

A well-rounded interface each implementation must implement

base.py

```python
from __future__ import annotations

import abc
from typing import Iterable, List, Optional, Protocol, Callable

from .types import (
    DocumentInput,
    ConvertOptions,
    ConvertedDocument,
    ProgressEvent,
    Capabilities,
)
from .configs import ConverterConfig


ProgressCallback = Callable[[ProgressEvent], None]


class Converter(abc.ABC):
    """
    Interface for document converters.

    Implementations must be stateless or externally re-entrant enough to allow
    multiple conversions in a row. Any network handles should be managed and
    closed in close().
    """

    def __init__(self, config: ConverterConfig):
        self.config = config

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-friendly name for this converter backend."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def capabilities(self) -> Capabilities:
        """Describe supported formats and features."""
        raise NotImplementedError

    @abc.abstractmethod
    def supports(self, doc: DocumentInput) -> bool:
        """Return True if this converter can handle the given document."""
        raise NotImplementedError

    @abc.abstractmethod
    def convert(
        self,
        doc: DocumentInput,
        options: Optional[ConvertOptions] = None,
        on_progress: Optional[ProgressCallback] = None,
    ) -> ConvertedDocument:
        """Convert a single document (blocking)."""
        raise NotImplementedError

    async def aconvert(
        self,
        doc: DocumentInput,
        options: Optional[ConvertOptions] = None,
        on_progress: Optional[ProgressCallback] = None,
    ) -> ConvertedDocument:
        """Async version of convert. Default impl runs in a thread."""
        # Implementations may override for true async I/O
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self.convert(doc, options, on_progress)
        )

    def convert_many(
        self,
        docs: Iterable[DocumentInput],
        options: Optional[ConvertOptions] = None,
        on_progress: Optional[ProgressCallback] = None,
    ) -> List[ConvertedDocument]:
        """Convert multiple documents sequentially. Override for batching."""
        return [self.convert(d, options, on_progress) for d in docs]

    async def aconvert_many(
        self,
        docs: Iterable[DocumentInput],
        options: Optional[ConvertOptions] = None,
        concurrency: int = 4,
        on_progress: Optional[ProgressCallback] = None,
    ) -> List[ConvertedDocument]:
        """Async conversion of multiple documents with bounded concurrency."""
        import asyncio
        sem = asyncio.Semaphore(concurrency)

        async def run(d: DocumentInput) -> ConvertedDocument:
            async with sem:
                return await self.aconvert(d, options, on_progress)

        return await asyncio.gather(*(run(d) for d in docs))

    def close(self) -> None:
        """Release any resources (network pools, temp dirs). Optional."""
        return
```

types.py

```python
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple, List, Union, IO
from pydantic import BaseModel, Field, model_validator
from pathlib import Path


BBox = Tuple[float, float, float, float]


class DocumentInput(BaseModel):
    source: str = Field(..., description="path | bytes | fileobj")
    path: Optional[Path] = None
    bytes_data: Optional[bytes] = None
    fileobj: Optional[IO[bytes]] = None
    filename: Optional[str] = None
    mime_type: Optional[str] = None

    @model_validator(mode="after")
    def _validate_source(self) -> "DocumentInput":
        if self.source == "path" and not self.path:
            raise ValueError("path required when source='path'")
        if self.source == "bytes" and self.bytes_data is None:
            raise ValueError("bytes_data required when source='bytes'")
        if self.source == "fileobj" and self.fileobj is None:
            raise ValueError("fileobj required when source='fileobj'")
        return self

    @classmethod
    def from_path(cls, p: Union[str, Path], mime_type: Optional[str] = None):
        p = Path(p)
        return cls(source="path", path=p, filename=p.name, mime_type=mime_type)

    @classmethod
    def from_bytes(
        cls, data: bytes, filename: Optional[str] = None, mime_type: Optional[str] = None
    ):
        return cls(
            source="bytes",
            bytes_data=data,
            filename=filename,
            mime_type=mime_type,
        )

    @classmethod
    def from_fileobj(
        cls, f: IO[bytes], filename: Optional[str] = None, mime_type: Optional[str] = None
    ):
        return cls(source="fileobj", fileobj=f, filename=filename, mime_type=mime_type)


class ConvertOptions(BaseModel):
    include_metadata: bool = True
    include_page_numbers: bool = True
    include_images: bool = True
    describe_images: bool = False
    image_prompt: Optional[str] = None
    extract_tables: bool = True
    table_strategy: str = "lines_strict"
    reading_order: bool = True
    page_range: Optional[Tuple[int, int]] = None  # inclusive 1-based range
    timeout_sec: Optional[float] = None


class ImageAsset(BaseModel):
    page_number: int = Field(..., ge=0)
    bbox: Optional[BBox] = None
    ext: str
    data: bytes
    description: Optional[str] = None


class TableAsset(BaseModel):
    page_number: int = Field(..., ge=0)
    bbox: Optional[BBox] = None
    rows: int = 0
    cols: int = 0
    markdown: str


class ConversionStats(BaseModel):
    total_pages: int = 0
    processed_pages: int = 0
    text_blocks: int = 0
    images: int = 0
    images_described: int = 0
    tables: int = 0
    total_time_sec: Optional[float] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class ConvertedDocument(BaseModel):
    source_name: Optional[str] = None
    markdown: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    images: List[ImageAsset] = Field(default_factory=list)
    tables: List[TableAsset] = Field(default_factory=list)
    stats: ConversionStats = Field(default_factory=ConversionStats)
    warnings: List[str] = Field(default_factory=list)


class ProgressEvent(BaseModel):
    stage: str
    page: Optional[int] = None
    total_pages: Optional[int] = None
    message: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)


class Capabilities(BaseModel):
    name: str
    supports_pdf: bool = True
    supports_docx: bool = True
    supports_images: bool = True
    supports_tables: bool = True
    requires_vision: bool = False
    supported_mime_types: List[str] = Field(default_factory=list)
```

configs.py

```python
from __future__ import annotations

from typing import Literal, Optional, Dict, Any, Union, Annotated
from pydantic import BaseModel, Field

# Optional: reuse your existing LLMConfig if desired
# from ..llm import LLMConfig


class BaseConverterConfig(BaseModel):
    type: str = Field(..., description="Discriminator")


class MarkItDownConfig(BaseConverterConfig):
    type: Literal["markitdown"]
    llm_base_url: str
    llm_model: str
    api_key: Optional[str] = None
    enable_plugins: bool = False


class DoclingConfig(BaseConverterConfig):
    type: Literal["docling"]
    vlm_base_url: str = "http://localhost:11434"
    vlm_model: str = "llava"
    prompt: Optional[str] = None
    timeout_sec: float = 600.0
    scale: float = 1.0


class DoclingVLMConfig(BaseConverterConfig):
    type: Literal["docling_vlm"]
    # relies on docling defaults; keep minimal for clarity


class PyMuPDFConfig(BaseConverterConfig):
    type: Literal["pymupdf"]
    image_describer: Optional[Dict[str, Any]] = None  # {type, model, base_url}
    default_table_strategy: str = "lines_strict"


ConverterConfig = Annotated[
    Union[MarkItDownConfig, DoclingConfig, DoclingVLMConfig, PyMuPDFConfig],
    Field(discriminator="type"),
]
```

factory.py

```python
from __future__ import annotations

from .configs import ConverterConfig
from .base import Converter
from .markitdown import MarkItDownConverter
from .docling import DoclingConverter
from .docling_vlm import DoclingVLMConverter
from .pymupdf import PyMuPDFConverter

_MAP = {
    "markitdown": MarkItDownConverter,
    "docling": DoclingConverter,
    "docling_vlm": DoclingVLMConverter,
    "pymupdf": PyMuPDFConverter,
}

def create_converter(config: ConverterConfig) -> Converter:
    key = config.type.lower()
    if key not in _MAP:
        raise ValueError(f"Unsupported converter type: {config.type}")
    return _MAP[key](config)
```

registry.py

```python
from __future__ import annotations

from typing import Callable, Dict, Any
from .base import Converter

class _Registry:
    def __init__(self) -> None:
        self._builders: Dict[str, Callable[..., Converter]] = {}

    def register(self, name: str, builder: Callable[..., Converter]) -> None:
        self._builders[name.lower()] = builder

    def create(self, name: str, **kwargs: Any) -> Converter:
        key = name.lower()
        if key not in self._builders:
            raise ValueError(f"Unknown converter: {name}")
        return self._builders[key](**kwargs)

registry = _Registry()
```

Example of an implementation adhering to the interface

markitdown.py

```python
from __future__ import annotations

from typing import Optional
from .base import Converter
from .types import (
    DocumentInput,
    ConvertOptions,
    ConvertedDocument,
    ProgressEvent,
    Capabilities,
)
from .configs import MarkItDownConfig


class MarkItDownConverter(Converter):
    def __init__(self, config: MarkItDownConfig):
        super().__init__(config)
        self._client = self._create_client(config)

    @property
    def name(self) -> str:
        return "MarkItDown"

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            name=self.name,
            supports_pdf=True,
            supports_docx=True,
            supports_images=True,
            supports_tables=False,
            requires_vision=True,
            supported_mime_types=[
                "application/pdf",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ],
        )

    def supports(self, doc: DocumentInput) -> bool:
        # You can implement detection via doc.mime_type or filename
        return True

    def convert(
        self,
        doc: DocumentInput,
        options: Optional[ConvertOptions] = None,
        on_progress: Optional[callable] = None,
    ) -> ConvertedDocument:
        if on_progress:
            on_progress(
                ProgressEvent(stage="start", message="MarkItDown conversion start")
            )

        # ... perform conversion, build markdown string ...
        markdown = "# Example\n\nConverted content here."

        if on_progress:
            on_progress(ProgressEvent(stage="finish", message="done"))

        return ConvertedDocument(
            source_name=doc.filename,
            markdown=markdown,
        )

    def _create_client(self, cfg: MarkItDownConfig):
        # Initialize any SDK clients here
        return None
```

Why this API?

- Strong typing via Pydantic discriminators keeps configuration safe and self-documenting.
- A single, minimal interface for implementations (convert/aconvert, supports, capabilities).
- A rich result object separates concerns (markdown vs. assets vs. stats).
- Optional progress events let you surface per-page progress without forcing a streaming API on all backends.
- Registry + factory makes it easy to add new providers without touching core code.
- Batch and async methods are standardized (implementors can override for efficiency).

Notes on migration from your current code

- Move provider-specific fields out of a generic ConverterConfig into specific config classes (MarkItDownConfig, DoclingConfig, PyMuPDFConfig).
- Retain your existing ExtractedImage semantics by mapping it to ImageAsset.
- Keep your PyMuPDF image describer as a pluggable option under PyMuPDFConfig.image_describer and/or ConvertOptions.describe_images/image_prompt.
- Maintain your logging but channel user-facing progress through ProgressEvent callbacks.

This keeps creation simple, usage ergonomic, and implementation details isolated, while giving you a consistent, type-safe API across backends.

Launch docling with:

```bash
docker run -p 5001:5001 -e DOCLING_SERVE_ENABLE_UI=1 -e DOCLING_SERVE_ENABLE_REMOTE_SERVICES=1 quay.io/docling-project/docling-serve
```

```bash
# curl -X POST 'http://localhost:5001/v1/convert/source' \
curl -X POST 'https://api.meatheadmathematician.com/test/v1/convert/source' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "options": {
      "from_formats": ["pdf"],
      "to_formats": ["md"],
      "abort_on_error": true,
      "do_picture_description": true,
      "image_export_mode": "embedded",
      "include_images": true,
      "picture_description_api": {
        "url": "http://host.docker.internal:11434/v1/chat/completions",
        "params": {
          "model": "granite3.2-vision:latest"
        },
        "timeout": 600,
        "prompt": "Describe this image in detail for a technical document."
      }
    },
    "sources": [{"kind": "http", "url": "https://arxiv.org/pdf/2501.17887"}]
  }' > test.json

curl -X POST 'https://api.meatheadmathematician.com/test/v1/convert/source' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "options": {
      "from_formats": ["pdf"],
      "to_formats": ["md"],
      "abort_on_error": true,
      "do_picture_description": false,
      "image_export_mode": "embedded",
      "include_images": true
    },
    "sources": [{"kind": "http", "url": "https://arxiv.org/pdf/2501.17887"}]
  }' > test.json
```
