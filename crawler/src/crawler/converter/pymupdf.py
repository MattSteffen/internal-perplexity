"""
PyMuPDF converter implementation with robust reading order, table extraction,
and concurrent image description.

This module provides a comprehensive converter implementation using PyMuPDF
for PDF processing with:
- multi-column to single-column reading order (column-aware)
- table detection (multi-strategy, dedup, markdown)
- image extraction with concurrent AI-powered descriptions and caching
"""

from __future__ import annotations

import base64
import hashlib
import math
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, IO, Literal

import fitz  # PyMuPDF
from pydantic import BaseModel, Field

from .base import Converter
from .types import (
    DocumentInput,
    ConvertOptions,
    ConvertedDocument,
    ImageAsset,
    TableAsset,
    ConversionStats,
    BBox,
)
from ..llm.llm import LLMConfig


# ----------------------------- Config and VLM ---------------------------------


class PyMuPDFConfig(BaseModel):
    """Configuration for PyMuPDF converter."""

    type: Literal["pymupdf"]
    vlm_config: Optional[LLMConfig] = Field(
        default=None, description="VLM configuration"
    )
    convert_options: Optional[ConvertOptions] = Field(
        default=None, description="Conversion options"
    )


class VLMInterface:
    """Abstract interface for image description services."""

    def describe_image(
        self, image_data: bytes, image_ext: str, prompt: Optional[str] = None
    ) -> str:
        raise NotImplementedError


class OllamaVLM(VLMInterface):
    """Implementation for Ollama API VLM."""

    def __init__(
        self,
        model_name: str = "llava",
        base_url: str = "http://localhost:11434",
        timeout_sec: int = 60,
    ):
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("OllamaVLM requires a non-empty model_name")
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("OllamaVLM requires a non-empty base_url")

        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec

        try:
            import requests  # type: ignore

            self.requests = requests
        except ImportError as e:
            raise ImportError(
                "requests library not found. Install with: pip install requests"
            ) from e

    def describe_image(
        self, image_data: bytes, image_ext: str, prompt: Optional[str] = None
    ) -> str:
        try:
            image_b64 = base64.b64encode(image_data).decode("utf-8")
            prompt = (
                prompt
                or "Describe this image in detail. Focus on the main content, "
                "objects, text, and any relevant information useful in a "
                "document context."
            )

            url = f"{self.base_url}/api/generate"
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False,
            }
            headers = {"Content-Type": "application/json"}

            resp = self.requests.post(
                url, json=payload, headers=headers, timeout=self.timeout_sec
            )
            if resp.status_code == 200:
                data = resp.json()
                desc = (data.get("response") or "").strip()
                return desc or f"[No description returned for {image_ext} image]"
            return f"[Ollama error: HTTP {resp.status_code}]"
        except Exception as e:
            return f"[Error describing image: {e}]"


class DummyVLM(VLMInterface):
    """Dummy implementation for testing and as a safe fallback."""

    def describe_image(
        self, image_data: bytes, image_ext: str, prompt: Optional[str] = None
    ) -> str:
        return f"[Dummy description for {image_ext} image of {len(image_data)} bytes]"


# ------------------------------- Utilities ------------------------------------


def _rect_to_bbox(r: fitz.Rect) -> BBox:
    return (float(r.x0), float(r.y0), float(r.x1), float(r.y1))


def _bbox_from_tuple(t: Tuple[float, float, float, float]) -> BBox:
    return (float(t[0]), float(t[1]), float(t[2]), float(t[3]))


def _bbox_overlap_ratio(b1: BBox, b2: BBox) -> float:
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = max(1e-9, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    return inter / a1


def _iou(b1: BBox, b2: BBox) -> float:
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(1e-9, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    a2 = max(1e-9, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    return inter / (a1 + a2 - inter)


def _dedup_tables(tables: List[TableAsset]) -> List[TableAsset]:
    out: List[TableAsset] = []
    for t in tables:
        if not any(_iou(t.bbox, u.bbox) > 0.9 for u in out):
            out.append(t)
    return out


def _convert_table_to_markdown(table_data: List[List[Any]]) -> str:
    """Convert table 2D list to markdown."""
    if not table_data:
        return ""

    def _cell_to_str(c: Any) -> str:
        s = "" if c is None else str(c)
        return s.replace("\n", " ").strip()

    header = table_data[0]
    md_lines = []
    header_row = "| " + " | ".join(_cell_to_str(c) for c in header) + " |"
    md_lines.append(header_row)
    sep = "| " + " | ".join("---" for _ in header) + " |"
    md_lines.append(sep)
    for row in table_data[1:]:
        md_lines.append("| " + " | ".join(_cell_to_str(c) for c in row) + " |")
    return "\n".join(md_lines)


def _hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _in_range(page_index_0: int, rng: Optional[Tuple[int, int]], total: int) -> bool:
    if not rng:
        return True
    s = max(1, min(rng[0], total))
    e = max(1, min(rng[1], total))
    return s <= (page_index_0 + 1) <= e


# ----------------------- Reading-order Text Extraction -------------------------


def _filter_out_table_words(words: List[Tuple], table_areas: List[BBox]) -> List[Tuple]:
    if not table_areas:
        return words
    out = []
    for w in words:
        wb = (w[0], w[1], w[2], w[3])
        if all(_bbox_overlap_ratio(wb, tb) <= 0.3 for tb in table_areas):
            out.append(w)
    return out


def _line_height_estimate(words: List[Tuple]) -> float:
    if not words:
        return 8.0
    heights = [abs(w[3] - w[1]) for w in words]
    heights.sort()
    return heights[min(len(heights) - 1, len(heights) // 2)]


def _find_column_splits(
    words: List[Tuple], page_width: float, bins: int = 64
) -> List[float]:
    """Find potential vertical splits; returns a list of x positions."""
    if not words:
        return []

    hist = [0] * bins
    for w in words:
        x0, _, x1, _ = w[:4]
        i0 = max(0, min(bins - 1, int(bins * (x0 / page_width))))
        i1 = max(0, min(bins - 1, int(bins * (x1 / page_width))))
        for i in range(i0, i1 + 1):
            hist[i] += 1

    max_val = max(hist) if hist else 0
    if max_val == 0:
        return []

    threshold = max(1, int(0.12 * max_val))
    valleys: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for i, h in enumerate(hist):
        if h <= threshold and start is None:
            start = i
        if (h > threshold or i == bins - 1) and start is not None:
            end = i if h > threshold else i
            valleys.append((start, end))
            start = None

    if not valleys:
        return []

    # Prefer widest near center
    valleys.sort(
        key=lambda ve: (
            -((ve[1] - ve[0]) + 1),
            abs(((ve[0] + ve[1]) / 2) - (bins / 2.0)),
        )
    )
    v0 = valleys[0]
    if (v0[1] - v0[0] + 1) < max(2, int(0.06 * bins)):
        return []

    split_x = ((v0[0] + v0[1] + 1) / 2.0) * (page_width / bins)
    return [split_x]  # good for common two-column pages


def _split_words_into_columns(
    words: List[Tuple], page_width: float
) -> List[List[Tuple]]:
    splits = _find_column_splits(words, page_width)
    if not splits:
        return [words]

    splits = sorted(splits)
    cols: List[List[Tuple]] = []
    last_x = -math.inf
    for sx in splits + [math.inf]:
        col = [w for w in words if last_x <= 0.5 * (w[0] + w[2]) < sx]
        if col:
            cols.append(col)
        last_x = sx
    return cols


def _words_to_lines(words: List[Tuple], y_tol: float) -> List[List[Tuple]]:
    words = sorted(words, key=lambda w: (w[1], w[0]))
    lines: List[List[Tuple]] = []
    cur: List[Tuple] = []
    cur_y: Optional[float] = None

    for w in words:
        y_mid = 0.5 * (w[1] + w[3])
        if cur and cur_y is not None and abs(y_mid - cur_y) > y_tol:
            cur.sort(key=lambda ww: ww[0])
            lines.append(cur)
            cur = [w]
            cur_y = y_mid
        else:
            if not cur:
                cur = [w]
                cur_y = y_mid
            else:
                cur.append(w)

    if cur:
        cur.sort(key=lambda ww: ww[0])
        lines.append(cur)
    return lines


def _join_line_text(line: List[Tuple]) -> str:
    parts: List[str] = []
    for i, w in enumerate(line):
        txt = str(w[4])
        if i == 0:
            parts.append(txt)
        else:
            prev = line[i - 1]
            gap = w[0] - prev[2]
            parts.append((" " if gap > 1.5 else "") + txt)
    return "".join(parts).strip()


def _lines_to_paragraphs(
    lines: List[List[Tuple]], line_gap_tol: float
) -> List[List[List[Tuple]]]:
    paras: List[List[List[Tuple]]] = []
    cur: List[List[Tuple]] = []
    last_y: Optional[float] = None
    for ln in lines:
        y_mid = 0.5 * (ln[0][1] + ln[0][3])
        if cur and last_y is not None and (y_mid - last_y) > line_gap_tol:
            paras.append(cur)
            cur = [ln]
        else:
            cur.append(ln)
        last_y = y_mid
    if cur:
        paras.append(cur)
    return paras


def _paragraph_text_from_lines(lines: List[List[Tuple]]) -> str:
    """Join line texts with hyphenation handling across lines."""
    line_texts = [_join_line_text(ln) for ln in lines if ln]
    if not line_texts:
        return ""
    buf = line_texts[0]
    for nxt in line_texts[1:]:
        if buf.endswith("-") and nxt and nxt[0].islower():
            buf = buf[:-1] + nxt
        else:
            buf += " " + nxt
    return buf.strip()


def extract_text_blocks_reading_order(
    page: fitz.Page, table_areas: List[BBox]
) -> List[Dict[str, Any]]:
    """Return text blocks (paragraphs) in reading order with bbox."""
    words = page.get_text("words")  # (x0,y0,x1,y1,"text", block, line, word)
    if not words:
        return []

    words = _filter_out_table_words(words, table_areas)
    if not words:
        return []

    page_width = float(page.rect.width)
    line_h = _line_height_estimate(words)
    y_tol = max(2.0, 0.6 * line_h)
    gap_tol = max(1.5 * line_h, 8.0)

    cols = _split_words_into_columns(words, page_width)
    blocks: List[Dict[str, Any]] = []
    block_no = 0
    for col in sorted(cols, key=lambda c: min(w[0] for w in c)):
        lines = _words_to_lines(col, y_tol=y_tol)
        paras = _lines_to_paragraphs(lines, line_gap_tol=gap_tol)
        for para in paras:
            text = _paragraph_text_from_lines(para)
            if not text.strip():
                continue
            xs = [min(w[0] for w in ln) for ln in para if ln]
            ys = [min(w[1] for w in ln) for ln in para if ln]
            xe = [max(w[2] for w in ln) for ln in para if ln]
            ye = [max(w[3] for w in ln) for ln in para if ln]
            bbox = (min(xs), min(ys), max(xe), max(ye))
            blocks.append(
                {"type": "text", "bbox": bbox, "text": text, "block_no": block_no}
            )
            block_no += 1
    return blocks


# -------------------------- Table and Image Extraction -------------------------


def extract_tables(page: fitz.Page, options: ConvertOptions) -> List[TableAsset]:
    if not options.extract_tables:
        return []

    found: List[TableAsset] = []

    def run(strategy: str) -> List[TableAsset]:
        res: List[TableAsset] = []
        try:
            tables = page.find_tables(strategy=strategy)
            if not tables:
                return []
            for tbl in tables:
                try:
                    data = tbl.extract()
                    md = _convert_table_to_markdown(data)
                    res.append(
                        TableAsset(
                            page_number=page.number,
                            bbox=_rect_to_bbox(tbl.bbox),
                            rows=len(data),
                            cols=len(data[0]) if data else 0,
                            markdown=md,
                        )
                    )
                except Exception:
                    # skip malformed table extraction
                    continue
            return res
        except Exception:
            return []

    # Try multiple strategies
    # Prefer "lines", then "text" if nothing found
    found.extend(run("lines"))
    if not found:
        found.extend(run("text"))

    # Some PyMuPDF versions may have "cells" too
    if not found:
        found.extend(run("cells"))

    return _dedup_tables(found)


def extract_images_from_page(
    pdf_doc: fitz.Document, page: fitz.Page, include_images: bool
) -> List[ImageAsset]:
    if not include_images:
        return []

    results: List[ImageAsset] = []
    image_list = page.get_images(full=True)
    for img in image_list:
        xref = img[0]
        try:
            base_image = pdf_doc.extract_image(xref)
            image_data = base_image.get("image", b"")
            image_ext = base_image.get("ext", "png") or "png"
            rects = page.get_image_rects(xref)
            if not rects:
                bbox: Optional[BBox] = None
                # Still include the image without bbox
                results.append(
                    ImageAsset(
                        page_number=page.number,
                        bbox=bbox,
                        data=image_data,
                        ext=image_ext,
                    )
                )
            else:
                # Create an asset per occurrence on the page
                for r in rects:
                    bbox = _rect_to_bbox(r)
                    results.append(
                        ImageAsset(
                            page_number=page.number,
                            bbox=bbox,
                            data=image_data,
                            ext=image_ext,
                        )
                    )
        except Exception:
            # skip problematic images
            continue
    return results


def _detect_image_caption_below(
    page: fitz.Page, img_bbox: Optional[BBox], search_factor: float = 2.0
) -> Optional[str]:
    """Heuristic caption detection from a region below the image."""
    if not img_bbox:
        return None

    words = page.get_text("words")
    if not words:
        return None

    # Estimate line height to set search band
    line_h = _line_height_estimate(words)
    y0 = img_bbox[3]
    y1 = img_bbox[3] + max(line_h * search_factor, 12.0)
    region_words = [
        w
        for w in words
        if (y0 - 1.0) <= w[1] <= y1 and img_bbox[0] <= w[0] <= img_bbox[2]
    ]
    if not region_words:
        return None

    # Build lines
    lines = _words_to_lines(region_words, y_tol=max(2.0, 0.6 * line_h))
    line_texts = [_join_line_text(ln) for ln in lines if ln]
    if not line_texts:
        return None

    # Prefer lines starting with "Fig", "Figure", "Image"
    for t in line_texts:
        ts = t.strip()
        low = ts.lower()
        if low.startswith("fig ") or low.startswith("fig.") or low.startswith("figure"):
            return ts
        if low.startswith("image ") or low.startswith("img "):
            return ts

    # Otherwise, return shortest line (often captions are short)
    return min(line_texts, key=len).strip()


def describe_images_concurrently(
    vlm: VLMInterface,
    images: List[ImageAsset],
    prompt: str,
    max_workers: int = 4,
) -> None:
    """Describe images concurrently with caching by content hash."""
    cache: Dict[str, str] = {}
    idxs = list(range(len(images)))

    def task(i: int) -> Tuple[int, str]:
        img = images[i]
        h = _hash_bytes(img.data)
        if h in cache:
            return i, cache[h]
        desc = vlm.describe_image(img.data, img.ext, prompt)
        cache[h] = desc
        return i, desc

    # If DummyVLM, we can still parallelize; it's cheap enough
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(task, i) for i in idxs]
        for f in as_completed(futs):
            i, desc = f.result()
            images[i].description = desc


# ------------------------------- Converter ------------------------------------


class PyMuPDFConverter(Converter):
    """Document converter using PyMuPDF with comprehensive content extraction."""

    DEFAULT_IMAGE_PROMPT = (
        "Describe this image in detail. Focus on the main content, objects, "
        "text, and any relevant information useful in a document context."
    )

    def __init__(self, config: PyMuPDFConfig):
        super().__init__(config)
        self.config = config
        self.vlm = self._create_vlm()

    @property
    def name(self) -> str:
        return "PyMuPDF"

    # --------------------------- Public API -----------------------------------

    def convert(self, doc: DocumentInput) -> ConvertedDocument:
        options = self.config.convert_options or ConvertOptions()
        start_time = time.time()

        # Prepare temp file if needed
        tmp_path: Optional[Path] = None
        try:
            if doc.source == "path":
                filepath = str(doc.path)
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    if doc.source == "bytes":
                        tmp.write(doc.bytes_data or b"")
                    else:
                        assert doc.fileobj is not None
                        tmp.write(doc.fileobj.read())
                    filepath = tmp.name
                    tmp_path = Path(tmp.name)

            with fitz.open(filepath) as pdf_doc:
                total_pages = len(pdf_doc)

                stats = ConversionStats(
                    total_pages=total_pages,
                    processed_pages=0,
                    text_blocks=0,
                    images=0,
                    images_described=0,
                    tables=0,
                    total_time_sec=None,
                )
                warnings: List[str] = []

                # Accumulators
                all_images: List[ImageAsset] = []
                all_tables: List[TableAsset] = []

                # Per-page contents to emit after VLM step
                per_page_items: Dict[int, List[Dict[str, Any]]] = {}

                # Prepend document header
                doc_title = doc.filename or "unknown"
                metadata = pdf_doc.metadata or {}

                # Iterate pages with range and timeout checks
                for pidx in range(total_pages):
                    if (
                        options.timeout_sec
                        and (time.time() - start_time) > options.timeout_sec
                    ):
                        warnings.append("Conversion timeout exceeded. Stopping early.")
                        break

                    if not _in_range(pidx, options.page_range, total_pages):
                        continue

                    page = pdf_doc[pidx]
                    page_items: List[Dict[str, Any]] = []

                    # 1) Tables first
                    try:
                        tables = extract_tables(page, options)
                    except Exception as e:
                        tables = []
                        warnings.append(f"Page {pidx + 1}: table extraction error: {e}")
                    all_tables.extend(tables)
                    stats.tables += len(tables)
                    table_areas = [t.bbox for t in tables]

                    for t in tables:
                        page_items.append(
                            {
                                "type": "table",
                                "bbox": t.bbox,
                                "table": t,
                            }
                        )

                    # 2) Text in reading order, excluding table areas
                    try:
                        if options.reading_order:
                            text_blocks = extract_text_blocks_reading_order(
                                page, table_areas
                            )
                        else:
                            # Simpler fallback: one block of page text
                            txt = page.get_text("text")
                            text_blocks = (
                                [
                                    {
                                        "type": "text",
                                        "bbox": _bbox_from_tuple(tuple(page.rect)),
                                        "text": txt.strip(),
                                        "block_no": 0,
                                    }
                                ]
                                if txt
                                else []
                            )
                    except Exception as e:
                        text_blocks = []
                        warnings.append(f"Page {pidx + 1}: text extraction error: {e}")

                    stats.text_blocks += len(text_blocks)
                    for tb in text_blocks:
                        page_items.append(
                            {
                                "type": "text",
                                "bbox": tb["bbox"],
                                "text": tb["text"],
                                "block_no": tb.get("block_no", 0),
                            }
                        )

                    # 3) Images
                    try:
                        images = extract_images_from_page(
                            pdf_doc, page, options.include_images
                        )
                    except Exception as e:
                        images = []
                        warnings.append(f"Page {pidx + 1}: image extraction error: {e}")
                    stats.images += len(images)

                    # Caption heuristics (stored temporarily; appended to
                    # description later)
                    image_captions: List[Optional[str]] = []
                    for img in images:
                        cap = _detect_image_caption_below(page, img.bbox)
                        image_captions.append(cap)

                    # Attach captions to a temp field on items for later merge
                    for i, img in enumerate(images):
                        page_items.append(
                            {
                                "type": "image",
                                "bbox": img.bbox,
                                "image": img,
                                "page_local_index": i,
                                "caption": image_captions[i],
                            }
                        )

                    all_images.extend(images)
                    stats.processed_pages += 1
                    per_page_items[pidx] = page_items

                # 4) Describe images (after collecting all)
                if options.describe_images and all_images:
                    try:
                        max_workers = 4
                        # If provided in VLM config, use it
                        if isinstance(self.vlm, OllamaVLM):
                            max_workers = 4  # default; adjust as needed
                        describe_images_concurrently(
                            self.vlm,
                            all_images,
                            options.image_prompt or self.DEFAULT_IMAGE_PROMPT,
                            max_workers=max_workers,
                        )
                    except Exception as e:
                        warnings.append(f"Image description failed: {e}")
                # augment images with captions
                for pidx, items in per_page_items.items():
                    for it in items:
                        if it["type"] == "image":
                            img = it["image"]
                            cap = it.get("caption")
                            if cap:
                                if img.description:
                                    if "caption:" not in img.description.lower():
                                        img.description = (
                                            img.description.rstrip()
                                            + f"\nCaption: {cap}"
                                        )
                                else:
                                    img.description = f"Caption: {cap}"

                # Count how many got described
                stats.images_described = sum(
                    1 for im in all_images if (im.description or "").strip()
                )

                # 5) Build markdown (after descriptions available)
                md_parts: List[str] = []
                md_parts.append(f"# Document: {doc_title}\n\n")
                if options.include_metadata and metadata:
                    md_parts.append("## Document Metadata\n\n")
                    for k, v in metadata.items():
                        if v:
                            md_parts.append(f"- {k}: {v}\n")
                    md_parts.append("\n")

                for pidx in range(total_pages):
                    if pidx not in per_page_items:
                        continue
                    if options.include_page_numbers:
                        md_parts.append(f"## Page {pidx + 1}\n\n")

                    # Merge content items by position. Keep text order stable
                    # by using (y, x, tie-breakers).
                    items = per_page_items[pidx]

                    def sort_key(it: Dict[str, Any]) -> Tuple[float, float, int, int]:
                        bbox = it.get("bbox")
                        y = bbox[1] if bbox else 0.0
                        x = bbox[0] if bbox else 0.0
                        tie1 = 0
                        tie2 = 0
                        if it["type"] == "text":
                            tie1 = -1  # text before others at same y
                            tie2 = it.get("block_no", 0)
                        elif it["type"] == "table":
                            tie1 = 0
                        else:  # image
                            tie1 = 1
                            tie2 = it.get("page_local_index", 0)
                        return (y, x, tie1, tie2)

                    items_sorted = sorted(items, key=sort_key)

                    # Local counters for labels per page
                    table_counter = 0
                    image_counter = 0

                    for it in items_sorted:
                        if it["type"] == "text":
                            md_parts.append(it["text"] + "\n\n")
                        elif it["type"] == "table":
                            table_counter += 1
                            t: TableAsset = it["table"]
                            md_parts.append(
                                f"**Table {table_counter}** "
                                f"({t.rows} rows × {t.cols} cols)\n\n"
                            )
                            md_parts.append(t.markdown + "\n\n")
                        else:
                            image_counter += 1
                            img: ImageAsset = it["image"]
                            # Use a resolvable placeholder; consumer maps this
                            # to ConvertedDocument.images
                            img_hash = _hash_bytes(img.data)[:8]
                            asset_id = (
                                f"asset:image:p{pidx + 1}_{image_counter}_{img_hash}"
                            )
                            desc = img.description or "[No description available]"
                            md_parts.append(
                                f"![Image p{pidx + 1} #{image_counter}]"
                                f"({asset_id})\n"
                            )
                            md_parts.append(f"*{desc}*\n\n")

                total_time = time.time() - start_time
                stats.total_time_sec = total_time

                return ConvertedDocument(
                    source_name=doc.filename,
                    markdown="".join(md_parts),
                    metadata=metadata,
                    images=all_images,
                    tables=all_tables,
                    stats=stats,
                    warnings=warnings,
                )
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass

    # -------------------------- Internal helpers ------------------------------

    def _create_vlm(self) -> VLMInterface:
        """Create and configure the VLM based on configuration."""
        cfg = self.config.vlm_config
        # If no VLM config or provider invalid, use DummyVLM
        if cfg is None:
            return DummyVLM()

        provider = getattr(cfg, "provider", None)
        if provider == "ollama":
            model_name = getattr(cfg, "model_name", "llava")
            base_url = getattr(cfg, "base_url", "http://localhost:11434")
            timeout_sec = getattr(cfg, "timeout_sec", 60)
            try:
                return OllamaVLM(
                    model_name=model_name,
                    base_url=base_url,
                    timeout_sec=timeout_sec,
                )
            except Exception:
                # Fallback silently to Dummy if Ollama misconfigured
                return DummyVLM()

        if provider == "dummy":
            return DummyVLM()

        # Unknown provider -> safe fallback
        return DummyVLM()
