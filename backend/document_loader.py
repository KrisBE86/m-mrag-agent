"""
Document loading with image extraction and three-level hierarchical chunking.

- Supports PDF (PyMuPDF for image extraction + PyPDFLoader for text) and Word
  (python-docx for image extraction + Docx2txtLoader for text).
- Three-level sliding-window chunking (aligned with SuperMew):
    L1 ≈ 1200 chars (coarse overview)
    L2 ≈ 600 chars  (medium paragraph)
    L3 ≈ 300 chars  (fine leaf, the retrieval unit)
- Image extraction: extracts embedded images from each page/section, detects
  captions (e.g. "图1. xxx"), or uses LLM to generate descriptive names.
- Each extracted image is linked to its nearest L3 text chunk via chunk_id.
- Chunk ID format: {filename}::p{page}::l{level}::{index}
"""

import base64
import io
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Caption detection patterns ───────────────────────────────────

CAPTION_PATTERNS = [
    re.compile(r"图\s*(\d+)[\.\、\s]+(.+?)(?:\n|图\s*\d+|$)", re.DOTALL),
    re.compile(r"图版\s*(\d+)[\.\、\s]+(.+?)(?:\n|图版\s*\d+|$)", re.DOTALL),
    re.compile(r"Figure\s+(\d+)[\.\:\s]+(.+?)(?:\n|Figure\s+\d+|$)", re.DOTALL | re.IGNORECASE),
    re.compile(r"插图\s*(\d+)[\.\、\s]+(.+?)(?:\n|插图\s*\d+|$)", re.DOTALL),
]


def _extract_captions(text: str) -> dict[int, str]:
    """Find all image captions in text. Returns {figure_number: caption_text}."""
    captions: dict[int, str] = {}
    for pattern in CAPTION_PATTERNS:
        for match in pattern.finditer(text):
            num = int(match.group(1))
            caption = match.group(2).strip()
            if num not in captions or len(caption) > len(captions[num]):
                captions[num] = caption
    return captions


# ═══════════════════════════════════════════════════════════════════
# Image extraction
# ═══════════════════════════════════════════════════════════════════

def _extract_images_from_pdf(file_path: str) -> list[dict]:
    """
    Extract embedded images from a PDF file using PyMuPDF (fitz).
    Returns list of {page_number, image_index, image_bytes, format}.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("  ⚠ PyMuPDF not installed. Install: pip install pymupdf")
        return []

    images = []
    doc = fitz.open(file_path)
    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)
        for img_idx, img_info in enumerate(image_list):
            try:
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                ext = base_image.get("ext", "png")
                images.append({
                    "page_number": page_num + 1,  # 1-indexed
                    "image_index": img_idx,
                    "image_bytes": image_bytes,
                    "format": ext,
                })
            except Exception:
                continue
    doc.close()
    return images


def _extract_images_from_docx(file_path: str) -> list[dict]:
    """
    Extract embedded images from a Word document using python-docx.
    Returns list of {paragraph_index, image_index, image_bytes, format}.
    """
    try:
        from docx import Document
        from docx.opc.constants import RELATIONSHIP_TYPE as RT
    except ImportError:
        print("  ⚠ python-docx not installed. Install: pip install python-docx")
        return []

    images = []
    doc = Document(file_path)

    # Extract images from the document's relationships.
    img_counter = 0
    for rel in doc.part.rels.values():
        if "image" not in rel.reltype:
            continue
        try:
            image_bytes = rel.target_part.blob
            # Get extension from target_ref (e.g. "media/image1.png")
            # ImagePart.ext was removed in newer python-docx versions
            import os as _os
            _, ext = _os.path.splitext(rel.target_ref)
            ext = (ext or ".png").lstrip(".")
            images.append({
                "page_number": 0,  # Word doesn't have pages; will associate by paragraph
                "image_index": img_counter,
                "image_bytes": image_bytes,
                "format": ext,
            })
            img_counter += 1
        except Exception:
            continue

    return images


def _save_image(image_bytes: bytes, output_dir: str | Path, filename: str, img_index: int, fmt: str) -> str:
    """Save an extracted image to disk and return the relative path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_filename = re.sub(r"[^\w\-\.]", "_", filename)
    img_filename = f"{safe_filename}_img{img_index:03d}.{fmt}"
    img_path = output_dir / img_filename
    img_path.write_bytes(image_bytes)
    return str(img_path)


# ═══════════════════════════════════════════════════════════════════
# Image naming (caption detection + LLM fallback)
# ═══════════════════════════════════════════════════════════════════

def _name_image_from_captions(
    image_index: int,
    page_number: int,
    page_text: str,
    captions: dict[int, str],
) -> tuple[str, str]:
    """
    Try to name an image from detected captions.
    Returns (name, caption_text) — caption_text may be empty.
    """
    # Simple heuristic: image_index on a page maps to figure number.
    figure_num = image_index + 1
    if figure_num in captions:
        caption = captions[figure_num]
        return caption, caption

    # Fallback: use page-level context.
    if page_text.strip():
        first_line = page_text.strip().split("\n")[0][:100]
        return first_line, ""
    return f"第{page_number}页图片{image_index + 1}", ""


def _name_image_with_llm(
    image_bytes: bytes,
    page_text: str,
    image_index: int,
    page_number: int,
) -> str:
    """
    Use LLM (DeepSeek) to generate a descriptive name for an image.
    Sends the image as base64 data URI with surrounding page text for context.
    """
    import os

    from dotenv import load_dotenv
    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI

    load_dotenv()

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    fmt = "png"
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes))
        fmt = img.format.lower() if img.format else "png"
    except Exception:
        pass

    data_uri = f"data:image/{fmt};base64,{b64}"
    context = page_text[:500] if page_text else "无"

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "deepseek-v4-flash"),
        temperature=0.1,
        base_url=os.getenv("BASE_URL", "https://api.deepseek.com/v1"),
        extra_body={"thinking": {"type": "disabled"}},
    )

    prompt = (
        f"以下是一本书籍第{page_number}页中的一幅图片（第{image_index + 1}张）。\n"
        f"该页文字内容摘要：{context}\n\n"
        "请根据图片内容和上下文，为这张图片生成一个简短的标题（15字以内），"
        "用于图像检索。只输出标题文本，不要其他内容。"
    )

    try:
        msg = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ]
        )
        result = llm.invoke([msg])
        return (result.content or "").strip()
    except Exception:
        return f"第{page_number}页图片{image_index + 1}"


def _describe_image_with_vlm(image_bytes: bytes) -> str:
    """Thin wrapper around vlm_client.describe_image_with_vlm for use in ingestion."""
    from backend.vlm_client import describe_image_with_vlm
    return describe_image_with_vlm(image_bytes)



# ═══════════════════════════════════════════════════════════════════
# Three-level chunking (aligned with SuperMew)
# ═══════════════════════════════════════════════════════════════════

class DocumentLoader:
    """
    Document loader: reads PDF/Word, extracts images, and performs
    three-level hierarchical chunking (aligned with SuperMew).
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        level_1_size = max(1200, chunk_size * 2)
        level_1_overlap = max(240, chunk_overlap * 2)
        level_2_size = max(600, chunk_size)
        level_2_overlap = max(120, chunk_overlap)
        level_3_size = max(300, chunk_size // 2)
        level_3_overlap = max(60, chunk_overlap // 2)

        self._splitter_level_1 = RecursiveCharacterTextSplitter(
            chunk_size=level_1_size,
            chunk_overlap=level_1_overlap,
            add_start_index=True,
            separators=["\n\n", "\n", "。", "！", "？", "，", "、", " ", ""],
        )
        self._splitter_level_2 = RecursiveCharacterTextSplitter(
            chunk_size=level_2_size,
            chunk_overlap=level_2_overlap,
            add_start_index=True,
            separators=["\n\n", "\n", "。", "！", "？", "，", "、", " ", ""],
        )
        self._splitter_level_3 = RecursiveCharacterTextSplitter(
            chunk_size=level_3_size,
            chunk_overlap=level_3_overlap,
            add_start_index=True,
            separators=["\n\n", "\n", "。", "！", "？", "，", "、", " ", ""],
        )

    @staticmethod
    def _build_chunk_id(filename: str, page_number: int, level: int, index: int) -> str:
        return f"{filename}::p{page_number}::l{level}::{index}"

    def _split_page_to_three_levels(
        self,
        text: str,
        base_doc: dict,
        page_global_chunk_idx: int,
    ) -> list[dict]:
        """Split a page's text into L1 → L2 → L3 hierarchy. Aligned with SuperMew."""
        if not text:
            return []

        root_chunks: list[dict] = []
        page_number = int(base_doc.get("page_number", 0))
        filename = base_doc["filename"]

        level_1_docs = self._splitter_level_1.create_documents([text], [base_doc])
        level_1_counter = 0
        level_2_counter = 0
        level_3_counter = 0

        for level_1_doc in level_1_docs:
            level_1_text = (level_1_doc.page_content or "").strip()
            if not level_1_text:
                continue
            level_1_id = self._build_chunk_id(filename, page_number, 1, level_1_counter)
            level_1_counter += 1

            level_1_chunk = {
                **base_doc,
                "text": level_1_text,
                "chunk_id": level_1_id,
                "parent_chunk_id": "",
                "root_chunk_id": level_1_id,
                "chunk_level": 1,
                "chunk_idx": page_global_chunk_idx,
            }
            page_global_chunk_idx += 1
            root_chunks.append(level_1_chunk)

            level_2_docs = self._splitter_level_2.create_documents([level_1_text], [base_doc])
            for level_2_doc in level_2_docs:
                level_2_text = (level_2_doc.page_content or "").strip()
                if not level_2_text:
                    continue
                level_2_id = self._build_chunk_id(filename, page_number, 2, level_2_counter)
                level_2_counter += 1

                level_2_chunk = {
                    **base_doc,
                    "text": level_2_text,
                    "chunk_id": level_2_id,
                    "parent_chunk_id": level_1_id,
                    "root_chunk_id": level_1_id,
                    "chunk_level": 2,
                    "chunk_idx": page_global_chunk_idx,
                }
                page_global_chunk_idx += 1
                root_chunks.append(level_2_chunk)

                level_3_docs = self._splitter_level_3.create_documents([level_2_text], [base_doc])
                for level_3_doc in level_3_docs:
                    level_3_text = (level_3_doc.page_content or "").strip()
                    if not level_3_text:
                        continue
                    level_3_id = self._build_chunk_id(filename, page_number, 3, level_3_counter)
                    level_3_counter += 1
                    root_chunks.append({
                        **base_doc,
                        "text": level_3_text,
                        "chunk_id": level_3_id,
                        "parent_chunk_id": level_2_id,
                        "root_chunk_id": level_1_id,
                        "chunk_level": 3,
                        "chunk_idx": page_global_chunk_idx,
                    })
                    page_global_chunk_idx += 1

        return root_chunks

    def load_document(
        self,
        file_path: str,
        filename: str,
        image_output_dir: str | Path = "data/reference_images",
        use_llm_naming: bool = False,
        use_vlm_description: bool = False,
    ) -> tuple[list[dict], list[dict]]:
        """
        Load a document, extract images, and chunk text.

        Args:
            file_path: Path to the document file.
            filename: Display filename for provenance tracking.
            image_output_dir: Directory to save extracted images.
            use_llm_naming: If True and no caption found, use LLM to name images.
            use_vlm_description: If True, use Doubao Vision to generate pure
                visual descriptions for each extracted image. Descriptions
                are stored in distinguishing_features and added as L3 text
                chunks for retrieval.

        Returns:
            (text_chunks, image_entries)
            - text_chunks: L1/L2/L3 chunks (L3 for Milvus text, L1/L2 for PostgreSQL).
            - image_entries: Extracted images with metadata for Image Milvus.
        """
        file_lower = filename.lower()

        if file_lower.endswith(".pdf"):
            doc_type = "PDF"
            loader = PyPDFLoader(file_path)
            raw_images = _extract_images_from_pdf(file_path)
        elif file_lower.endswith((".docx", ".doc")):
            doc_type = "Word"
            loader = Docx2txtLoader(file_path)
            raw_images = _extract_images_from_docx(file_path)
        else:
            raise ValueError(f"不支持的文件类型: {filename}")

        # ── Load text ────────────────────────────────────────────
        try:
            raw_docs = loader.load()
        except Exception as e:
            raise Exception(f"处理文档失败: {str(e)}")

        # ── Extract captions from full text ──────────────────────
        full_text = "\n".join(doc.page_content for doc in raw_docs)

        # ── Build per-page data ──────────────────────────────────
        text_chunks: list[dict] = []
        image_entries: list[dict] = []

        # Group images by page.
        images_by_page: dict[int, list[dict]] = {}
        for img in raw_images:
            page = img.get("page_number", 0)
            images_by_page.setdefault(page, []).append(img)

        page_global_chunk_idx = 0
        for doc in raw_docs:
            page_num = doc.metadata.get("page", 0)
            page_text = (doc.page_content or "").strip()

            base_doc = {
                "filename": filename,
                "file_path": file_path,
                "file_type": doc_type,
                "page_number": page_num,
            }

            # Three-level chunking for this page.
            page_chunks = self._split_page_to_three_levels(
                text=page_text,
                base_doc=base_doc,
                page_global_chunk_idx=page_global_chunk_idx,
            )
            page_global_chunk_idx += len(page_chunks)
            text_chunks.extend(page_chunks)

            # ── Process images for this page ─────────────────────
            page_images = images_by_page.get(page_num, [])
            if not page_images:
                # Also match page 0 (Word docs use 0).
                page_images = images_by_page.get(0, [])

            captions = _extract_captions(page_text)

            # Find parent IDs for VLM chunks (if any images on this page)
            l1_chunks_on_page = [c for c in page_chunks if c.get("chunk_level") == 1]
            l2_chunks_on_page = [c for c in page_chunks if c.get("chunk_level") == 2]
            page_root_id = l1_chunks_on_page[0]["chunk_id"] if l1_chunks_on_page else ""
            page_parent_id = l2_chunks_on_page[0]["chunk_id"] if l2_chunks_on_page else ""

            for img_data in page_images:
                img_idx = img_data["image_index"]
                img_bytes = img_data["image_bytes"]
                fmt = img_data.get("format", "png")

                # Save image to disk.
                saved_path = _save_image(
                    img_bytes, image_output_dir, filename, img_idx, fmt,
                )

                # Name the image.
                name, caption = _name_image_from_captions(
                    img_idx, page_num, page_text, captions,
                )

                if not name or name.startswith(f"第{page_num}页图片"):
                    if use_llm_naming:
                        name = _name_image_with_llm(
                            img_bytes, page_text, img_idx, page_num,
                        )
                    # Fallback: use surrounding text summary.
                    if not name or name.startswith(f"第{page_num}页图片"):
                        name = page_text[:80] if page_text else f"第{page_num}页图片{img_idx + 1}"

                # ── VLM visual description (optional) ──────────────────
                vlm_description = ""
                if use_vlm_description:
                    vlm_description = _describe_image_with_vlm(img_bytes)
                    if vlm_description:
                        print(f"    ✓ 图片 {img_idx+1} VLM 描述已生成 ({len(vlm_description)}字)")

                # Link to the nearest L3 chunk on this page.
                l3_chunks = [c for c in page_chunks if c.get("chunk_level") == 3]
                linked_chunk_id = l3_chunks[0]["chunk_id"] if l3_chunks else ""
                if l3_chunks and img_idx < len(l3_chunks):
                    linked_chunk_id = l3_chunks[img_idx]["chunk_id"]

                image_entries.append({
                    "chunk_id": linked_chunk_id,
                    "image_path": saved_path,
                    "poi_name": name,
                    "poi_description": caption or name,
                    "distinguishing_features": vlm_description or caption or "",
                    "tags": f"图{img_idx + 1}, {doc_type}, p{page_num}",
                    "site": "",
                    "cave": "",
                })

                # ── Create VLM text chunk for retrieval ──────────────
                if vlm_description:
                    vlm_chunk_id = f"{filename}::p{page_num}::vlm::{img_idx}"
                    text_chunks.append({
                        **base_doc,
                        "text": f"[图片视觉描述] {vlm_description}",
                        "chunk_id": vlm_chunk_id,
                        "parent_chunk_id": page_parent_id,
                        "root_chunk_id": page_root_id,
                        "chunk_level": 3,
                        "chunk_idx": page_global_chunk_idx + 1000 + img_idx,
                        "site": "",
                        "cave": "",
                        "poi_name": name,
                    })

        return text_chunks, image_entries

    def load_documents_from_folder(
        self,
        folder_path: str,
        image_output_dir: str | Path = "data/reference_images",
        use_llm_naming: bool = False,
    ) -> tuple[list[dict], list[dict]]:
        """
        Load all supported documents from a folder.

        Returns: (all_text_chunks, all_image_entries)
        """
        all_text_chunks: list[dict] = []
        all_image_entries: list[dict] = []

        for filename in os.listdir(folder_path):
            file_lower = filename.lower()
            if not (
                file_lower.endswith(".pdf")
                or file_lower.endswith((".docx", ".doc"))
            ):
                continue

            file_path = os.path.join(folder_path, filename)
            try:
                text_chunks, image_entries = self.load_document(
                    file_path, filename, image_output_dir, use_llm_naming,
                )
                all_text_chunks.extend(text_chunks)
                all_image_entries.extend(image_entries)
                print(f"  ✓ {filename}: {len(text_chunks)} 文本块, {len(image_entries)} 图片")
            except Exception as e:
                print(f"  ✗ {filename}: {str(e)}")
                continue

        return all_text_chunks, all_image_entries


# ═══════════════════════════════════════════════════════════════════
# Ingestion helper
# ═══════════════════════════════════════════════════════════════════

def ingest_document(
    file_path: str,
    use_llm_naming: bool = False,
    use_vlm_description: bool = False,
) -> None:
    """
    End-to-end ingestion: document → chunking → PostgreSQL + dual Milvus.

    Args:
        file_path: Path to the document.
        use_llm_naming: If True, use LLM to name images without captions.
        use_vlm_description: If True, use Doubao Vision to generate pure
            visual descriptions for images. Descriptions replace image
            placeholders in text and are indexed as L3 chunks.
    """
    from backend.milvus_writer import milvus_writer
    from backend.parent_chunk_store import parent_chunk_store

    file_path = Path(file_path)
    filename = file_path.name

    print(f"正在处理: {filename}")
    loader = DocumentLoader()
    text_chunks, image_entries = loader.load_document(
        str(file_path), filename,
        use_llm_naming=use_llm_naming,
        use_vlm_description=use_vlm_description,
    )

    # Separate L1/L2 (parent chunks) from L3 (leaf chunks).
    l1_l2_chunks = [c for c in text_chunks if c.get("chunk_level") in (1, 2)]
    l3_chunks = [c for c in text_chunks if c.get("chunk_level") == 3]

    # Write L1/L2 → PostgreSQL.
    if l1_l2_chunks:
        count = parent_chunk_store.upsert_documents(l1_l2_chunks)
        print(f"  ✓ L1/L2 父块 → PostgreSQL: {count} 条")

    # Write L3 text → Text Milvus.
    if l3_chunks:
        milvus_writer.write_text_chunks(
            l3_chunks,
            progress_callback=lambda done, total: print(
                f"\r  ⏳ L3 文本向量化: {done}/{total}", end=""
            ),
        )
        print(f"\n  ✓ L3 文本块 → Text Milvus: {len(l3_chunks)} 条")

    # Write images → Image Milvus.
    if image_entries:
        milvus_writer.write_image_pois(
            image_entries,
            progress_callback=lambda done, total: print(
                f"\r  ⏳ 图片向量化: {done}/{total}", end=""
            ),
        )
        print(f"\n  ✓ 图片 → Image Milvus: {len(image_entries)} 条")

    # Summary.
    print(
        f"\n  摄入完成: {len(l1_l2_chunks)} L1/L2 + "
        f"{len(l3_chunks)} L3 + {len(image_entries)} 图片"
    )
