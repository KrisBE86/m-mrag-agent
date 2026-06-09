"""
文档加载，包含图片提取和三级层次分块。

- 支持 PDF（PyMuPDF 提取图片 + PyPDFLoader 提取文本）和 Word
  （python-docx 提取图片 + Docx2txtLoader 提取文本）。
- 三级滑动窗口分块（对齐 SuperMew）：
    L1 ≈ 1200 字符（粗略概述）
    L2 ≈ 600 字符（中等段落）
    L3 ≈ 300 字符（精细叶子节点，检索单元）
- 图片提取：从每页/每节提取嵌入图片，检测题注（如 "图1. xxx"），
  或使用 LLM 生成描述性名称。
- 每张提取的图片通过 chunk_id 关联到最近的 L3 文本块。
- 块 ID 格式：{filename}::p{page}::l{level}::{index}
"""

import base64
from html.parser import HTMLParser
import io
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import unquote, urljoin, urlparse

from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.document_cleaner import clean_document_pages
from backend.document_parser import parse_document

# ── 题注检测模式 ───────────────────────────────────

CAPTION_PATTERNS = [
    re.compile(r"图\s*(\d+)[\.\、\s]+(.+?)(?:\n|图\s*\d+|$)", re.DOTALL),
    re.compile(r"图版\s*(\d+)[\.\、\s]+(.+?)(?:\n|图版\s*\d+|$)", re.DOTALL),
    re.compile(r"Figure\s+(\d+)[\.\:\s]+(.+?)(?:\n|Figure\s+\d+|$)", re.DOTALL | re.IGNORECASE),
    re.compile(r"插图\s*(\d+)[\.\、\s]+(.+?)(?:\n|插图\s*\d+|$)", re.DOTALL),
]

MIN_EXTRACTED_IMAGE_SIDE = 80
IMAGE_PLACEHOLDER_PATTERN = re.compile(r"<!--\s*image\s*-->", re.IGNORECASE)


def _extract_captions(text: str) -> dict[int, str]:
    """在文本中查找所有图片题注。返回 {图片编号: 题注文本}。"""
    captions: dict[int, str] = {}
    for pattern in CAPTION_PATTERNS:
        for match in pattern.finditer(text):
            num = int(match.group(1))
            caption = match.group(2).strip()
            if num not in captions or len(caption) > len(captions[num]):
                captions[num] = caption
    return captions


# ═══════════════════════════════════════════════════════════════════
# 图片提取
# ═══════════════════════════════════════════════════════════════════


class _HTMLImageParser(HTMLParser):
    """Collect image references from HTML in document order."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.images: list[dict] = []

    def handle_starttag(self, tag: str, attrs):
        if tag.lower() != "img":
            return
        attr_map = {key.lower(): value for key, value in attrs if key}
        src = (attr_map.get("src") or attr_map.get("data-src") or "").strip()
        if not src:
            return
        self.images.append({
            "src": src,
            "alt": (attr_map.get("alt") or "").strip(),
            "title": (attr_map.get("title") or "").strip(),
            "class": (attr_map.get("class") or "").strip(),
            "width": (attr_map.get("width") or "").strip(),
            "height": (attr_map.get("height") or "").strip(),
        })


def _extension_from_image_type(content_type: str, src: str) -> str:
    mime = content_type.split(";", 1)[0].strip().lower()
    if mime == "image/jpeg":
        return "jpg"
    if mime == "image/png":
        return "png"
    if mime == "image/gif":
        return "gif"
    if mime == "image/webp":
        return "webp"
    if mime in {"image/jp2", "image/jpx", "image/jpm", "image/jpeg2000"}:
        return "jpx"
    suffix = Path(unquote(urlparse(src).path)).suffix.lower().lstrip(".")
    return suffix or "png"


def _decode_data_image(src: str) -> tuple[bytes, str] | None:
    header, sep, data = src.partition(",")
    if not sep or ";base64" not in header.lower() or not header.lower().startswith("data:image/"):
        return None
    fmt = header.split(";", 1)[0].split("/", 1)[-1] or "png"
    try:
        return base64.b64decode(data), fmt
    except Exception:
        return None


def _download_html_image(src: str, page_url: str | None, html_path: Path) -> tuple[bytes, str, str] | None:
    data_image = _decode_data_image(src)
    if data_image:
        image_bytes, fmt = data_image
        return image_bytes, fmt, "data:image"

    parsed = urlparse(src)
    if parsed.scheme in {"http", "https"} or page_url:
        try:
            import httpx
            from backend.url_ingestor import validate_public_http_url

            image_url = validate_public_http_url(urljoin(page_url or "", src))
            response = httpx.get(
                image_url,
                timeout=20.0,
                follow_redirects=True,
                headers={
                    "User-Agent": "MRagAgent-HTMLImageFetcher/1.0",
                    "Accept": "image/avif,image/webp,image/png,image/jpeg,image/*,*/*;q=0.8",
                },
            )
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if content_type and not content_type.lower().startswith("image/"):
                return None
            return response.content, _extension_from_image_type(content_type, image_url), image_url
        except Exception as e:
            print(f"  ⚠ HTML 图片下载失败 {src}: {e}")
            return None

    local_path = html_path.parent / unquote(parsed.path)
    if not local_path.exists() or not local_path.is_file():
        return None
    try:
        return local_path.read_bytes(), _extension_from_image_type("", str(local_path)), str(local_path)
    except Exception:
        return None


def _extract_images_from_html(file_path: str, source_url: str | None = None) -> list[dict]:
    """Extract and download HTML <img> resources in document order."""
    path = Path(file_path)
    try:
        html = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"  ⚠ HTML 读取失败，无法提取图片: {e}")
        return []

    parser = _HTMLImageParser()
    parser.feed(html)

    #这里暂时只针对敦煌研究院做了站点优先规则，降级后只根据图片大小做简单区分
    #TODO:设定通用规则
    content_refs = [
        ref for ref in parser.images
        if "img_vsb_content" in (ref.get("class") or "")
    ]
    image_refs = content_refs or parser.images

    images: list[dict] = []
    for img_idx, ref in enumerate(image_refs):
        downloaded = _download_html_image(ref["src"], source_url, path)
        if not downloaded:
            continue
        image_bytes, fmt, resolved_url = downloaded
        images.append({
            "page_number": 0,
            "image_index": img_idx,
            "image_bytes": image_bytes,
            "format": fmt,
            "alt": ref.get("alt", ""),
            "title": ref.get("title", ""),
            "source_image_url": resolved_url,
        })
    return images


def _extract_images_from_pdf(file_path: str) -> list[dict]:
    """
    使用 PyMuPDF (fitz) 从 PDF 文件中提取嵌入图片。
    返回 {page_number, image_index, image_bytes, format} 列表。
    """
    try:
        import fitz  # PyMuPDF 库
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
                try:
                    pix = fitz.Pixmap(doc, xref)
                    if pix.n - pix.alpha > 3:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    image_bytes = pix.tobytes("png")
                    ext = "png"
                except Exception:
                    image_bytes = base_image["image"]
                    ext = base_image.get("ext", "png")
                images.append({
                    "page_number": page_num + 1,  # 页码从 1 开始
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
    使用 python-docx 从 Word 文档中提取嵌入图片。
    返回 {paragraph_index, image_index, image_bytes, format} 列表。
    """
    try:
        from docx import Document
        from docx.opc.constants import RELATIONSHIP_TYPE as RT
    except ImportError:
        print("  ⚠ python-docx not installed. Install: pip install python-docx")
        return []

    images = []
    doc = Document(file_path)

    # 从文档的关系数据中提取图片。
    img_counter = 0
    for rel in doc.part.rels.values():
        if "image" not in rel.reltype:
            continue
        try:
            image_bytes = rel.target_part.blob
            # 从 target_ref 获取扩展名（如 "media/image1.png"）
            # ImagePart.ext 在较新版本的 python-docx 中已被移除
            import os as _os
            _, ext = _os.path.splitext(rel.target_ref)
            ext = (ext or ".png").lstrip(".")
            images.append({
                "page_number": 0,  # Word 没有页码概念；将按段落关联
                "image_index": img_counter,
                "image_bytes": image_bytes,
                "format": ext,
            })
            img_counter += 1
        except Exception:
            continue

    return images


def _save_image(
    image_bytes: bytes,
    output_dir: str | Path,
    filename: str,
    img_index: int,
    fmt: str,
) -> str | None:
    """将提取的图片统一保存为 PNG；过小装饰图返回 None。"""
    try:
        from PIL import Image
    except ImportError:
        print("  ⚠ Pillow not installed. Install: pip install Pillow")
        return None

    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()
    except Exception as e:
        print(f"  ⚠ 图片 {img_index + 1} 无法解码，已跳过: {e}")
        return None

    width, height = img.size
    if width < MIN_EXTRACTED_IMAGE_SIDE or height < MIN_EXTRACTED_IMAGE_SIDE:
        print(f"  ↷ 图片 {img_index + 1} 过小 ({width}x{height})，已跳过")
        return None

    if img.mode in {"RGBA", "LA"} or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        background.alpha_composite(rgba)
        img = background.convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_filename = re.sub(r"[^\w\-\.]", "_", filename)
    img_filename = f"{safe_filename}_img{img_index:03d}.png"
    img_path = output_dir / img_filename
    img.save(img_path, format="PNG")
    return str(img_path)


# ═══════════════════════════════════════════════════════════════════
# 图片命名（题注检测 + LLM 回退）
# ═══════════════════════════════════════════════════════════════════

def _name_image_from_captions(
    image_index: int,
    page_number: int,
    page_text: str,
    captions: dict[int, str],
) -> tuple[str, str]:
    """
    尝试根据检测到的题注为图片命名。
    返回 (名称, 题注文本) — 题注文本可能为空。
    """
    # 简单启发式：页面上的图片索引对应图片编号。
    figure_num = image_index + 1
    if figure_num in captions:
        caption = captions[figure_num]
        return caption, caption

    # 回退：使用页面级上下文。
    if page_text.strip():
        first_line = page_text.strip().split("\n")[0][:100]
        return first_line, ""
    return f"第{page_number}页图片{image_index + 1}", ""


def _caption_after_image_placeholder(page_text: str, image_index: int) -> str:
    """Use the first non-empty line after the matching image marker as a caption."""
    matches = list(IMAGE_PLACEHOLDER_PATTERN.finditer(page_text or ""))
    if image_index >= len(matches):
        return ""

    following = (page_text or "")[matches[image_index].end():]
    for line in following.splitlines():
        line = line.strip()
        if not line:
            continue
        if IMAGE_PLACEHOLDER_PATTERN.search(line):
            return ""
        if len(line) <= 120:
            return line.lstrip("# ").strip()
        return ""
    return ""


def _name_image_with_llm(
    image_bytes: bytes,
    page_text: str,
    image_index: int,
    page_number: int,
) -> str:
    """
    使用 LLM (DeepSeek) 为图片生成描述性名称。
    将图片以 base64 data URI 形式发送，附带周围页面文本作为上下文。
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
    """对 vlm_client.describe_image_with_vlm 的薄封装，用于文档入库。"""
    from backend.vlm_client import describe_image_with_vlm
    return describe_image_with_vlm(image_bytes)


def _format_image_description_block(record: dict) -> str:
    label = f"图{int(record.get('image_index', 0)) + 1}"
    name = (record.get("poi_name") or "").strip()
    caption = (record.get("caption") or "").strip()
    description = (record.get("vlm_description") or "").strip()

    heading = f"[图片视觉描述：{label}"
    if name:
        heading += f"｜{name}"
    heading += "]"

    lines = [heading]
    if caption and caption != name:
        lines.append(f"题注：{caption}")
    if description:
        lines.append(description)
    return "\n".join(lines)


def _inject_image_descriptions_into_page_text(page_text: str, image_records: list[dict]) -> str:
    """Insert VLM image descriptions into the corresponding page text before chunking."""
    records = [item for item in image_records if (item.get("vlm_description") or "").strip()]
    if not records:
        return page_text

    text = (page_text or "").strip()
    remaining_records: list[dict] = []
    for record in records:
        block = _format_image_description_block(record)
        text, replaced = IMAGE_PLACEHOLDER_PATTERN.subn(block, text, count=1)
        if not replaced:
            remaining_records.append(record)

    pending_blocks: list[str] = []

    for record in remaining_records:
        block = _format_image_description_block(record)
        caption = (record.get("caption") or "").strip()
        if caption and caption in text:
            text = text.replace(caption, f"{caption}\n\n{block}", 1)
        else:
            pending_blocks.append(block)

    if pending_blocks:
        appendix = "\n\n".join(["## 本页图片描述", *pending_blocks])
        text = f"{text}\n\n{appendix}" if text else appendix

    return text.strip()



# ═══════════════════════════════════════════════════════════════════
# 三级分块
# ═══════════════════════════════════════════════════════════════════

class DocumentLoader:
    """
    文档加载器：读取 PDF/Word 文档，提取图片，并执行
    三级层次分块。
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
        """将页面文本分割为 L1 → L2 → L3 层次结构。对齐 SuperMew。"""
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
        use_vlm_description: bool = True,
        source_url: str | None = None,
    ) -> tuple[list[dict], list[dict]]:
        """
        加载文档，提取图片并进行文本分块。

        Args:
            file_path: 文档文件路径。
            filename: 显示用的文件名，用于来源追踪。
            image_output_dir: 保存提取图片的目录。
            use_llm_naming: 若为 True 且未找到题注，则使用 LLM 为图片命名。
            use_vlm_description: 若为 True，则使用豆包视觉为每张提取的图片
                生成纯视觉描述。描述会插入对应页文本后再进入三级分块，
                同时存储在 distinguishing_features 中。

        Returns:
            (text_chunks, image_entries)
            - text_chunks: L1/L2/L3 块（L3 用于 Milvus 文本检索，L1/L2 用于 PostgreSQL）。
            - image_entries: 提取的图片及其元数据，用于 Image Milvus。
        """
        file_lower = filename.lower()

        if file_lower.endswith(".pdf"):
            doc_type = "PDF"
            raw_images = _extract_images_from_pdf(file_path)
        elif file_lower.endswith((".docx", ".doc")):
            doc_type = "Word"
            raw_images = _extract_images_from_docx(file_path)
        elif file_lower.endswith((".html", ".htm")):
            doc_type = "HTML"
            raw_images = _extract_images_from_html(file_path, source_url=source_url)
        else:
            raise ValueError(f"不支持的文件类型: {filename}")

        # ── Docling 结构化解析 ───────────────────────────────────
        try:
            parsed_doc = parse_document(
                file_path,
                filename=filename,
                source_url=source_url,
            )
        except Exception as e:
            raise Exception(f"处理文档失败: {str(e)}")

        # ── 文档清洗 ────────────────────────────────────────────
        cleaned_doc = clean_document_pages(
            [
                {
                    "text": page.text,
                    "page_number": page.page_number,
                    "metadata": page.metadata,
                }
                for page in parsed_doc.pages
            ],
            title=filename,
            metadata={
                "filename": filename,
                "file_type": doc_type,
                "source_url": source_url or "",
                "parser": parsed_doc.metadata.get("parser", "docling"),
            },
        )
        cleaned_pages_by_number = {
            page.page_number: page
            for page in cleaned_doc.pages
        }

        # ── 构建逐页数据 ──────────────────────────────────
        text_chunks: list[dict] = []
        image_entries: list[dict] = []

        # 按页分组图片。
        images_by_page: dict[int, list[dict]] = {}
        for img in raw_images:
            page = img.get("page_number", 0)
            images_by_page.setdefault(page, []).append(img)

        page_global_chunk_idx = 0
        saved_image_counter = 0
        for page in parsed_doc.pages:
            page_num = page.page_number
            cleaned_page = cleaned_pages_by_number.get(page_num)
            page_text = (cleaned_page.text if cleaned_page else "").strip()

            # ── 处理本页图片 ─────────────────────
            page_images = images_by_page.get(page_num, [])
            if not page_images and page_num == 0:
                # 同时匹配 page 0（Word 文档使用 0）。
                page_images = images_by_page.get(0, [])
            if not page_images and len(parsed_doc.pages) == 1:
                page_images = [img for items in images_by_page.values() for img in items]

            captions = _extract_captions(page_text)
            image_records: list[dict] = []

            for img_data in page_images:
                img_idx = img_data["image_index"]
                img_bytes = img_data["image_bytes"]
                fmt = img_data.get("format", "png")

                saved_index = saved_image_counter
                saved_image_counter += 1
                saved_path = _save_image(img_bytes, image_output_dir, filename, saved_index, fmt)
                if not saved_path:
                    continue

                saved_img_bytes = Path(saved_path).read_bytes()

                name, caption = _name_image_from_captions(
                    img_idx, page_num, page_text, captions,
                )
                html_label = (img_data.get("alt") or img_data.get("title") or "").strip()
                if html_label:
                    name = html_label
                    caption = html_label
                elif not caption:
                    marker_caption = _caption_after_image_placeholder(page_text, len(image_records))
                    if marker_caption:
                        name = marker_caption
                        caption = marker_caption

                if not name or name.startswith(f"第{page_num}页图片"):
                    if use_llm_naming:
                        name = _name_image_with_llm(
                            saved_img_bytes, page_text, img_idx, page_num,
                        )
                    if not name or name.startswith(f"第{page_num}页图片"):
                        name = page_text[:80] if page_text else f"第{page_num}页图片{img_idx + 1}"

                vlm_description = ""
                if use_vlm_description:
                    vlm_description = _describe_image_with_vlm(saved_img_bytes)
                    if vlm_description:
                        print(f"    ✓ 图片 {img_idx+1} VLM 描述已生成 ({len(vlm_description)}字)")

                image_records.append({
                    "image_index": img_idx,
                    "image_path": saved_path,
                    "poi_name": name,
                    "caption": caption,
                    "vlm_description": vlm_description,
                })

            page_text = _inject_image_descriptions_into_page_text(page_text, image_records)
            if not page_text:
                continue

            base_doc = {
                "filename": filename,
                "file_path": file_path,
                "file_type": doc_type,
                "page_number": page_num,
                "source_url": source_url or "",
            }

            # 对注入图片描述后的本页文本进行三级分块。
            page_chunks = self._split_page_to_three_levels(
                text=page_text,
                base_doc=base_doc,
                page_global_chunk_idx=page_global_chunk_idx,
            )
            page_global_chunk_idx += len(page_chunks)
            text_chunks.extend(page_chunks)

            for record in image_records:
                img_idx = int(record.get("image_index", 0))
                # 关联到本页最近的 L3 块。
                l3_chunks = [c for c in page_chunks if c.get("chunk_level") == 3]
                linked_chunk_id = l3_chunks[0]["chunk_id"] if l3_chunks else ""
                if l3_chunks and img_idx < len(l3_chunks):
                    linked_chunk_id = l3_chunks[img_idx]["chunk_id"]

                image_entries.append({
                    "chunk_id": linked_chunk_id,
                    "filename": filename,
                    "image_path": record.get("image_path", ""),
                    "poi_name": record.get("poi_name", ""),
                    "poi_description": record.get("caption") or record.get("poi_name", ""),
                    "distinguishing_features": record.get("vlm_description") or record.get("caption") or "",
                    "tags": f"图{img_idx + 1}, {doc_type}, p{page_num}",
                    "source_url": source_url or "",
                    "site": "",
                    "cave": "",
                })

        return text_chunks, image_entries

    def load_documents_from_folder(
        self,
        folder_path: str,
        image_output_dir: str | Path = "data/reference_images",
        use_llm_naming: bool = False,
    ) -> tuple[list[dict], list[dict]]:
        """
        从文件夹中加载所有支持的文档。

        Returns: (all_text_chunks, all_image_entries)
        """
        all_text_chunks: list[dict] = []
        all_image_entries: list[dict] = []

        for filename in os.listdir(folder_path):
            file_lower = filename.lower()
            if not (
                file_lower.endswith(".pdf")
                or file_lower.endswith((".docx", ".doc"))
                or file_lower.endswith((".html", ".htm"))
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
# 入库辅助函数
# ═══════════════════════════════════════════════════════════════════

def ingest_document(
    file_path: str,
    use_llm_naming: bool = False,
    use_vlm_description: bool = True,
    source_url: str | None = None,
) -> None:
    """
    端到端入库：文档 → 分块 → PostgreSQL + 双 Milvus 集合。

    Args:
        file_path: 文档路径。
        use_llm_naming: 若为 True，为缺少题注的图片使用 LLM 命名。
        use_vlm_description: 若为 True，使用豆包视觉为图片生成纯视觉描述。
            描述会插入对应页文本后再进入三级分块。
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
        source_url=source_url,
    )

    # 将 L1/L2（父块）与 L3（叶子块）分开。
    l1_l2_chunks = [c for c in text_chunks if c.get("chunk_level") in (1, 2)]
    l3_chunks = [c for c in text_chunks if c.get("chunk_level") == 3]

    # 写入 L1/L2 → PostgreSQL。
    if l1_l2_chunks:
        count = parent_chunk_store.upsert_documents(l1_l2_chunks)
        print(f"  ✓ L1/L2 父块 → PostgreSQL: {count} 条")

    # 写入 L3 文本 → Text Milvus。
    if l3_chunks:
        milvus_writer.write_text_chunks(
            l3_chunks,
            progress_callback=lambda done, total: print(
                f"\r  ⏳ L3 文本向量化: {done}/{total}", end=""
            ),
        )
        print(f"\n  ✓ L3 文本块 → Text Milvus: {len(l3_chunks)} 条")

    # 写入图片 → Image Milvus。
    if image_entries:
        milvus_writer.write_image_pois(
            image_entries,
            progress_callback=lambda done, total: print(
                f"\r  ⏳ 图片向量化: {done}/{total}", end=""
            ),
        )
        print(f"\n  ✓ 图片 → Image Milvus: {len(image_entries)} 条")

    # 汇总。
    print(
        f"\n  摄入完成: {len(l1_l2_chunks)} L1/L2 + "
        f"{len(l3_chunks)} L3 + {len(image_entries)} 图片"
    )
