"""
ImageParser — 使用 VLM（Qwen2.5-VL-7B-Instruct-4bit via mlx-vlm）生成图片描述，
作为单页文档返回，走标准文本 pipeline（chunk → embed → Qdrant + FTS5）。

支持格式：.jpg / .jpeg / .png / .webp / .heic / .heif

VLM 懒加载，首次处理图片时自动下载并加载（~4GB 4-bit 量化）。
在 ml_executor 线程内运行，与 Reranker / LLM 共享 Metal command queue。
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.ingest.pdf_analyzer import PageContent, ParsedDocument

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = (
    "请用中文详细描述这张图片的内容，"
    "包括图片中的文字、图表、场景、人物、颜色等所有可见信息。"
    "如果图片包含表格或数据，请列出关键数据。"
    "输出纯文字描述，不要使用 Markdown 格式。"
)


class ImageParser:
    def __init__(
        self,
        vlm_model: str = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
        prompt: str = DEFAULT_PROMPT,
        max_tokens: int = 512,
    ):
        self._vlm_model_name = vlm_model
        self._prompt = prompt
        self._max_tokens = max_tokens
        self._model = None
        self._processor = None
        self._config = None

    def _ensure_model(self):
        if self._model is not None:
            return
        from mlx_vlm import load
        from mlx_vlm.utils import load_config
        logger.info(f"[image_parser] Loading VLM: {self._vlm_model_name}")
        self._model, self._processor = load(self._vlm_model_name)
        self._config = load_config(self._vlm_model_name)
        logger.info("[image_parser] VLM ready")

    def _normalize_image(self, file_path: Path) -> Path:
        """将 HEIC/HEIF 转换为 PNG，其余格式直接返回原路径。"""
        suffix = file_path.suffix.lower()
        if suffix in (".heic", ".heif"):
            try:
                import pillow_heif
                pillow_heif.register_heif_opener()
            except ImportError:
                raise ImportError("pillow-heif required for HEIC support: pip install pillow-heif")
        if suffix in (".heic", ".heif"):
            from PIL import Image
            import tempfile
            img = Image.open(str(file_path))
            fd, tmp_path = tempfile.mkstemp(suffix=".png")
            import os
            os.close(fd)
            tmp = Path(tmp_path)
            img.save(str(tmp), "PNG")
            return tmp
        return file_path

    def parse(self, file_path: Path) -> ParsedDocument:
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        self._ensure_model()

        # Normalize image format
        image_path = self._normalize_image(file_path)
        tmp_created = image_path != file_path

        try:
            messages = [{"role": "user", "content": self._prompt}]
            prompt = apply_chat_template(
                self._processor,
                self._config,
                messages,
                num_images=1,
            )
            result = generate(
                self._model,
                self._processor,
                prompt,
                image=str(image_path),
                max_tokens=self._max_tokens,
                verbose=False,
            )
            description = result.text.strip()
            logger.info(f"[image_parser] {file_path.name}: {len(description)} chars description")
        finally:
            if tmp_created and image_path.exists():
                image_path.unlink()

        if not description:
            description = f"[图片文件: {file_path.name}，VLM 描述生成失败]"

        page = PageContent(page_num=1, text=description, headers=[], is_ocr=True)
        return ParsedDocument(
            file_path=file_path,
            file_name=file_path.name,
            total_pages=1,
            is_scanned=True,
            pages=[page],
        )
