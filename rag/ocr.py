import base64
import re

import httpx

from rag.config import MISTRAL_API_KEY, MISTRAL_OCR_MODEL

OCR_URL = "https://api.mistral.ai/v1/ocr"
IMAGE_MARKDOWN_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")


class OcrError(RuntimeError):
    pass


def ocr_enabled() -> bool:
    return bool(MISTRAL_API_KEY)


def clean_markdown(markdown: str) -> str:
    text = IMAGE_MARKDOWN_RE.sub("", markdown)
    text = re.sub(r"\$\^\{([^}]*)\}\$", r"\1", text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n+", "\n", text).strip()


def ocr_pdf_pages(data: bytes, pages: list[int]) -> dict[int, str]:
    """OCR one-based PDF page numbers with Mistral and return cleaned text by page."""
    if not pages:
        return {}
    if not MISTRAL_API_KEY:
        raise OcrError("Set MISTRAL_API_KEY to OCR scanned PDFs.")
    try:
        response = httpx.post(
            OCR_URL,
            headers={"Authorization": f"Bearer {MISTRAL_API_KEY}"},
            json={
                "model": MISTRAL_OCR_MODEL,
                "document": {
                    "type": "document_url",
                    "document_url": "data:application/pdf;base64," + base64.b64encode(data).decode(),
                },
                "pages": [page - 1 for page in pages],
                "include_image_base64": False,
                "include_blocks": False,
            },
            timeout=300,
        )
        response.raise_for_status()
        result = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise OcrError(f"Mistral OCR request failed: {exc}") from exc
    return {
        item["index"] + 1: clean_markdown(item.get("markdown") or "")
        for item in result.get("pages", [])
    }
