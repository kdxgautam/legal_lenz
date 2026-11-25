from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path

import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag import db
from rag.config import MAX_UPLOAD_BYTES, MAX_UPLOAD_PAGES
from rag.llm import embed_texts
from rag.ocr import OcrError, ocr_enabled, ocr_pdf_pages
from rag.storage import delete_object, upload_pdf

logger = logging.getLogger(__name__)

MAX_ARTICLE_NUMBER = 395
MIN_PAGE_TEXT_CHARS = 100


class IngestError(ValueError):
    pass


def safe_filename(name: str) -> str:
    return Path(name or "document.pdf").name[:180] or "document.pdf"


def validate_pdf(data: bytes, max_pages: int | None = MAX_UPLOAD_PAGES) -> fitz.Document:
    if len(data) > MAX_UPLOAD_BYTES:
        raise IngestError("PDF must be 20 MB or smaller.")
    if not data.startswith(b"%PDF-"):
        raise IngestError("File does not look like a PDF.")
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise IngestError("PDF is corrupt or unreadable.") from exc
    if doc.is_encrypted:
        doc.close()
        raise IngestError("Encrypted PDFs are not supported.")
    if max_pages is not None and doc.page_count > max_pages:
        doc.close()
        raise IngestError("PDF must be 300 pages or fewer.")
    return doc


def clean_text(text: str, strip_footnotes: bool = False) -> str:
    if strip_footnotes:
        # Constitution pages separate amendment footnotes with a line of underscores.
        text = re.split(r"_{2,}", text)[0]
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n+", "\n", text).strip()


def extract_articles(text: str) -> list[dict]:
    articles = []
    current_article = None
    current_content = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        match = re.match(r"^(?:\d+\[)?(\d+[A-Z]?)\.\s", line)
        if match:
            number = match.group(1)
            if int(re.match(r"\d+", number).group()) > MAX_ARTICLE_NUMBER:
                continue
            if current_article and current_content:
                articles.append({"article": current_article, "content": "\n".join(current_content)})
            current_article = f"Article {number}"
            current_content = [line]
        elif current_article:
            current_content.append(line)
    if current_article and current_content:
        articles.append({"article": current_article, "content": "\n".join(current_content)})
    return articles


def page_texts(doc: fitz.Document, document_type: str) -> dict[int, str]:
    strip_footnotes = document_type == "constitution"
    return {
        index: clean_text(page.get_text(), strip_footnotes)
        for index, page in enumerate(doc, start=1)
    }


def with_ocr_text(data: bytes, texts: dict[int, str]) -> dict[int, str]:
    """Fill pages without a usable text layer, such as scans, from OCR."""
    missing = [page for page, text in texts.items() if len(text) < MIN_PAGE_TEXT_CHARS]
    if not missing or not ocr_enabled():
        return texts
    ocr_texts = ocr_pdf_pages(data, missing)
    return texts | {
        page: text for page, text in ocr_texts.items() if page in texts and len(text) > len(texts[page])
    }


def units_from_texts(texts: dict[int, str], document_type: str) -> list[dict]:
    units = []
    for index, text in texts.items():
        if len(text) < MIN_PAGE_TEXT_CHARS:
            continue
        articles = extract_articles(text) if document_type == "constitution" else []
        if articles:
            units.extend({"page": index, **article} for article in articles)
        else:
            units.append({"page": index, "article": None, "content": text})
    return units


def page_units(doc: fitz.Document, document_type: str) -> list[dict]:
    return units_from_texts(page_texts(doc, document_type), document_type)


def chunk_units(units: list[dict]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = []
    for unit in units:
        for part in splitter.split_text(unit["content"]):
            chunks.append({"page": unit["page"], "article": unit["article"], "content": part})
    return chunks


def ingest_pdf(data: bytes, filename: str, document_type: str, owner_email: str | None = None) -> str:
    if document_type not in {"constitution", "upload"}:
        raise IngestError("document_type must be constitution or upload.")
    if document_type == "upload" and not owner_email:
        raise IngestError("Uploads require an authenticated owner.")
    doc = validate_pdf(data, MAX_UPLOAD_PAGES if document_type == "upload" else None)
    filename = "Constitution of India.pdf" if document_type == "constitution" else safe_filename(filename)
    object_name = f"{document_type}/{uuid.uuid4()}.pdf"
    document_id = None
    try:
        upload_pdf(data, object_name)
        document_id = db.new_document(owner_email, filename, object_name, document_type)
        texts = page_texts(doc, document_type)
        ocr_failed = False
        try:
            texts = with_ocr_text(data, texts)
        except OcrError:
            logger.exception("OCR failed for %s", filename)
            ocr_failed = True
        chunks = chunk_units(units_from_texts(texts, document_type))
        if not chunks:
            if ocr_failed:
                raise IngestError("Text recognition failed for this scanned PDF. Please try again later.")
            raise IngestError("No readable text was found in this PDF.")
        embeddings = embed_texts([chunk["content"] for chunk in chunks], "RETRIEVAL_DOCUMENT")
        db.insert_chunks(document_id, [{**chunk, "embedding": embedding} for chunk, embedding in zip(chunks, embeddings)])
        db.set_document_status(document_id, "ready")
        return document_id
    except Exception:
        if document_id:
            db.set_document_status(document_id, "failed")
        delete_object(object_name)
        raise
    finally:
        doc.close()
