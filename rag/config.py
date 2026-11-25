import os


def _setting(name: str, default=""):
    value = os.getenv(name)
    if value not in (None, ""):
        return value
    try:
        import streamlit as st

        return st.secrets.get(name, default)
    except Exception:
        return default


def _int(name: str, default: int) -> int:
    return int(_setting(name, default))


def _float(name: str, default: float) -> float:
    return float(_setting(name, default))

PROJECT_ID = _setting("GOOGLE_CLOUD_PROJECT", "project-2d06ae6e-1f76-495e-809")
LOCATION = _setting("GOOGLE_CLOUD_LOCATION", "global")
GENERATION_MODEL = _setting("GENERATION_MODEL", "gemini-3.5-flash-lite")
EMBEDDING_MODEL = _setting("EMBEDDING_MODEL", "gemini-embedding-2")
EMBEDDING_DIM = 768

UPLOAD_BUCKET = _setting("UPLOAD_BUCKET")
DATABASE_URL = _setting("DATABASE_URL")
INSTANCE_CONNECTION_NAME = _setting("INSTANCE_CONNECTION_NAME")
DB_NAME = _setting("DB_NAME", "legal_lenz")
DB_USER = _setting("DB_USER")

MISTRAL_API_KEY = _setting("MISTRAL_API_KEY")
MISTRAL_OCR_MODEL = _setting("MISTRAL_OCR_MODEL", "mistral-ocr-latest")

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_UPLOAD_PAGES = 300
UPLOAD_RETENTION_DAYS = 7
CHAT_RETENTION_DAYS = 30
MAX_CHAT_SESSIONS = 5

VECTOR_ORIGINAL_K = _int("VECTOR_ORIGINAL_K", 5)
VECTOR_REWRITE_K = _int("VECTOR_REWRITE_K", 5)
FTS_K = _int("FTS_K", 5)
FUSION_K = _int("FUSION_K", 10)
RERANK_CANDIDATES = _int("RERANK_CANDIDATES", 5)
FINAL_CONTEXT_K = _int("FINAL_CONTEXT_K", 5)
RRF_CONSTANT = _int("RRF_CONSTANT", 60)
EXACT_RRF_WEIGHT = _float("EXACT_RRF_WEIGHT", 3.0)
RERANK_MIN_SCORE = _float("RERANK_MIN_SCORE", 0.5)
