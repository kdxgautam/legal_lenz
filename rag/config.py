import os


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "project-3e52c857-15d9-4fe6-b2f")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gemini-3.5-flash-lite")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2")
EMBEDDING_DIM = 768

UPLOAD_BUCKET = os.getenv("UPLOAD_BUCKET", "")
DATABASE_URL = os.getenv("DATABASE_URL", "")
INSTANCE_CONNECTION_NAME = os.getenv("INSTANCE_CONNECTION_NAME", "")
DB_NAME = os.getenv("DB_NAME", "legal_lenz")
DB_USER = os.getenv("DB_USER", "")

APPROVED_EMAILS = {
    email.strip().lower()
    for email in os.getenv("APPROVED_EMAILS", "").split(",")
    if email.strip()
}

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_UPLOAD_PAGES = 300
UPLOAD_RETENTION_DAYS = 7

VECTOR_ORIGINAL_K = _int("VECTOR_ORIGINAL_K", 8)
VECTOR_REWRITE_K = _int("VECTOR_REWRITE_K", 8)
FTS_K = _int("FTS_K", 8)
FUSION_K = _int("FUSION_K", 20)
RERANK_CANDIDATES = _int("RERANK_CANDIDATES", 15)
FINAL_CONTEXT_K = _int("FINAL_CONTEXT_K", 8)
RRF_CONSTANT = _int("RRF_CONSTANT", 60)
EXACT_RRF_WEIGHT = _float("EXACT_RRF_WEIGHT", 3.0)
RERANK_MIN_SCORE = _float("RERANK_MIN_SCORE", 0.5)
