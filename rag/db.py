from __future__ import annotations

import json
import re
import ssl
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from sqlalchemy import create_engine, text

from rag.config import (
    DATABASE_URL,
    DB_NAME,
    DB_USER,
    INSTANCE_CONNECTION_NAME,
    UPLOAD_RETENTION_DAYS,
)

_engine = None
_connector = None


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    document_name: str
    page: int
    article: str | None
    content: str
    citation: int | None = None
    metadata: dict = field(default_factory=dict)
    distance: float | None = None
    lexical_score: float | None = None


def sqlalchemy_url(url: str) -> str:
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+pg8000://", 1)
    return url


def engine_args(url: str) -> dict:
    parsed = urlsplit(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    sslmode = query.pop("sslmode", None)
    query.pop("channel_binding", None)
    cleaned_url = urlunsplit(parsed._replace(query=urlencode(query)))
    args = {"url": sqlalchemy_url(cleaned_url), "pool_pre_ping": True}
    if sslmode in {"require", "verify-ca", "verify-full"}:
        args["connect_args"] = {"ssl_context": ssl.create_default_context()}
    return args


def engine():
    global _connector, _engine
    if _engine:
        return _engine
    if DATABASE_URL:
        _engine = create_engine(**engine_args(DATABASE_URL))
        return _engine
    if not (INSTANCE_CONNECTION_NAME and DB_USER):
        raise RuntimeError("Set DATABASE_URL or INSTANCE_CONNECTION_NAME and DB_USER.")
    from google.cloud.sql.connector import Connector

    _connector = Connector()

    def getconn():
        return _connector.connect(
            INSTANCE_CONNECTION_NAME,
            "pg8000",
            user=DB_USER,
            db=DB_NAME,
            enable_iam_auth=True,
        )

    _engine = create_engine("postgresql+pg8000://", creator=getconn, pool_pre_ping=True)
    return _engine


@contextmanager
def tx():
    with engine().begin() as conn:
        yield conn


def vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{v:.8f}" for v in values) + "]"


def new_document(
    owner_email: str | None,
    filename: str,
    gcs_object: str,
    doc_type: str,
    metadata: dict | None = None,
) -> str:
    document_id = str(uuid.uuid4())
    expires_at = None
    if doc_type == "upload":
        expires_at = datetime.now(timezone.utc) + timedelta(days=UPLOAD_RETENTION_DAYS)
    with tx() as conn:
        conn.execute(
            text(
                """
                INSERT INTO documents
                    (id, owner_email, original_filename, gcs_object, type, status, expires_at, metadata)
                VALUES
                    (:id, :owner, :filename, :gcs_object, :type, 'processing', :expires_at,
                     CAST(:metadata AS jsonb))
                """
            ),
            {
                "id": document_id,
                "owner": owner_email.lower() if owner_email else None,
                "filename": filename,
                "gcs_object": gcs_object,
                "type": doc_type,
                "expires_at": expires_at,
                "metadata": json.dumps(metadata or {}),
            },
        )
    return document_id


def set_document_status(document_id: str, status: str) -> None:
    with tx() as conn:
        conn.execute(text("UPDATE documents SET status = :status WHERE id = :id"), {"id": document_id, "status": status})


def insert_chunks(document_id: str, rows: list[dict]) -> None:
    if not rows:
        return
    with tx() as conn:
        document = conn.execute(
            text("SELECT original_filename, type, metadata FROM documents WHERE id = :id"),
            {"id": document_id},
        ).one()._mapping
        document_metadata = {
            "document_name": document["original_filename"],
            "document_type": document["type"],
            **(document["metadata"] or {}),
        }
        conn.execute(
            text(
                """
                INSERT INTO chunks (id, document_id, page, article, content, embedding, metadata)
                VALUES (:id, :document_id, :page, :article, :content, CAST(:embedding AS vector),
                        CAST(:metadata AS jsonb))
                """
            ),
            [
                {
                    "id": str(uuid.uuid4()),
                    "document_id": document_id,
                    "page": row["page"],
                    "article": row.get("article"),
                    "content": row["content"],
                    "embedding": vector_literal(row["embedding"]),
                    "metadata": json.dumps(document_metadata | row.get("metadata", {})),
                }
                for row in rows
            ],
        )


def list_documents(owner_email: str) -> list[dict]:
    with engine().connect() as conn:
        return [dict(row._mapping) for row in conn.execute(
            text(
                """
                SELECT id, original_filename, type, status, created_at, expires_at
                FROM documents
                WHERE (
                    type IN ('constitution', 'statute')
                    OR (owner_email = :email AND expires_at > now())
                )
                AND status IN ('ready', 'processing', 'failed')
                ORDER BY type, created_at DESC
                """
            ),
            {"email": owner_email.lower()},
        )]


def get_owned_document(owner_email: str, document_id: str) -> dict | None:
    with engine().connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT id, gcs_object, type
                FROM documents
                WHERE id = :id
                  AND owner_email = :email
                  AND type = 'upload'
                  AND expires_at > now()
                """
            ),
            {"id": document_id, "email": owner_email.lower()},
        ).first()
    return dict(row._mapping) if row else None


def delete_document_row(document_id: str) -> None:
    with tx() as conn:
        conn.execute(text("DELETE FROM documents WHERE id = :id"), {"id": document_id})


def expired_documents(limit: int = 100) -> list[dict]:
    with engine().connect() as conn:
        return [dict(row._mapping) for row in conn.execute(
            text(
                """
                SELECT id, gcs_object
                FROM documents
                WHERE type = 'upload' AND expires_at <= now()
                ORDER BY expires_at
                LIMIT :limit
                """
            ),
            {"limit": limit},
        )]


def previous_statute_documents(act_short_name: str, exclude_id: str) -> list[dict]:
    with engine().connect() as conn:
        return [dict(row._mapping) for row in conn.execute(
            text(
                """
                SELECT id, gcs_object
                FROM documents
                WHERE type = 'statute'
                  AND metadata->>'act_short_name' = :act
                  AND id <> :exclude_id
                """
            ),
            {"act": act_short_name, "exclude_id": exclude_id},
        )]


_SCOPE_SQL = """
    d.status = 'ready'
    AND (
        d.type IN ('constitution', 'statute')
        OR (
            d.type = 'upload'
            AND d.owner_email = :email
            AND d.expires_at > now()
            AND d.id::text = ANY(:ids)
        )
    )
"""

def _chunk_select(article: str = "c.article") -> str:
    return f"""
        c.id, c.document_id, d.original_filename AS document_name, c.page,
        {article} AS article, c.content,
        jsonb_build_object('document_type', d.type) || d.metadata || c.metadata AS metadata
    """


def _params(owner_email: str, selected_ids: list[str]) -> dict:
    return {"email": owner_email.lower(), "ids": selected_ids}


def _chunk(data: dict) -> Chunk:
    data["id"] = str(data["id"])
    data["document_id"] = str(data["document_id"])
    if isinstance(data.get("metadata"), str):
        data["metadata"] = json.loads(data["metadata"])
    return Chunk(**data)


def _chunks(rows) -> list[Chunk]:
    return [_chunk(dict(row._mapping)) for row in rows]


def vector_search(
    owner_email: str,
    selected_ids: list[str],
    query_embedding: list[float],
    limit: int,
) -> list[Chunk]:
    params = _params(owner_email, selected_ids) | {
        "embedding": vector_literal(query_embedding),
        "limit": limit,
    }
    sql = f"""
        SELECT {_chunk_select()}, c.embedding <=> CAST(:embedding AS vector) AS distance
        FROM chunks c JOIN documents d ON d.id = c.document_id
        WHERE {_SCOPE_SQL}
        ORDER BY distance
        LIMIT :limit
    """
    with engine().connect() as conn:
        return _chunks(conn.execute(text(sql), params))


def full_text_search(
    owner_email: str,
    selected_ids: list[str],
    query: str,
    limit: int,
) -> list[Chunk]:
    if not query.strip():
        return []
    params = _params(owner_email, selected_ids) | {"query": query, "limit": limit}
    sql = f"""
        WITH q AS (SELECT websearch_to_tsquery('english', :query) AS value)
        SELECT {_chunk_select()}, NULL::float AS distance,
               ts_rank_cd(c.search_vector, q.value) AS lexical_score
        FROM chunks c JOIN documents d ON d.id = c.document_id CROSS JOIN q
        WHERE {_SCOPE_SQL} AND c.search_vector @@ q.value
        ORDER BY lexical_score DESC, c.id
        LIMIT :limit
    """
    with engine().connect() as conn:
        return _chunks(conn.execute(text(sql), params))


def exact_search(
    owner_email: str,
    selected_ids: list[str],
    statute_references: list[tuple[str, str]],
    articles: list[str],
    limit: int,
) -> list[Chunk]:
    params = _params(owner_email, selected_ids) | {"limit": limit}
    statute_sql = f"""
        SELECT {_chunk_select()}, NULL::float AS distance
        FROM chunks c JOIN documents d ON d.id = c.document_id
        WHERE {_SCOPE_SQL}
          AND d.type = 'statute'
          AND d.metadata->>'act_short_name' = :act
          AND c.metadata->>'section_number' = :section
        ORDER BY (c.metadata->>'chunk_index')::integer
        LIMIT :limit
    """
    article_sql = f"""
        SELECT {_chunk_select(':article')}, NULL::float AS distance
        FROM chunks c JOIN documents d ON d.id = c.document_id
        WHERE {_SCOPE_SQL}
          AND d.type = 'constitution'
          AND (
              upper(c.article) = upper(:article)
              OR (c.article IS NOT NULL AND c.content ~* :article_pattern)
          )
          AND c.page = (
              SELECT min(c2.page)
              FROM chunks c2 JOIN documents d2 ON d2.id = c2.document_id
              WHERE d2.type = 'constitution'
                AND d2.status = 'ready'
                AND (
                    upper(c2.article) = upper(:article)
                    OR (c2.article IS NOT NULL AND c2.content ~* :article_pattern)
                )
          )
        ORDER BY c.page, c.id
        LIMIT :limit
    """
    rows = []
    with engine().connect() as conn:
        for act, section in statute_references:
            rows.extend(conn.execute(text(statute_sql), params | {"act": act, "section": section}))
        for article in articles:
            number = article.removeprefix("Article ")
            rows.extend(conn.execute(
                text(article_sql),
                params | {"article": article, "article_pattern": rf"(^|\n|\[)\s*{re.escape(number)}\.\s"},
            ))
    return _chunks(rows)


def retrieve(
    owner_email: str,
    selected_ids: list[str],
    query_embedding: list[float],
    article: str | None,
    statute_act: str | None = None,
    statute_section: str | None = None,
) -> list[Chunk]:
    """Backward-compatible vector/exact retrieval used by older callers."""
    exact = exact_search(
        owner_email,
        selected_ids,
        [(statute_act, statute_section)] if statute_act and statute_section else [],
        [article] if article else [],
        8,
    )
    seen = {chunk.id for chunk in exact}
    return (exact + [c for c in vector_search(owner_email, selected_ids, query_embedding, 8) if c.id not in seen])[:8]
