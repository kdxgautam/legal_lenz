import os

import pytest
from sqlalchemy import create_engine, text

from rag import db


pytestmark = pytest.mark.skipif(
    not os.getenv("TEST_DATABASE_URL"),
    reason="set TEST_DATABASE_URL to run pgvector integration tests",
)


def test_pgvector_exact_statute_and_owner_isolation():
    old_engine = db._engine
    test_engine = create_engine(**db.engine_args(os.environ["TEST_DATABASE_URL"]))
    db._engine = test_engine
    ids = []
    vector = [0.0] * 767 + [1.0]
    try:
        with test_engine.begin() as conn:
            for migration in sorted(__import__("pathlib").Path("migrations").glob("*.sql")):
                for statement in [part.strip() for part in migration.read_text().split(";") if part.strip()]:
                    conn.execute(text(statement))

        statute_id = db.new_document(
            None,
            "BNS.pdf",
            "test/bns.pdf",
            "statute",
            {"act_short_name": "BNS"},
        )
        bnss_id = db.new_document(
            None,
            "BNSS.pdf",
            "test/bnss.pdf",
            "statute",
            {"act_short_name": "BNSS"},
        )
        owner_id = db.new_document("owner@example.com", "owner.pdf", "test/owner.pdf", "upload")
        other_id = db.new_document("other@example.com", "other.pdf", "test/other.pdf", "upload")
        ids.extend([statute_id, bnss_id, owner_id, other_id])
        db.insert_chunks(
            statute_id,
            [{
                "page": 88,
                "article": None,
                "content": "303. Theft.",
                "embedding": vector,
                "metadata": {"section_number": "303", "chunk_index": 1, "page_end": 88},
            }],
        )
        db.insert_chunks(
            bnss_id,
            [{
                "page": 10,
                "article": None,
                "content": "303. BNSS procedure.",
                "embedding": vector,
                "metadata": {"section_number": "303", "chunk_index": 1},
            }],
        )
        for document_id, content in [
            (owner_id, "employment termination clause owner text"),
            (other_id, "employment termination clause other text"),
        ]:
            db.insert_chunks(document_id, [{"page": 1, "article": None, "content": content, "embedding": vector}])
            db.set_document_status(document_id, "ready")
        db.set_document_status(statute_id, "ready")
        db.set_document_status(bnss_id, "ready")

        chunks = db.retrieve(
            "owner@example.com",
            [owner_id, other_id],
            vector,
            None,
            "BNS",
            "303",
        )

        assert chunks[0].metadata["section_number"] == "303"
        assert chunks[0].metadata["act_short_name"] == "BNS"
        assert all("other text" not in chunk.content for chunk in chunks)

        exact = db.exact_search("owner@example.com", [], [("BNS", "303")], [], 8)
        assert exact and all(chunk.metadata["act_short_name"] == "BNS" for chunk in exact)

        lexical = db.full_text_search(
            "owner@example.com",
            [owner_id, other_id],
            "employment termination clause",
            8,
        )
        assert any("owner text" in chunk.content for chunk in lexical)
        assert all("other text" not in chunk.content for chunk in lexical)

        with test_engine.begin() as conn:
            conn.execute(text("UPDATE documents SET expires_at = now() - interval '1 day' WHERE id = :id"), {"id": owner_id})
        assert all(
            "owner text" not in chunk.content
            for chunk in db.full_text_search("owner@example.com", [owner_id], "termination clause", 8)
        )
    finally:
        if ids:
            with test_engine.begin() as conn:
                conn.execute(text("DELETE FROM documents WHERE id::text = ANY(:ids)"), {"ids": ids})
        test_engine.dispose()
        db._engine = old_engine
