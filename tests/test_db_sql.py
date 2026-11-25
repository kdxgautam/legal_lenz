from rag.db import _sanitize_sources, title_from_question, engine_args, sqlalchemy_url, vector_literal


def test_vector_literal_is_pgvector_input():
    assert vector_literal([1, 0.25]) == "[1.00000000,0.25000000]"


def test_plain_postgres_url_uses_pg8000():
    assert sqlalchemy_url("postgresql://u:p@h/db") == "postgresql+pg8000://u:p@h/db"


def test_neon_sslmode_becomes_pg8000_ssl_context():
    args = engine_args("postgresql://u:p@h/db?sslmode=require&channel_binding=require")
    assert args["url"] == "postgresql+pg8000://u:p@h/db"
    assert "ssl_context" in args["connect_args"]


def test_chat_titles_are_short_and_sources_drop_excerpts():
    assert title_from_question("  hello   world  ") == "hello world"
    assert len(title_from_question("x" * 100)) == 80
    assert _sanitize_sources([{"citation": 1, "text": "private excerpt", "metadata": {"section": "1"}}]) == [
        {"citation": 1, "metadata": {"section": "1"}}
    ]


def test_iterative_scan_requires_pgvector_0_8():
    from rag.db import _pgvector_version

    assert _pgvector_version("0.8.0") >= (0, 8)
    assert _pgvector_version("0.10.1") >= (0, 8)
    assert _pgvector_version("0.7.4") < (0, 8)
    assert _pgvector_version(None) < (0, 8)


def test_insert_chunks_batches_rows(monkeypatch):
    from contextlib import contextmanager

    from rag import db

    class FakeResult:
        def one(self):
            return type("Row", (), {"_mapping": {"original_filename": "a.pdf", "type": "upload", "metadata": {}}})()

    statements = []

    class FakeConn:
        def execute(self, statement, params=None):
            statements.append((str(statement), params))
            return FakeResult()

    @contextmanager
    def fake_tx():
        yield FakeConn()

    monkeypatch.setattr(db, "tx", fake_tx)
    rows = [{"page": 1, "article": None, "content": f"chunk {i}", "embedding": [0.0]} for i in range(250)]
    db.insert_chunks("doc-1", rows)
    inserts = [params for sql, params in statements if sql.startswith("INSERT INTO chunks")]
    assert [sum(key.startswith("content_") for key in params) for params in inserts] == [100, 100, 50]
    assert inserts[2]["content_49"] == "chunk 249"
