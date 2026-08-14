from rag.db import engine_args, sqlalchemy_url, vector_literal


def test_vector_literal_is_pgvector_input():
    assert vector_literal([1, 0.25]) == "[1.00000000,0.25000000]"


def test_plain_postgres_url_uses_pg8000():
    assert sqlalchemy_url("postgresql://u:p@h/db") == "postgresql+pg8000://u:p@h/db"


def test_neon_sslmode_becomes_pg8000_ssl_context():
    args = engine_args("postgresql://u:p@h/db?sslmode=require&channel_binding=require")
    assert args["url"] == "postgresql+pg8000://u:p@h/db"
    assert "ssl_context" in args["connect_args"]
