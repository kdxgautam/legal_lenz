CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id uuid PRIMARY KEY,
    owner_email text,
    original_filename text NOT NULL,
    gcs_object text NOT NULL,
    type text NOT NULL CHECK (type IN ('constitution', 'upload')),
    status text NOT NULL CHECK (status IN ('processing', 'ready', 'failed')),
    created_at timestamptz NOT NULL DEFAULT now(),
    expires_at timestamptz,
    CHECK (
        (type = 'constitution' AND owner_email IS NULL AND expires_at IS NULL)
        OR
        (type = 'upload' AND owner_email IS NOT NULL AND expires_at IS NOT NULL)
    )
);

CREATE TABLE IF NOT EXISTS chunks (
    id uuid PRIMARY KEY,
    document_id uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page integer NOT NULL CHECK (page >= 1),
    article text,
    content text NOT NULL,
    embedding vector(768) NOT NULL
);

CREATE INDEX IF NOT EXISTS documents_owner_idx ON documents(owner_email, expires_at);
CREATE INDEX IF NOT EXISTS documents_type_status_idx ON documents(type, status);
CREATE INDEX IF NOT EXISTS chunks_document_idx ON chunks(document_id);
CREATE INDEX IF NOT EXISTS chunks_article_idx ON chunks(article);
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx ON chunks USING hnsw (embedding vector_cosine_ops);
