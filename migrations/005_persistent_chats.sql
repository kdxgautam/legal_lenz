CREATE TABLE IF NOT EXISTS chat_sessions (
    id uuid PRIMARY KEY,
    owner_email text NOT NULL,
    title text NOT NULL CHECK (length(trim(title)) > 0),
    selected_document_ids jsonb NOT NULL DEFAULT '[]'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    last_activity_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id bigserial PRIMARY KEY,
    session_id uuid NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role text NOT NULL CHECK (role IN ('user', 'assistant')),
    content text NOT NULL,
    sources jsonb NOT NULL DEFAULT '[]'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS chat_sessions_owner_activity_idx
    ON chat_sessions (owner_email, last_activity_at DESC);
CREATE INDEX IF NOT EXISTS chat_messages_session_order_idx
    ON chat_messages (session_id, id);
