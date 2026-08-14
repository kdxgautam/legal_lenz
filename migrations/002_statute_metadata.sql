ALTER TABLE documents ADD COLUMN IF NOT EXISTS metadata jsonb NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS metadata jsonb NOT NULL DEFAULT '{}'::jsonb;

ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_type_check;
ALTER TABLE documents ADD CONSTRAINT documents_type_check
    CHECK (type IN ('constitution', 'statute', 'upload'));

ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_check;
ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_owner_expiry_check;
ALTER TABLE documents ADD CONSTRAINT documents_owner_expiry_check CHECK (
    (type IN ('constitution', 'statute') AND owner_email IS NULL AND expires_at IS NULL)
    OR
    (type = 'upload' AND owner_email IS NOT NULL AND expires_at IS NOT NULL)
);

CREATE INDEX IF NOT EXISTS documents_statute_short_name_idx
    ON documents ((metadata->>'act_short_name'))
    WHERE type = 'statute';
CREATE INDEX IF NOT EXISTS chunks_section_number_idx
    ON chunks ((metadata->>'section_number'))
    WHERE metadata ? 'section_number';
