DROP INDEX IF EXISTS chunks_search_vector_gin_idx;

ALTER TABLE chunks DROP COLUMN IF EXISTS search_vector;

ALTER TABLE chunks ADD COLUMN search_vector tsvector
GENERATED ALWAYS AS (
    setweight(to_tsvector('english', coalesce(content, '')), 'B') ||
    setweight(to_tsvector('english',
        coalesce(article, '') || ' ' ||
        coalesce(metadata->>'document_name', '') || ' ' ||
        coalesce(metadata->>'act_name', '') || ' ' ||
        coalesce(metadata->>'act_short_name', '') || ' ' ||
        coalesce(metadata->>'section_number', '') || ' ' ||
        coalesce(metadata->>'section_title', '') || ' ' ||
        coalesce(metadata->>'chapter_title', '')
    ), 'A')
) STORED;

CREATE INDEX chunks_search_vector_gin_idx ON chunks USING gin (search_vector);
