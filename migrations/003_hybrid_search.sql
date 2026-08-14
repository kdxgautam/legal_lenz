UPDATE chunks c
SET metadata = jsonb_build_object('document_name', d.original_filename) || d.metadata || c.metadata
FROM documents d
WHERE d.id = c.document_id
  AND (
      c.metadata->>'document_name' IS DISTINCT FROM d.original_filename
      OR NOT c.metadata @> d.metadata
  );

ALTER TABLE chunks ADD COLUMN IF NOT EXISTS search_vector tsvector
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

CREATE INDEX IF NOT EXISTS chunks_search_vector_gin_idx
ON chunks USING gin (search_vector);
