# Legal Lenz

Controlled Streamlit RAG service for approved Google users. It stores PDFs in private Cloud Storage, stores metadata/chunks/pgvector embeddings in PostgreSQL, and uses Gemini through ADC. The student setup uses Neon; Cloud SQL remains the Cloud Run production option.

## Local Setup

```bash
uv sync
cp .env.example .env
set -a
source .env
set +a
uv run python manage.py migrate
uv run python manage.py ingest-constitution data/pdfs/constitution.pdf
uv run python manage.py ingest-statute data/pdfs/bhartiya_nyay_sanhita.pdf --act BNS
uv run python manage.py ingest-statute 'data/pdfs/the_bharatiya_nagarik_suraksha_sanhita,_2023.pdf' --act BNSS
uv run python manage.py ingest-statute data/pdfs/bhartiya_sakshya.pdf --act BSA
uv run streamlit run app.py
```

For local Google ADC, keep `application_default_credentials.json` in this folder. It is ignored by Git.

## Streamlit Auth

Create `.streamlit/secrets.toml` locally or mount it from Secret Manager in Cloud Run:

```toml
[auth]
redirect_uri = "https://YOUR_CLOUD_RUN_URL/oauth2callback"
cookie_secret = "replace-with-long-random-secret"
client_id = "GOOGLE_OIDC_CLIENT_ID"
client_secret = "GOOGLE_OIDC_CLIENT_SECRET"
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
```

Only emails in `APPROVED_EMAILS` can enter the app after Google login.

## Operations

```bash
set -a
source .env
set +a
uv run python manage.py migrate
uv run python manage.py ingest-constitution data/pdfs/constitution.pdf
uv run python manage.py ingest-statute data/pdfs/bhartiya_nyay_sanhita.pdf --act BNS
uv run python manage.py ingest-statute 'data/pdfs/the_bharatiya_nagarik_suraksha_sanhita,_2023.pdf' --act BNSS
uv run python manage.py ingest-statute data/pdfs/bhartiya_sakshya.pdf --act BSA
uv run python manage.py cleanup
uv run python manage.py debug-retrieval "What does Section 303 of BNS say?" --email you@example.com
uv run python evals/run_retrieval_eval.py --email you@example.com
uv run pytest
docker build .
```

`manage.py migrate` applies numbered SQL files, including chat tables and search indexes. `manage.py cleanup` removes expired uploads and chat sessions inactive for 30 days.

Deployment helpers:

```bash
UPLOAD_BUCKET=legal-lenz-private ./infra/provision.sh
INSTANCE_CONNECTION_NAME=... DB_NAME=legal_lenz DB_USER=... UPLOAD_BUCKET=... APPROVED_EMAILS=... ./infra/deploy.sh
IMAGE=gcr.io/project-3e52c857-15d9-4fe6-b2f/legal-lenz:SHA ./infra/cleanup-job.sh
```

## Data Retention

Uploaded PDFs, metadata, and embeddings expire after seven days unless the user deletes them sooner. `manage.py cleanup` permanently deletes the GCS object and cascades database rows. Disable Cloud Storage soft deletion on the upload bucket so deletion is permanent.

Constitution and statute documents have no owner and no expiry. Chat sessions are stored in PostgreSQL per approved email, capped at five active sessions, and expire after 30 days of inactivity. Persisted citations keep metadata only; source excerpts from private uploads are not stored in chat history.

## Persistent Chats

Each approved user can keep up to five active chats. The sidebar lets users create, switch, rename, and permanently delete chats. A chat stores messages, source citation metadata, and that chat's selected upload IDs, so switching chats restores the relevant document selection.

User isolation is enforced in SQL: every chat read, write, rename, selection update, and delete includes the authenticated lowercase email. Expired, deleted, or foreign uploads are removed from restored selections and remain blocked by retrieval scope filters.

Persisted source records intentionally omit source excerpts. Excerpts are shown only for the live answer response so private uploaded text still disappears when the upload expires or is deleted.

## Statute Indexing

BNS and BNSS use their Arrangement of Sections for chapter/title metadata and their enacted body for content. BSA uses its Gazette margin titles. Sections are never merged: sections up to 1,000 characters remain whole, while larger sections split first at subsection boundaries and then use the existing 200-character-overlap splitter only when needed.

Act metadata lives in `documents.metadata`; chapter, section, subsection, page range, and chunk order live in `chunks.metadata`. For example, BNS Section 303 has document metadata containing `{"act_short_name": "BNS", "effective_from": "2024-07-01"}` and chunk metadata containing `{"chapter_number": "XVII", "section_number": "303", "section_title": "Theft", "page_end": 89, "chunk_index": 1}`.

`What does Section 303 of BNS say?` is analyzed as `BNS + 303`. SQL retrieves those chunks first in legal order, fills unused context slots with semantic pgvector results, deduplicates, caps context at eight, and sends the numbered sources to Gemini.

Known source quirks handled by the parser include front-matter section tables, BNS `255.—` punctuation, the missing BNSS Chapter V body heading, BNSS schedules, BSA margin titles, and joined Gazette headings such as `CHAPTERI`. The PDFs contain extractable text, so OCR is not used.

## Hybrid Retrieval V2

Natural-language questions receive one structured Gemini analysis containing a retrieval rewrite, legal domains, Acts, Sections, Articles, and keywords. Explicit Article and Act/Section references use deterministic parsing and skip this analysis call.

The original question and rewrite receive separate 768-dimensional pgvector searches. PostgreSQL full-text search uses a stored `tsvector` and GIN index, while exact references use direct metadata lookup. Results merge by stable chunk identity and use Reciprocal Rank Fusion before a structured Gemini rerank. If analysis or reranking fails, retrieval falls back to the original query or fused ranking. The answer model always receives the original question and authoritative source text.

All three SQL search paths share the same scope: ready Constitution/statutes are shared, while uploads must belong to the current email, be selected, and be unexpired. The CLI debug command prints each retrieval stage and is not exposed in the Streamlit UI.

Optional tuning variables are `VECTOR_ORIGINAL_K`, `VECTOR_REWRITE_K`, `FTS_K`, `FUSION_K`, `RERANK_CANDIDATES`, `FINAL_CONTEXT_K`, `RRF_CONSTANT`, `EXACT_RRF_WEIGHT`, and `RERANK_MIN_SCORE`. Defaults live in `rag/config.py`; they do not need to be added to `.env`.

Live evaluation on the indexed Constitution, BNS, BNSS, and BSA corpus produced:

| Metric | Original vector | V2 hybrid |
| --- | ---: | ---: |
| Recall@3 | 0.696 | 0.913 |
| Recall@5 | 0.739 | 0.913 |
| MRR | 0.691 | 0.955 |
| Exact-reference hit rate | 0.600 | 1.000 |

These are measured results from 22 cases; the upload case was skipped because no upload fixture was selected. Natural-language Act inference and reranking remain model-dependent. The current known weak cases are inferred Article 21A wording in the legacy Constitution index and occasional omission of BNS Section 303 from a cross-statute arrest question. Explicit references are deterministic and scored 1.000 across all ten exact cases.

## Production Checklist

1. Enable Run, Cloud Build, Cloud SQL Admin, Cloud Storage, Secret Manager, and Vertex AI APIs.
2. Create Cloud SQL PostgreSQL with the `vector` extension available.
3. Create a private bucket in `asia-south1` and disable soft deletion.
4. Create Secret Manager entry `legal-lenz-streamlit-secrets` containing Streamlit OIDC config.
5. Grant the Cloud Run service account Cloud SQL Client, Vertex AI User, Storage Object Admin on the bucket, and Secret Manager Secret Accessor.
6. Run migration through a job, then index the Constitution and three statutes.
7. Deploy one Cloud Run revision in `asia-south1`, smoke test auth/retrieval/upload/delete, then send traffic.
8. Smoke test chat create/switch/rename/delete and verify the five-chat limit for an approved email.
9. Schedule the cleanup job daily and alert on Cloud Run errors, cleanup failures, Gemini failures, and Cloud SQL pressure.
