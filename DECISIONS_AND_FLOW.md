# Legal Lenz: Decisions And Flow

This file records important project decisions: when they were made, where they apply, what was decided, and why. Update it with architecture, dependency, security, data, or deployment changes.

## Decision Log

### DEC-001: Use Streamlit for the application UI

- **When:** Before 2026-08-14
- **Status:** Active
- **Where:** `app.py`
- **Decision:** Run Legal Lenz as a Streamlit chat application.
- **Reason:** Streamlit provides upload, auth, session state, chat controls, and browser UI with little code.
- **Consequence:** UI and request handling live in one Cloud Run process.

### DEC-002: Use local Chroma databases for retrieval

- **When:** Before 2026-08-14
- **Status:** Superseded by DEC-008
- **Where:** Formerly `rag/ingest.py`, `rag/retriever.py`, `data/chroma/`
- **Decision:** Store Constitution and upload indexes in local Chroma databases.
- **Reason:** Local persistence avoided a hosted vector database.
- **Consequence:** Not compatible with disposable Cloud Run storage.

### DEC-003: Separate base and uploaded-document indexes

- **When:** Before 2026-08-14
- **Status:** Superseded by DEC-008
- **Where:** Formerly `data/chroma/base`, `data/chroma/uploads/`
- **Decision:** Keep Constitution and uploaded documents in separate local indexes.
- **Reason:** Uploaded content needed isolation from the shared Constitution index.
- **Consequence:** Replaced by ownership-enforced SQL queries.

### DEC-004: Use local Hugging Face MiniLM embeddings

- **When:** Before 2026-08-14
- **Status:** Superseded by DEC-009
- **Where:** Formerly `rag/ingest.py`, `rag/retriever.py`
- **Decision:** Embed with `sentence-transformers/all-MiniLM-L6-v2`.
- **Reason:** Small local model with no embedding API dependency.
- **Consequence:** Replaced because production should use managed Gemini embeddings and avoid Torch/model downloads.

### DEC-005: Use Groq for answer generation

- **When:** Before 2026-08-14
- **Status:** Superseded by DEC-010
- **Where:** Formerly `rag/llm.py`, `.env`
- **Decision:** Generate with Groq `llama-3.3-70b-versatile`.
- **Reason:** It was the first wired LLM provider.
- **Consequence:** Removed to standardize on Google ADC and Gemini.

### DEC-006: Keep Google ADC credentials local to the project

- **When:** 2026-08-14
- **Status:** Active for local development only
- **Where:** `application_default_credentials.json`, `.gitignore`
- **Decision:** Keep local ADC in the project folder and exclude it from Git.
- **Reason:** Local Google client libraries can authenticate without global credential setup.
- **Consequence:** Cloud Run uses its service account instead.

### DEC-007: Use Python 3.12 and one dependency source

- **When:** Before 2026-08-14; reaffirmed 2026-08-15
- **Status:** Active
- **Where:** `pyproject.toml`, `uv.lock`
- **Decision:** Keep dependencies in `pyproject.toml` with `uv.lock`; remove `requirements.txt`.
- **Reason:** One dependency source avoids drift.
- **Consequence:** Install with `uv sync`.

### DEC-008: Store application data in Cloud SQL PostgreSQL and private Cloud Storage

- **When:** 2026-08-15
- **Status:** Active
- **Where:** `rag/db.py`, `rag/storage.py`, `migrations/001_initial.sql`, `manage.py`
- **Decision:** Store PDFs in private GCS and metadata/chunks/768d vectors in Cloud SQL PostgreSQL with pgvector.
- **Reason:** Cloud Run storage is disposable and retrieval must enforce user ownership.
- **Consequence:** Schema is migrated by `manage.py migrate`, not created by Streamlit startup.

### DEC-009: Use Gemini embeddings at 768 dimensions

- **When:** 2026-08-15
- **Status:** Active
- **Where:** `rag/llm.py`, `rag/ingest.py`, `rag/retriever.py`
- **Decision:** Use `gemini-embedding-2` with 768-dimensional output.
- **Reason:** Managed embedding service matches the Google Cloud deployment model.
- **Consequence:** Model or dimension changes require re-indexing.

### DEC-010: Use Gemini 3.5 Flash-Lite for generation

- **When:** 2026-08-15
- **Status:** Active
- **Where:** `rag/llm.py`, `rag/chat.py`
- **Decision:** Use `gemini-3.5-flash-lite` through `google-genai`, Vertex AI, ADC, and stable `v1`.
- **Reason:** Keeps generation on Google Cloud with the same auth path as embeddings.
- **Consequence:** Quota/model-access errors surface as safe user-facing failures.

### DEC-011: Require Google OIDC and approved-email allowlist

- **When:** 2026-08-15
- **Status:** Active
- **Where:** `app.py`, `.streamlit/secrets.toml`, `APPROVED_EMAILS`
- **Decision:** Use native `st.login()`, `st.user`, and `st.logout()`; deny emails not in `APPROVED_EMAILS`.
- **Reason:** This is a controlled app, not anonymous public access.
- **Consequence:** OIDC secrets are mounted from Secret Manager in production.

### DEC-012: Retain uploads for seven days

- **When:** 2026-08-15
- **Status:** Active
- **Where:** `rag/db.py`, `manage.py cleanup`, Cloud Run job, Cloud Storage bucket policy
- **Decision:** Uploaded PDFs, metadata, and embeddings expire after seven days and are permanently deleted.
- **Reason:** Minimize private document retention.
- **Consequence:** Bucket soft deletion must be disabled and cleanup must run daily.

### DEC-013: Store Indian statutes as section-aware shared documents

- **When:** 2026-08-15
- **Status:** Active
- **Where:** `rag/statutes.py`, `rag/db.py`, `manage.py`, `migrations/002_statute_metadata.sql`
- **Decision:** Parse BNS, BNSS, and BSA into chapter/section units, store flexible metadata in JSONB, and retrieve exact Act/Section references before semantic candidates.
- **Reason:** Legal sections are meaningful citation and retrieval boundaries; arbitrary character chunks can mix unrelated provisions.
- **Consequence:** Statute model or parser changes require re-ingestion, while existing Constitution and upload records remain compatible.

### DEC-014: Use PostgreSQL hybrid retrieval with RRF and Gemini reranking

- **When:** 2026-08-16
- **Status:** Active
- **Where:** `rag/retriever.py`, `rag/db.py`, `migrations/003_hybrid_search.sql`
- **Decision:** Combine deterministic references, original/rewrite pgvector searches, PostgreSQL FTS, RRF, and structured Gemini reranking.
- **Reason:** Legal relevance depends on exact authority and lexical wording as well as semantic similarity; PostgreSQL already provides the needed search features.
- **Consequence:** Natural questions add analysis and reranking calls, while exact questions skip analysis and every LLM failure has a deterministic retrieval fallback.

## Current Application Flow

### Startup

```text
streamlit run app.py
  -> user signs in with Google OIDC
  -> app rejects emails not in APPROVED_EMAILS
  -> app uses ready shared Constitution/statutes and owned unexpired uploads from PostgreSQL
```

### PDF ingestion

```text
User uploads PDF
  -> app.py reads bytes without saving locally
  -> rag/ingest.py checks %PDF-, size, encryption, readability, and page count
  -> private GCS object name is generated with UUID
  -> documents row is created as processing
  -> text is extracted with one-based page numbers
  -> Constitution article extraction runs only for document_type=constitution
  -> chunks are embedded with gemini-embedding-2
  -> chunks are stored in Cloud SQL pgvector
  -> document status becomes ready
```

### Statute ingestion

```text
manage.py ingest-statute PDF --act BNS|BNSS|BSA
  -> split front matter from enacted body
  -> read arrangement or Gazette margin titles
  -> build chapter/section units with page ranges
  -> split only inside oversized sections, preferring subsections
  -> embed and store JSONB metadata plus vector(768)
  -> publish new ready index, then remove the older index for that Act
```

### Question answering

```text
User asks a question
  -> chat history is read from st.session_state only
  -> explicit Act/Section/Article references are parsed deterministically
  -> natural questions receive one structured Gemini legal-query analysis
  -> original and distinct rewritten queries are embedded with gemini-embedding-2
  -> pgvector, PostgreSQL FTS, and exact metadata lookup retrieve scoped candidates
  -> every strategy enforces the same ownership, expiry, and selection predicate
  -> candidates merge by stable identity and rank through RRF
  -> Gemini reranks fused candidates; failure retains fused order
  -> exact chunks are preserved and final context is capped at 8 diverse chunks
  -> no Gemini answer call is made when context is empty
  -> the original question and authoritative text produce numbered citations and source details
```

### Cleanup

```text
Cloud Run job runs daily
  -> manage.py cleanup finds expired upload documents
  -> deletes GCS object, ignoring already-missing objects
  -> deletes document row
  -> chunks cascade through foreign keys
```

## New Decision Template

```markdown
### DEC-NNN: Short decision title

- **When:** YYYY-MM-DD
- **Status:** Proposed | Active | Superseded
- **Where:** Files, modules, services, or environments affected
- **Decision:** What was chosen
- **Reason:** Why it was chosen
- **Consequence:** Important benefits, costs, and follow-up work
```
