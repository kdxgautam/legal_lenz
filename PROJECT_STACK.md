# Legal Lenz Stack

Keep this file updated whenever a model, managed service, storage resource, vector setting, or runtime dependency changes.

| Area | Current choice | Version / setting | Why we use it |
| --- | --- | --- | --- |
| App runtime | Streamlit | `streamlit[auth]` from `pyproject.toml` | Fast Python UI with native chat, upload, sidebar controls, and OIDC login. |
| Auth | Streamlit OIDC with Google | `.streamlit/secrets.toml`, `APPROVED_EMAILS` | Google proves identity; allowlist keeps the app controlled for approved users only. |
| Generation model | Gemini Flash-Lite | `gemini-3.5-flash-lite` | Low-cost Google-hosted answer generation using ADC/Vertex AI. |
| Embedding model | Gemini Embedding | `gemini-embedding-2` | Managed embeddings on Google Cloud; avoids local Torch/SentenceTransformer downloads. |
| Embedding size | pgvector vectors | `vector(768)` | 768 dimensions keeps storage/cost lower while matching Gemini embedding support. |
| Vector database | PostgreSQL + pgvector | `CREATE EXTENSION vector` | One database handles metadata, ownership, retention, and vector search. |
| Lexical retrieval | PostgreSQL full-text search | stored `tsvector`, GIN, `websearch_to_tsquery` | Adds phrase/keyword retrieval without another search service. |
| Candidate fusion | Reciprocal Rank Fusion | `RRF_CONSTANT=60`, exact weight `3` | Combines incomparable vector and lexical ranks while keeping exact references strong. |
| Query analysis / reranking | Gemini Flash-Lite structured JSON | one analysis and one rerank call for natural questions | Enriches legal queries and removes legally irrelevant semantic matches with deterministic fallbacks. |
| Database | Neon Postgres for local/student setup | `DATABASE_URL` | Lower-cost Postgres with pgvector support; easier on student credits than always-on Cloud SQL. |
| Production database option | Cloud SQL PostgreSQL | `INSTANCE_CONNECTION_NAME`, `DB_USER`, `DB_NAME` | Google-managed production option for Cloud Run with IAM/service-account access. |
| PDF storage | Private Cloud Storage bucket | `UPLOAD_BUCKET` | Cloud Run disk is disposable; PDFs need private managed object storage. |
| Upload retention | Hard delete | 7 days | Minimize private document retention and storage cost. |
| Chat history | PostgreSQL tables | 5 active chats/user, 30-day inactivity expiry | Keeps user conversations available across logins without storing private source excerpts. |
| Chat privacy | Persist metadata only | no saved source excerpts | Keeps citation context while preserving the seven-day private upload deletion guarantee. |
| PDF parser | PyMuPDF | `pymupdf` | Reliable PDF text extraction and page counting with one dependency. |
| Text splitter | LangChain text splitter only | `langchain-text-splitters` | Keeps chunking utility without broad LangChain provider packages. |
| Statute parser | PyMuPDF blocks + standard-library state machine | BNS 358, BNSS 531, BSA 170 sections | Preserves Act, chapter, section, subsection, and page ranges without OCR or another parsing dependency. |
| Local credentials | Google ADC JSON | `application_default_credentials.json` | Lets local Gemini/GCS calls authenticate; ignored by Git. |
| Production credentials | Cloud Run service account | ADC from runtime | No credential file in container; Google-managed identity is safer. |
| Container | Docker | `python:3.12-slim` | Small boring Python base image for Cloud Run. |
| Dependency manager | uv | `pyproject.toml` + `uv.lock` | One dependency source with reproducible installs. |

## Superseded Choices

| Old choice | Replaced by | Reason |
| --- | --- | --- |
| Groq `llama-3.3-70b-versatile` | Gemini `gemini-3.5-flash-lite` | Standardize on Google ADC/Vertex AI and remove extra provider secrets. |
| Chroma local vector DB | PostgreSQL + pgvector | Cloud Run storage is disposable; SQL enforces ownership and retention. |
| MiniLM local embeddings | `gemini-embedding-2` | Avoid local model downloads, Torch, and deployment weight. |
| Pinecone dependency | pgvector | Dedicated vector DB is unnecessary until pgvector is too slow. |
| `requirements.txt` | `pyproject.toml` + `uv.lock` | Avoid dependency drift. |
