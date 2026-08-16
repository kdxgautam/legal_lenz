import argparse
import uuid
from pathlib import Path

from sqlalchemy import text

from rag import db
from rag.ingest import chunk_units, page_units, validate_pdf
from rag.llm import embed_texts
from rag.retriever import retrieve_with_trace
from rag.storage import delete_object, upload_pdf
from rag.statutes import STATUTES, chunk_statute_units, extract_statute_structure


def migrate() -> None:
    with db.tx() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                name text PRIMARY KEY,
                applied_at timestamptz NOT NULL DEFAULT now()
            )
        """))
        applied = {row[0] for row in conn.execute(text("SELECT name FROM schema_migrations"))}
        for migration in sorted(Path("migrations").glob("*.sql")):
            if migration.name in applied:
                continue
            for statement in [part.strip() for part in migration.read_text().split(";") if part.strip()]:
                conn.execute(text(statement))
            conn.execute(text("INSERT INTO schema_migrations (name) VALUES (:name)"), {"name": migration.name})
            print(f"Applied {migration.name}")


def ingest_constitution(path: str, full: bool = False) -> None:
    data = Path(path).read_bytes()
    doc = validate_pdf(data, max_pages=None)
    object_name = "constitution/full.pdf" if full else "constitution/fundamental-rights.pdf"
    upload_pdf(data, object_name)
    with db.tx() as conn:
        conn.execute(text("DELETE FROM documents WHERE type = 'constitution'"))
    title = "Constitution of India" if full else "Constitution of India - Fundamental Rights"
    document_id = db.new_document(None, title, object_name, "constitution")
    units = page_units(doc, "constitution")
    if not full:
        units = [unit for unit in units if unit["page"] <= 60 and unit["article"]]
    chunks = chunk_units(units)
    print(f"Embedding {len(chunks)} chunks...")
    embeddings = embed_texts([chunk["content"] for chunk in chunks], "RETRIEVAL_DOCUMENT")
    db.insert_chunks(document_id, [{**chunk, "embedding": embedding} for chunk, embedding in zip(chunks, embeddings)])
    db.set_document_status(document_id, "ready")
    doc.close()


def ingest_statute(path: str, act: str) -> None:
    definition = STATUTES[act]
    data = Path(path).read_bytes()
    doc = validate_pdf(data, max_pages=None)
    document_id = None
    uploaded = False
    object_name = f"statutes/{act.lower()}/{uuid.uuid4()}.pdf"
    try:
        units, report = extract_statute_structure(doc, definition)
        chunks = chunk_statute_units(units)
        for chunk in chunks:
            chunk["metadata"] = {**definition.metadata(), **chunk["metadata"]}
        report.chunks_created = len(chunks)
        if not chunks:
            raise ValueError("No statute sections were detected.")
        for warning in report.warnings:
            print(f"WARNING: {warning}")

        upload_pdf(data, object_name)
        uploaded = True
        document_id = db.new_document(
            None,
            f"{definition.act_name}.pdf",
            object_name,
            "statute",
            definition.metadata(),
        )
        print(f"Embedding {len(chunks)} chunks...")
        embeddings = embed_texts([chunk["content"] for chunk in chunks], "RETRIEVAL_DOCUMENT")
        report.embeddings_created = len(embeddings)
        db.insert_chunks(
            document_id,
            [{**chunk, "embedding": embedding} for chunk, embedding in zip(chunks, embeddings)],
        )
        db.set_document_status(document_id, "ready")

        for old in db.previous_statute_documents(act, document_id):
            db.delete_document_row(old["id"])
            try:
                delete_object(old["gcs_object"])
            except Exception as exc:
                print(f"WARNING: Old GCS object could not be deleted: {exc}")

        print(f"Act: {definition.act_name}")
        print(f"Pages processed: {report.pages_processed}")
        print(f"Chapters detected: {report.chapters_detected}")
        print(f"Sections detected: {report.sections_detected}")
        print(f"Chunks created: {report.chunks_created}")
        print(f"Embeddings created: {report.embeddings_created}")
        print(f"Failed sections: {report.failed_sections}")
    except Exception:
        if document_id:
            db.set_document_status(document_id, "failed")
        if uploaded:
            delete_object(object_name)
        raise
    finally:
        doc.close()


def cleanup(limit: int) -> None:
    for doc in db.expired_documents(limit):
        try:
            delete_object(doc["gcs_object"])
        except Exception as exc:
            print(f"WARNING: GCS object could not be deleted: {exc}")
        db.delete_document_row(doc["id"])
    deleted_chats = db.delete_expired_chats(limit)
    if deleted_chats:
        print(f"Deleted {deleted_chats} expired chats")


def debug_retrieval(question: str, email: str, document_ids: list[str]) -> None:
    trace = retrieve_with_trace(email, document_ids, question)
    analysis = trace.analysis
    print("QUERY ANALYSIS")
    print(f"Original: {analysis.original_query}")
    print(f"Rewrite: {analysis.search_query}")
    print(f"Domains: {', '.join(analysis.legal_domains) or 'none'}")
    print(f"Acts: {', '.join(analysis.acts) or 'none'}")
    print(f"Sections: {', '.join(analysis.sections) or 'none'}")
    print(f"Articles: {', '.join(analysis.articles) or 'none'}")
    print(f"Keywords: {', '.join(analysis.keywords) or 'none'}")
    print()

    def print_chunks(label, chunks):
        print(label)
        if not chunks:
            print("none")
        for index, chunk in enumerate(chunks, 1):
            legal_unit = chunk.metadata.get("section_number") or chunk.article or "-"
            act = chunk.metadata.get("act_short_name") or chunk.metadata.get("document_type") or "-"
            print(f"{index}. {chunk.id} | {act} | {legal_unit} | page {chunk.page}")
        print()

    print_chunks("ORIGINAL VECTOR", trace.original_vector)
    print_chunks("REWRITE VECTOR", trace.rewrite_vector)
    print_chunks("FULL TEXT", trace.full_text)
    print_chunks("EXACT MATCH", trace.exact)
    print(f"MERGED CANDIDATES: {len(trace.fused)}")
    print("RERANKED" + (" (fusion fallback)" if trace.rerank_fallback else ""))
    for index, candidate in enumerate(trace.reranked, 1):
        chunk = candidate.chunk
        unit = chunk.metadata.get("section_number") or chunk.article or "-"
        score = candidate.rerank_score if candidate.rerank_score is not None else candidate.fused_score
        print(f"{index}. {chunk.document_name} | {unit} | score {score:.4f}")
    print()
    print_chunks("FINAL CONTEXT", trace.final)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("migrate")
    constitution = sub.add_parser("ingest-constitution")
    constitution.add_argument("path", default="data/pdfs/constitution.pdf", nargs="?")
    constitution_full = sub.add_parser("ingest-constitution-full")
    constitution_full.add_argument("path", default="data/pdfs/constitution.pdf", nargs="?")
    statute = sub.add_parser("ingest-statute")
    statute.add_argument("path")
    statute.add_argument("--act", choices=sorted(STATUTES), required=True)
    clean = sub.add_parser("cleanup")
    clean.add_argument("--limit", type=int, default=100)
    debug = sub.add_parser("debug-retrieval")
    debug.add_argument("question")
    debug.add_argument("--email", required=True)
    debug.add_argument("--document-id", action="append", default=[])
    args = parser.parse_args()

    if args.command == "migrate":
        migrate()
    elif args.command == "ingest-constitution":
        ingest_constitution(args.path)
    elif args.command == "ingest-constitution-full":
        ingest_constitution(args.path, full=True)
    elif args.command == "ingest-statute":
        ingest_statute(args.path, args.act)
    elif args.command == "cleanup":
        cleanup(args.limit)
    elif args.command == "debug-retrieval":
        debug_retrieval(args.question, args.email, args.document_id)


if __name__ == "__main__":
    main()
