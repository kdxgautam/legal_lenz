from rag.ingest import ingest_pdf


ingest_pdf(
    pdf_path="data/pdfs/constitution.pdf",
    persist_directory="data/chroma/base",
)