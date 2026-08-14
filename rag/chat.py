from rag.llm import generate_text
from rag.prompts import ANSWER_PROMPT
from rag.retriever import retrieve_chunks

NO_CONTEXT_ANSWER = "I could not find sufficient information in the selected documents."


def build_context(chunks) -> str:
    return "\n\n".join(
        (
            f"[{chunk.citation}] {chunk.document_name}"
            f" | {_page_label(chunk.page, chunk.metadata.get('page_end'))}"
            f" | {_legal_label(chunk)}\n"
            f"{chunk.content}"
        )
        for chunk in chunks
    )


def _page_label(page: int, page_end: int | None) -> str:
    return f"pages {page}-{page_end}" if page_end and page_end != page else f"page {page}"


def _legal_label(chunk) -> str:
    metadata = chunk.metadata
    if metadata.get("section_number"):
        chapter = f" | Chapter {metadata['chapter_number']}" if metadata.get("chapter_number") else ""
        return f"Section {metadata['section_number']}{chapter}"
    return chunk.article or "No article"


def ask_question(owner_email: str, selected_document_ids: list[str], question: str, history: list[dict]) -> dict:
    chunks = retrieve_chunks(owner_email, selected_document_ids, question, history)
    if not chunks:
        return {"answer": NO_CONTEXT_ANSWER, "sources": []}
    text_history = "\n".join(f"{item['role']}: {item['content']}" for item in history[-6:])
    answer = generate_text(ANSWER_PROMPT.format(
        context=build_context(chunks),
        history=text_history or "None",
        question=question,
    ))
    return {
        "answer": answer or NO_CONTEXT_ANSWER,
        "sources": [
            {
                "citation": chunk.citation,
                "document_name": chunk.document_name,
                "page": chunk.page,
                "page_end": chunk.metadata.get("page_end"),
                "article": chunk.article,
                "metadata": chunk.metadata,
                "text": chunk.content,
            }
            for chunk in chunks
        ],
    }
