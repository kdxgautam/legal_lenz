import pytest

from rag import chat, config
from rag.db import Chunk
from rag.ingest import IngestError, extract_articles, safe_filename, validate_pdf
from rag.retriever import (
    LegalQueryAnalysis,
    RetrievalCandidate,
    analyze_query,
    extract_article_number,
    extract_statute_references,
    merge_candidates,
    rerank_candidates,
    retrieve_with_trace,
)


def chunk(chunk_id="c1", act=None, section=None, article=None, content="text"):
    metadata = {"document_type": "statute" if act else "constitution"}
    if act:
        metadata |= {"act_short_name": act, "section_number": section, "chunk_index": 1}
    return Chunk(chunk_id, f"d-{act or 'constitution'}", "Document", 1, article, content, metadata=metadata)


def test_article_21a_is_parsed():
    articles = extract_articles("21A. Right to education\nText here\n22. Protection\nMore text")
    assert articles[0]["article"] == "Article 21A"
    assert extract_article_number("Explain article 21A") == "Article 21A"
    footnoted = extract_articles("21. Life\nText\n2[21A. Right to education\nEducation text")
    assert [item["article"] for item in footnoted] == ["Article 21", "Article 21A"]


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Section 303 BNS", [("BNS", "303")]),
        ("BNS 303", [("BNS", "303")]),
        ("Section 303 of Bharatiya Nyaya Sanhita", [("BNS", "303")]),
        ("BNS Section 303 and BNSS Section 35", [("BNS", "303"), ("BNSS", "35")]),
    ],
)
def test_statute_references_are_act_specific(question, expected):
    references = extract_statute_references(question)
    assert [(ref.act_short_name, ref.section_number) for ref in references] == expected


def test_exact_query_bypasses_gemini(monkeypatch):
    monkeypatch.setattr("rag.retriever.generate_json", lambda *_: pytest.fail("analysis should be skipped"))
    analysis = analyze_query("Explain BNS Section 303")
    assert analysis.exact_reference_detected
    assert analysis.acts == ["BNS"]
    assert analysis.sections == ["303"]


def test_structured_analysis_and_failure_fallback(monkeypatch):
    monkeypatch.setattr("rag.retriever.generate_json", lambda *_: {
        "search_query": "arrest without warrant theft",
        "legal_domains": ["criminal procedure"],
        "acts": ["BNS", "Bharatiya Nagarik Suraksha Sanhita"],
        "sections": [],
        "articles": [],
        "keywords": ["arrest", "theft"],
        "exact_reference_detected": False,
    })
    analysis = analyze_query("Can police arrest me for theft?")
    assert analysis.search_query == "arrest without warrant theft"
    assert analysis.acts == ["BNS", "BNSS"]
    monkeypatch.setattr("rag.retriever.generate_json", lambda *_: (_ for _ in ()).throw(RuntimeError()))
    assert analyze_query("question").search_query == "question"


def test_merge_deduplicates_and_exact_wins():
    same = chunk(act="BNS", section="303")
    other = chunk("c2", "BNSS", "303")
    merged = merge_candidates({
        "original": [other, same],
        "rewrite": [same],
        "full_text": [same],
        "exact": [same],
    })
    assert len(merged) == 2
    assert merged[0].chunk.id == same.id
    assert merged[0].exact_match
    assert merged[1].chunk.metadata["act_short_name"] == "BNSS"


def test_exact_metadata_replaces_vector_metadata_for_same_chunk():
    vector = chunk(article="Article 21")
    exact = chunk(article="Article 21A")
    merged = merge_candidates({"original": [vector], "rewrite": [], "full_text": [], "exact": [exact]})
    assert merged[0].chunk.article == "Article 21A"


def test_reranker_failure_uses_fusion(monkeypatch):
    candidates = [RetrievalCandidate(chunk())]
    analysis = LegalQueryAnalysis("q", "q")
    monkeypatch.setattr("rag.retriever.generate_json", lambda *_: (_ for _ in ()).throw(RuntimeError()))
    ranked, fallback = rerank_candidates("q", analysis, candidates)
    assert ranked == candidates
    assert fallback


def test_original_and_rewrite_vectors_execute(monkeypatch):
    original = chunk("original")
    rewritten = chunk("rewrite")
    monkeypatch.setattr("rag.retriever.analyze_query", lambda *_: LegalQueryAnalysis("q", "rewrite"))
    monkeypatch.setattr("rag.retriever.embed_texts", lambda texts, _: [[0.0], [1.0]])
    calls = []
    monkeypatch.setattr(
        "rag.retriever.db.vector_search",
        lambda *args: calls.append(args[2]) or ([original] if args[2] == [0.0] else [rewritten]),
    )
    monkeypatch.setattr("rag.retriever.db.full_text_search", lambda *args: [])
    monkeypatch.setattr("rag.retriever.db.exact_search", lambda *args: [])
    monkeypatch.setattr("rag.retriever.rerank_candidates", lambda q, a, c: (c, False))
    trace = retrieve_with_trace("a@example.com", [], "q")
    assert calls == [[0.0], [1.0]]
    assert {item.id for item in trace.final} == {"original", "rewrite"}


def test_exact_retrieval_skips_slow_paths(monkeypatch):
    exact = chunk("exact", "BNS", "303")
    monkeypatch.setattr("rag.retriever.db.exact_search", lambda *args: [exact])
    monkeypatch.setattr("rag.retriever.embed_texts", lambda *_: pytest.fail("exact query should not embed"))
    monkeypatch.setattr("rag.retriever.db.vector_search", lambda *_: pytest.fail("exact query should not vector search"))
    monkeypatch.setattr("rag.retriever.db.full_text_search", lambda *_: pytest.fail("exact query should not FTS"))
    monkeypatch.setattr("rag.retriever.rerank_candidates", lambda *_: pytest.fail("exact query should not rerank"))
    trace = retrieve_with_trace("a@example.com", [], "What does Section 303 of BNS say?")
    assert [item.id for item in trace.final] == ["exact"]


def test_lexical_search_runs_once(monkeypatch):
    monkeypatch.setattr("rag.retriever.analyze_query", lambda *_: LegalQueryAnalysis(
        "q", "arrest theft", keywords=["arrest", "theft"]
    ))
    monkeypatch.setattr("rag.retriever.embed_texts", lambda texts, _: [[0.0] for _ in texts])
    monkeypatch.setattr("rag.retriever.db.vector_search", lambda *args: [])
    monkeypatch.setattr("rag.retriever.db.exact_search", lambda *args: [])
    monkeypatch.setattr("rag.retriever.rerank_candidates", lambda q, a, c: (c, False))
    calls = []
    monkeypatch.setattr("rag.retriever.db.full_text_search", lambda *args: calls.append(args[2]) or [])
    retrieve_with_trace("a@example.com", [], "q")
    assert calls == ['"arrest" OR "theft"']


def test_generic_filename_is_sanitized():
    assert safe_filename("../../secret.pdf") == "secret.pdf"
    assert safe_filename("") == "document.pdf"


def test_rejects_non_pdf_signature():
    with pytest.raises(IngestError):
        validate_pdf(b"not a pdf")


def test_constitution_validator_can_skip_page_cap(monkeypatch):
    class FakeDoc:
        is_encrypted = False
        page_count = 301

    monkeypatch.setattr("fitz.open", lambda **_: FakeDoc())
    assert validate_pdf(b"%PDF- fake", max_pages=None).page_count == 301


def test_context_uses_numbered_sources():
    context = chat.build_context([Chunk("c1", "d1", "Constitution", 42, "Article 21A", "education", 1)])
    assert "[1] Constitution | page 42 | Article 21A" in context


def test_no_context_skips_answer_model(monkeypatch):
    monkeypatch.setattr(chat, "retrieve_chunks", lambda *args: [])
    monkeypatch.setattr(chat, "generate_text", lambda *_: pytest.fail("answer model should not run"))
    result = chat.ask_question("a@example.com", [], "hello", [])
    assert result == {"answer": chat.NO_CONTEXT_ANSWER, "sources": []}


def test_answer_receives_original_question(monkeypatch):
    monkeypatch.setattr(chat, "retrieve_chunks", lambda *args: [chunk(article="Article 21")])
    prompts = []
    monkeypatch.setattr(chat, "generate_text", lambda prompt: prompts.append(prompt) or "answer [1]")
    chat.ask_question("a@example.com", [], "What did I actually ask?", [])
    assert "Original user question:\nWhat did I actually ask?" in prompts[0]


def test_allowlist_is_lowercase(monkeypatch):
    monkeypatch.setattr(config, "APPROVED_EMAILS", {"user@example.com"})
    assert "user@example.com" in config.APPROVED_EMAILS
