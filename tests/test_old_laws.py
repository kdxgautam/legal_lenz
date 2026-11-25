import pytest

from rag import chat
from rag.db import Chunk
from rag.old_laws import extract_old_law_references, lookup
from rag.retriever import LegalQueryAnalysis, RetrievalTrace, analyze_query, retrieve_with_trace
from rag.statutes import StatuteReference


def refs(analysis):
    return [(ref.act_short_name, ref.section_number) for ref in analysis.exact_statute_references]


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What is IPC 420?", [("IPC", "420", ("318(4)",))]),
        ("punishment under 302 IPC", [("IPC", "302", ("103(1)",))]),
        ("Section 498A I.P.C.", [("IPC", "498A", ("85", "86"))]),
        ("Section 438 of the CrPC", [("CrPC", "438", ("482",))]),
        ("u/s 154 Cr.P.C.", [("CrPC", "154", ("173",))]),
        ("Is 65B of the Evidence Act still valid?", [("IEA", "65B", ("63",))]),
        ("IPC 1860 overview", []),
        ("What is BNS 303?", []),
    ],
)
def test_old_law_references_map_to_new_sections(question, expected):
    found = extract_old_law_references(question)
    assert [(ref.old_act, ref.old_section, ref.new_sections) for ref in found] == expected


def test_repealed_section_without_equivalent_gets_note():
    reference = lookup("IPC", "124A")
    assert reference.new_sections == ()
    assert "no corresponding BNS section" in reference.note()


def test_old_law_question_uses_exact_new_section(monkeypatch):
    monkeypatch.setattr("rag.retriever.generate_json", lambda *_: pytest.fail("analysis should be skipped"))
    analysis = analyze_query("What is the punishment under IPC 420?")
    assert analysis.exact_reference_detected
    assert refs(analysis) == [("BNS", "318")]
    assert analysis.notes == ["IPC section 420 corresponds to BNS section 318(4) (BPR&D correspondence table)."]


def test_bare_section_follow_up_uses_act_from_history(monkeypatch):
    monkeypatch.setattr("rag.retriever.generate_json", lambda *_: pytest.fail("analysis should be skipped"))
    history = [
        {"role": "user", "content": "What does BNS Section 303 say?"},
        {"role": "assistant", "content": "It defines theft [1]."},
    ]
    assert refs(analyze_query("And section 304?", history)) == [("BNS", "304")]
    assert refs(analyze_query("BNS 303 and section 304")) == [("BNS", "303"), ("BNS", "304")]


def test_bare_section_follow_up_maps_old_act_from_history(monkeypatch):
    monkeypatch.setattr("rag.retriever.generate_json", lambda *_: pytest.fail("analysis should be skipped"))
    analysis = analyze_query("what about section 406?", [{"role": "user", "content": "Explain IPC 420"}])
    expected = [(ref.act_short_name, ref.section_number) for ref in lookup("IPC", "406").statute_references]
    assert refs(analysis) == expected


def test_bare_section_of_unknown_act_ignores_history(monkeypatch):
    monkeypatch.setattr("rag.retriever.generate_json", lambda *_: {"search_query": "contract act section 10"})
    history = [{"role": "user", "content": "What does BNS Section 303 say?"}]
    analysis = analyze_query("What is section 10 of the Contract Act?", history)
    assert not analysis.exact_reference_detected
    assert analysis.inferred_statute_references == []


def test_model_references_must_appear_in_conversation(monkeypatch):
    monkeypatch.setattr("rag.retriever.generate_json", lambda *_: {
        "search_query": "theft punishment",
        "acts": ["BNS"],
        "statute_references": [
            {"act": "BNS", "section": "303"},
            {"act": "BNS", "section": "305"},
            {"act": "CrPC", "section": "41"},
        ],
        "articles": ["Article 21", "Article 22"],
    })
    history = [{"role": "user", "content": "What does BNS 303 say? Also CrPC 41 and Article 21."}]
    analysis = analyze_query("What is the punishment under that section?", history)
    assert analysis.inferred_statute_references == [StatuteReference("BNS", "303"), StatuteReference("BNSS", "35")]
    assert analysis.articles == ["Article 21"]
    assert not analysis.exact_reference_detected


def test_inferred_references_are_boosted_but_reranked(monkeypatch):
    inferred = Chunk("c303", "d-bns", "BNS", 1, None, "303. Theft.", metadata={
        "act_short_name": "BNS", "section_number": "303", "chunk_index": 1,
    })
    analysis = LegalQueryAnalysis("q", "q", inferred_statute_references=[StatuteReference("BNS", "303")])
    monkeypatch.setattr("rag.retriever.analyze_query", lambda *_: analysis)
    monkeypatch.setattr("rag.retriever.embed_texts", lambda texts, _: [[0.0] for _ in texts])
    monkeypatch.setattr("rag.retriever.db.vector_search", lambda *args: [])
    monkeypatch.setattr("rag.retriever.db.full_text_search", lambda *args: [])
    monkeypatch.setattr("rag.retriever.db.exact_search", lambda owner, ids, statutes, articles, limit, **_: (
        [inferred] if statutes else []
    ))
    pools = []
    monkeypatch.setattr("rag.retriever.rerank_candidates", lambda q, a, c: pools.append(c) or ([], False))
    trace = retrieve_with_trace("a@example.com", [], "q")
    assert [candidate.chunk.id for candidate in pools[0]] == ["c303"]
    assert not pools[0][0].exact_match
    assert trace.final == []


def test_no_context_answer_includes_old_law_notes(monkeypatch):
    analysis = LegalQueryAnalysis("q", "q", old_law_references=[lookup("IPC", "124A")])
    monkeypatch.setattr(chat, "retrieve_with_trace", lambda *args: RetrievalTrace(analysis=analysis))
    result = chat.ask_question("a@example.com", [], "What is IPC 124A?", [])
    assert result["answer"].startswith(chat.NO_CONTEXT_ANSWER)
    assert "IPC section 124A has no corresponding BNS section" in result["answer"]
