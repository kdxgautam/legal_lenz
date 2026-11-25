from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace

from rag import db
from rag.config import (
    EXACT_RRF_WEIGHT,
    FINAL_CONTEXT_K,
    FTS_K,
    FUSION_K,
    RERANK_CANDIDATES,
    RERANK_MIN_SCORE,
    RRF_CONSTANT,
    VECTOR_ORIGINAL_K,
    VECTOR_REWRITE_K,
)
from rag.llm import embed_texts, generate_json
from rag.old_laws import OldLawReference, extract_old_law_references, lookup, mentioned_old_acts
from rag.prompts import ANALYSIS_PROMPT, RERANK_PROMPT
from rag.statutes import STATUTES, StatuteReference


ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "search_query": {"type": "string"},
        "legal_domains": {"type": "array", "items": {"type": "string"}},
        "acts": {"type": "array", "items": {"type": "string"}},
        "statute_references": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"act": {"type": "string"}, "section": {"type": "string"}},
                "required": ["act", "section"],
                "additionalProperties": False,
            },
        },
        "articles": {"type": "array", "items": {"type": "string"}},
        "keywords": {"type": "array", "items": {"type": "string"}},
        "exact_reference_detected": {"type": "boolean"},
    },
    "required": [
        "search_query", "legal_domains", "acts", "statute_references", "articles",
        "keywords", "exact_reference_detected",
    ],
    "additionalProperties": False,
}

HISTORY_TURNS = 6
BARE_SECTION_RE = re.compile(
    r"\b(?:sections?|sec\.?|u/s\.?)\s*(\d+[a-z]?(?:\s*(?:,|and|&|or)\s*\d+[a-z]?)*)\b", re.I
)
UNKNOWN_ACT_RE = re.compile(r"\b(?:act|code|sanhita|adhiniyam)\b", re.I)

RERANK_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "string"},
                    "score": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["chunk_id", "score", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["results"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class LegalQueryAnalysis:
    original_query: str
    search_query: str
    legal_domains: list[str] = field(default_factory=list)
    acts: list[str] = field(default_factory=list)
    sections: list[str] = field(default_factory=list)
    articles: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    exact_reference_detected: bool = False
    exact_statute_references: list[StatuteReference] = field(default_factory=list)
    # References the model resolved from the conversation; boosted but still reranked.
    inferred_statute_references: list[StatuteReference] = field(default_factory=list)
    old_law_references: list[OldLawReference] = field(default_factory=list)

    @property
    def notes(self) -> list[str]:
        return [reference.note() for reference in self.old_law_references]


@dataclass
class RetrievalCandidate:
    chunk: db.Chunk
    original_rank: int | None = None
    rewrite_rank: int | None = None
    lexical_rank: int | None = None
    exact_rank: int | None = None
    inferred_rank: int | None = None
    fused_score: float = 0.0
    rerank_score: float | None = None
    rerank_reason: str = ""

    @property
    def exact_match(self) -> bool:
        return self.exact_rank is not None


@dataclass
class RetrievalTrace:
    analysis: LegalQueryAnalysis
    original_vector: list[db.Chunk] = field(default_factory=list)
    rewrite_vector: list[db.Chunk] = field(default_factory=list)
    full_text: list[db.Chunk] = field(default_factory=list)
    exact: list[db.Chunk] = field(default_factory=list)
    inferred: list[db.Chunk] = field(default_factory=list)
    fused: list[RetrievalCandidate] = field(default_factory=list)
    reranked: list[RetrievalCandidate] = field(default_factory=list)
    final: list[db.Chunk] = field(default_factory=list)
    rerank_fallback: bool = False


def extract_article_numbers(question: str) -> list[str]:
    return list(dict.fromkeys(
        f"Article {match.upper()}"
        for match in re.findall(r"\barticle\s+(\d+[a-z]?)\b", question, re.I)
    ))


def extract_article_number(question: str) -> str | None:
    articles = extract_article_numbers(question)
    return articles[0] if articles else None


def _act_pattern(short_name: str) -> str:
    full_name = re.escape(STATUTES[short_name].act_name.removesuffix(", 2023"))
    return rf"(?:\b{short_name}\b|\b{full_name}(?:,?\s*2023)?\b)"


def extract_statute_references(question: str) -> list[StatuteReference]:
    references = []
    for act in STATUTES:
        alias = _act_pattern(act)
        patterns = (
            rf"\bsection\s+(\d+[a-z]?)\s+(?:of\s+(?:the\s+)?)?{alias}",
            rf"{alias}\s+(?:section\s+)?(\d+[a-z]?)\b",
        )
        for pattern in patterns:
            for section in re.findall(pattern, question, re.I):
                reference = StatuteReference(act, section.upper())
                if reference not in references:
                    references.append(reference)
    return references


def extract_statute_reference(question: str) -> StatuteReference | None:
    references = extract_statute_references(question)
    return references[0] if references else None


def _strings(value) -> list[str]:
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))


def _normalize_acts(values: list[str]) -> list[str]:
    normalized = []
    for value in values:
        for act in STATUTES:
            if re.search(_act_pattern(act), value, re.I) and act not in normalized:
                normalized.append(act)
    return normalized


def _fts_query(analysis: LegalQueryAnalysis) -> str:
    terms = analysis.keywords or re.findall(r"[a-z0-9]+", analysis.search_query, re.I)
    terms = [term.replace('"', " ").strip() for term in terms if len(term.strip()) > 2]
    return " OR ".join(f'"{term}"' for term in dict.fromkeys(terms))


def _lexical_search(owner_email: str, selected_ids: list[str], analysis: LegalQueryAnalysis) -> list[db.Chunk]:
    return db.full_text_search(owner_email, selected_ids, _fts_query(analysis), FTS_K)


def _mentioned_acts(text: str) -> list[str]:
    return _normalize_acts([text]) + mentioned_old_acts(text)


def _resolve_section(act: str, section: str) -> tuple[list[StatuteReference], OldLawReference | None]:
    """Resolve a section of a current Act or a repealed Act to current statute references."""
    section = section.upper()
    number = int(re.match(r"\d+", section).group())
    if act in STATUTES:
        return ([StatuteReference(act, section)] if number <= STATUTES[act].final_section else []), None
    old = lookup(act, section)
    return (old.statute_references, old) if old else ([], None)


def _context_act(question: str, history: list[dict] | None) -> str | None:
    """The single Act a bare "section N" refers to: named in the question, else in the latest turn naming one."""
    acts = _mentioned_acts(question)
    if acts or UNKNOWN_ACT_RE.search(question):
        return acts[0] if len(acts) == 1 else None
    for item in reversed((history or [])[-HISTORY_TURNS:]):
        acts = _mentioned_acts(item.get("content", ""))
        if acts:
            return acts[0] if len(acts) == 1 else None
    return None


def _bare_section_numbers(question: str) -> list[str]:
    return list(dict.fromkeys(
        number.upper()
        for group in BARE_SECTION_RE.findall(question)
        for number in re.findall(r"\d+[a-z]?", group, re.I)
    ))


def explicit_references(
    question: str,
    history: list[dict] | None = None,
) -> tuple[list[StatuteReference], list[OldLawReference]]:
    references = extract_statute_references(question)
    old_references = extract_old_law_references(question)
    written = {ref.section_number for ref in references} | {ref.old_section for ref in old_references}
    bare = [number for number in _bare_section_numbers(question) if number not in written]
    act = _context_act(question, history) if bare else None
    for number in bare if act else []:
        resolved, old = _resolve_section(act, number)
        references.extend(resolved)
        if old:
            old_references.append(old)
    for old in old_references:
        references.extend(old.statute_references)
    return list(dict.fromkeys(references)), list(dict.fromkeys(old_references))


def _inferred_references(
    data: dict,
    context_text: str,
) -> tuple[list[StatuteReference], list[OldLawReference]]:
    references, old_references = [], []
    for item in data.get("statute_references") or []:
        if not isinstance(item, dict):
            continue
        section = re.sub(r"^section\s+", "", str(item.get("section", "")).strip(), flags=re.I)
        match = re.match(r"\d+[a-z]{0,2}", section, re.I)
        acts = _mentioned_acts(str(item.get("act", "")))
        # Only accept numbers that literally appear in the question or conversation.
        if not match or len(acts) != 1 or not re.search(rf"\b{re.escape(match.group())}\b", context_text, re.I):
            continue
        resolved, old = _resolve_section(acts[0], match.group())
        references.extend(resolved)
        if old:
            old_references.append(old)
    return list(dict.fromkeys(references)), list(dict.fromkeys(old_references))


def analyze_query(question: str, history: list[dict] | None = None) -> LegalQueryAnalysis:
    references, old_references = explicit_references(question, history)
    articles = extract_article_numbers(question)
    exact_acts = list(dict.fromkeys(reference.act_short_name for reference in references))
    exact_sections = list(dict.fromkeys(reference.section_number for reference in references))
    if references or articles:
        return LegalQueryAnalysis(
            original_query=question,
            search_query=question,
            acts=exact_acts,
            sections=exact_sections,
            articles=articles,
            exact_reference_detected=True,
            exact_statute_references=references,
            old_law_references=old_references,
        )

    history_text = "\n".join(
        f"{item.get('role', '')}: {item.get('content', '')}" for item in (history or [])[-HISTORY_TURNS:]
    )
    try:
        data = generate_json(
            ANALYSIS_PROMPT.format(history=history_text or "None", question=question),
            ANALYSIS_SCHEMA,
        )
        search_query = str(data.get("search_query", "")).strip() or question
        context_text = f"{question}\n{history_text}"
        inferred, inferred_old = _inferred_references(data, context_text)
        articles = [
            f"Article {number.upper()}"
            for value in _strings(data.get("articles"))
            if (number := re.sub(r"^article\s+", "", value, flags=re.I))
            and re.search(rf"\b{re.escape(number)}\b", context_text, re.I)
        ]
        return LegalQueryAnalysis(
            original_query=question,
            search_query=search_query,
            legal_domains=_strings(data.get("legal_domains")),
            acts=list(dict.fromkeys(
                _normalize_acts(_strings(data.get("acts"))) + [ref.act_short_name for ref in inferred]
            )),
            sections=list(dict.fromkeys(ref.section_number for ref in inferred)),
            articles=list(dict.fromkeys(articles)),
            keywords=_strings(data.get("keywords")),
            exact_reference_detected=False,
            exact_statute_references=[],
            inferred_statute_references=inferred,
            old_law_references=list(dict.fromkeys(old_references + inferred_old)),
        )
    except Exception:
        return LegalQueryAnalysis(original_query=question, search_query=question, old_law_references=old_references)


def _candidate_key(chunk: db.Chunk):
    metadata = chunk.metadata
    if metadata.get("section_number"):
        return chunk.document_id, metadata["section_number"], metadata.get("chunk_index")
    return chunk.id


def merge_candidates(results: dict[str, list[db.Chunk]]) -> list[RetrievalCandidate]:
    candidates = {}
    rank_fields = {
        "original": "original_rank",
        "rewrite": "rewrite_rank",
        "full_text": "lexical_rank",
        "exact": "exact_rank",
        "inferred": "inferred_rank",
    }
    for strategy, chunks in results.items():
        for rank, chunk in enumerate(chunks, 1):
            key = _candidate_key(chunk)
            candidate = candidates.setdefault(key, RetrievalCandidate(chunk=chunk))
            if strategy in ("exact", "inferred"):
                candidate.chunk = chunk
            setattr(candidate, rank_fields[strategy], rank)

    for candidate in candidates.values():
        for rank in (candidate.original_rank, candidate.rewrite_rank, candidate.lexical_rank):
            if rank is not None:
                candidate.fused_score += 1 / (RRF_CONSTANT + rank)
        for rank in (candidate.exact_rank, candidate.inferred_rank):
            if rank is not None:
                candidate.fused_score += EXACT_RRF_WEIGHT / (RRF_CONSTANT + rank)
    return sorted(candidates.values(), key=lambda item: (-item.fused_score, item.chunk.id))[:FUSION_K]


def rerank_candidates(
    question: str,
    analysis: LegalQueryAnalysis,
    candidates: list[RetrievalCandidate],
) -> tuple[list[RetrievalCandidate], bool]:
    if not candidates:
        return [], False
    if RERANK_CANDIDATES <= 0:
        return candidates, True
    initial = list(candidates[:RERANK_CANDIDATES])
    required = []
    for act in analysis.acts:
        best = next((item for item in candidates if item.chunk.metadata.get("act_short_name") == act), None)
        if best and best not in initial and best not in required:
            required.append(best)
    pool = initial[:RERANK_CANDIDATES - len(required)] + required
    payload = [
        {
            "chunk_id": item.chunk.id,
            "source_type": item.chunk.metadata.get("document_type"),
            "document": item.chunk.document_name,
            "act": item.chunk.metadata.get("act_short_name"),
            "section": item.chunk.metadata.get("section_number"),
            "section_title": item.chunk.metadata.get("section_title"),
            "chapter": item.chunk.metadata.get("chapter_title"),
            "article": item.chunk.article,
            "page": item.chunk.page,
            "text": item.chunk.content,
        }
        for item in pool
    ]
    try:
        data = generate_json(
            RERANK_PROMPT.format(
                question=question,
                analysis=json.dumps({
                    "search_query": analysis.search_query,
                    "domains": analysis.legal_domains,
                    "acts": analysis.acts,
                    "sections": analysis.sections,
                    "articles": analysis.articles,
                    "keywords": analysis.keywords,
                }),
                candidates=json.dumps(payload),
            ),
            RERANK_SCHEMA,
        )
        if not isinstance(data.get("results"), list):
            raise ValueError("Reranker returned no results list.")
        by_id = {candidate.chunk.id: candidate for candidate in pool}
        ranked = []
        for item in data["results"]:
            candidate = by_id.get(str(item.get("chunk_id", "")))
            score = float(item.get("score", -1))
            if candidate and 0 <= score <= 1 and score >= RERANK_MIN_SCORE and candidate not in ranked:
                candidate.rerank_score = score
                candidate.rerank_reason = str(item.get("reason", ""))
                ranked.append(candidate)
        ranked.sort(key=lambda item: (-(item.rerank_score or 0), -item.fused_score))
        return ranked, False
    except Exception as exc:
        print(f"Reranking failed; using fused ranking: {exc}")
        return candidates, True


def _legal_unit(candidate: RetrievalCandidate):
    chunk = candidate.chunk
    metadata = chunk.metadata
    return chunk.document_id, metadata.get("section_number") or chunk.article or chunk.id


def select_context(
    ranked: list[RetrievalCandidate],
    exact: list[RetrievalCandidate],
) -> list[db.Chunk]:
    selected = []
    seen = set()
    counts = {}

    def add(candidate: RetrievalCandidate, enforce_diversity: bool) -> bool:
        key = _candidate_key(candidate.chunk)
        unit = _legal_unit(candidate)
        if key in seen or (enforce_diversity and counts.get(unit, 0) >= 2):
            return False
        seen.add(key)
        counts[unit] = counts.get(unit, 0) + 1
        selected.append(candidate.chunk)
        return True

    for candidate in sorted(exact, key=lambda item: item.exact_rank or 0):
        add(candidate, False)
        if len(selected) == FINAL_CONTEXT_K:
            break
    for candidate in ranked:
        if len(selected) == FINAL_CONTEXT_K:
            break
        add(candidate, True)
    for candidate in ranked:
        if len(selected) == FINAL_CONTEXT_K:
            break
        add(candidate, False)
    return [replace(chunk, citation=index) for index, chunk in enumerate(selected, 1)]


def retrieve_with_trace(
    owner_email: str,
    selected_ids: list[str],
    question: str,
    history: list[dict] | None = None,
) -> RetrievalTrace:
    analysis = analyze_query(question, history)
    exact = db.exact_search(
        owner_email,
        selected_ids,
        [(ref.act_short_name, ref.section_number) for ref in analysis.exact_statute_references],
        extract_article_numbers(question),
        FINAL_CONTEXT_K,
        query=_fts_query(analysis),
    )
    inferred = []
    if not analysis.exact_reference_detected and (analysis.inferred_statute_references or analysis.articles):
        inferred = db.exact_search(
            owner_email,
            selected_ids,
            [(ref.act_short_name, ref.section_number) for ref in analysis.inferred_statute_references],
            analysis.articles,
            FINAL_CONTEXT_K,
            query=_fts_query(analysis),
        )
    if analysis.exact_reference_detected and exact:
        fused = merge_candidates({"original": [], "rewrite": [], "full_text": [], "exact": exact})
        exact_candidates = [candidate for candidate in fused if candidate.exact_match]
        final = select_context(fused, exact_candidates)
        return RetrievalTrace(
            analysis=analysis,
            exact=exact,
            fused=fused,
            reranked=fused,
            final=final,
        )

    queries = list(dict.fromkeys(
        query.strip() for query in (question, analysis.search_query) if query.strip()
    ))
    embeddings = embed_texts(queries, "RETRIEVAL_QUERY")
    original = db.vector_search(owner_email, selected_ids, embeddings[0], VECTOR_ORIGINAL_K)
    rewrite = []
    if len(queries) > 1:
        rewrite = db.vector_search(owner_email, selected_ids, embeddings[1], VECTOR_REWRITE_K)
    full_text = _lexical_search(owner_email, selected_ids, analysis)
    fused = merge_candidates({
        "original": original,
        "rewrite": rewrite,
        "full_text": full_text,
        "exact": exact,
        "inferred": inferred,
    })
    exact_candidates = [candidate for candidate in fused if candidate.exact_match]
    if analysis.exact_reference_detected:
        reranked, fallback = fused, False
    else:
        reranked, fallback = rerank_candidates(question, analysis, fused)
    final = select_context(reranked, exact_candidates) if reranked or exact_candidates else []
    return RetrievalTrace(
        analysis=analysis,
        original_vector=original,
        rewrite_vector=rewrite,
        full_text=full_text,
        exact=exact,
        inferred=inferred,
        fused=fused,
        reranked=reranked,
        final=final,
        rerank_fallback=fallback,
    )

