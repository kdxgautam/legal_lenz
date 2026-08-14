REWRITE_PROMPT = """Rewrite the latest question as a standalone legal question.
Return only the rewritten question.

History:
{history}

Question:
{question}
"""

ANALYSIS_PROMPT = """Analyze this Indian-law question for document retrieval.
Return concise structured fields. The search query is a retrieval aid, not an answer.
Suggest every Act reasonably needed to cover both an underlying offence or right and
the related procedure; do not return only the primary Act. Sections and Articles must contain only
references explicitly written by the user; never infer or invent their numbers.
Keep search_query short and use BNS, BNSS, or BSA aliases instead of full Act names.

Recent conversation:
{history}

Question:
{question}
"""

RERANK_PROMPT = """Rank only passages that directly help answer the original question.
Consider exact legal references, source appropriateness, definitions, procedure, punishment,
limitations, and whether a passage is merely semantically similar. Do not alter source text.
For cross-statute questions, retain distinct passages for the underlying offence and its procedure;
a passage need not answer the whole question by itself.
Return every candidate exactly once; use a low score for irrelevant passages.
Scores must be between 0 and 1.

Original question:
{question}

Query analysis:
{analysis}

Candidates:
{candidates}
"""

ANSWER_PROMPT = """You are Legal Lenz. Answer only from the retrieved context.
This is informational support, not legal advice.

Rules:
- If the context is insufficient, say you could not find sufficient information.
- Use numbered inline citations like [1] for every factual claim.
- Do not invent case law, penalties, dates, or legal advice.
- Keep the answer concise and grounded.

Context:
{context}

Recent conversation:
{history}

Original user question:
{question}
"""
