import re

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
)

from rag.llm import llm

from rag.retriever import (
    retrieve_chunks,
    load_vectorstore,
)

from rag.prompts import (
    SYSTEM_PROMPT,
)


# -----------------------------------
# CHAT STORE
# -----------------------------------
chat_store = {}


# -----------------------------------
# GET HISTORY
# -----------------------------------
def get_chat_history(
    session_id,
):

    if session_id not in chat_store:

        chat_store[session_id] = []

    return chat_store[session_id]


# -----------------------------------
# REWRITE QUERY
# -----------------------------------
def rewrite_query(
    question,
    history,
):

    if not history:

        return question

    prompt = f"""
Rewrite the latest question into a standalone question.

CHAT HISTORY:
{history}

QUESTION:
{question}
"""

    response = llm.invoke(
        prompt
    )

    return response.content


# -----------------------------------
# EXTRACT ARTICLE NUMBER
# -----------------------------------
def extract_article_number(
    question,
):

    match = re.search(
        r"article\s+(\d+)",
        question.lower(),
    )

    if match:

        return (
            f"Article {match.group(1)}"
        )

    return None


# -----------------------------------
# BUILD CONTEXT
# -----------------------------------
def build_context(
    chunks,
):

    context = ""

    for chunk in chunks:

        metadata = (
            chunk.metadata
        )

        context += f"""

SOURCE:
{metadata.get("source_name")}

ARTICLE:
{metadata.get("article")}

PAGE:
{metadata.get("page")}

TEXT:
{chunk.page_content}

-----------------------------------
"""

    return context


# -----------------------------------
# RETRIEVE CHUNKS
# -----------------------------------
def retrieve_all_chunks(
    question,
    session_id,
    upload_db_path=None,
):

    all_chunks = []

    # -----------------------------------
    # USER UPLOADS FIRST
    # -----------------------------------
    if upload_db_path:

        try:

            upload_chunks = (
                retrieve_chunks(
                    query=question,
                    persist_directory=(
                        upload_db_path
                    ),
                    k=5,
                )
            )

            all_chunks.extend(
                upload_chunks
            )

        except Exception as e:

            print(
                f"Upload retrieval failed: {e}"
            )

    # -----------------------------------
    # EXACT ARTICLE MATCH
    # -----------------------------------
    article_name = (
        extract_article_number(
            question
        )
    )

    if article_name:

        try:

            base_vectorstore = (
                load_vectorstore(
                    "data/chroma/base"
                )
            )

            exact_matches = (
                base_vectorstore.similarity_search(
                    query=question,
                    k=3,
                    filter={
                        "article": (
                            article_name
                        )
                    },
                )
            )

            if exact_matches:

                all_chunks.extend(
                    exact_matches
                )

                return all_chunks

        except Exception as e:

            print(
                f"Exact retrieval failed: {e}"
            )

    # -----------------------------------
    # CONSTITUTION RETRIEVAL
    # -----------------------------------
    try:

        base_chunks = retrieve_chunks(
            query=question,
            persist_directory=(
                "data/chroma/base"
            ),
            k=3,
        )

        all_chunks.extend(
            base_chunks
        )

    except Exception as e:

        print(
            f"Base retrieval failed: {e}"
        )

    return all_chunks


# -----------------------------------
# GENERATE ANSWER
# -----------------------------------
def generate_answer(
    question,
    context,
    history,
):

    prompt = f"""
{SYSTEM_PROMPT}

-----------------------------------
CHAT HISTORY
-----------------------------------

{history}

-----------------------------------
RETRIEVED CONTEXT
-----------------------------------

{context}

-----------------------------------
QUESTION
-----------------------------------

{question}

-----------------------------------
INSTRUCTIONS
-----------------------------------

Answer ONLY using retrieved context.

If answering about:
- Constitution Articles → prioritize exact matches
- Uploaded PDFs → prioritize uploaded content

Be concise and factual.
"""

    response = llm.invoke(
        prompt
    )

    return response.content


# -----------------------------------
# MAIN CHAT FUNCTION
# -----------------------------------
def ask_question(
    question,
    session_id="default",
    upload_db_path=None,
):

    history = get_chat_history(
        session_id
    )

    standalone_question = (
        rewrite_query(
            question,
            history,
        )
    )

    chunks = retrieve_all_chunks(
        standalone_question,
        session_id,
        upload_db_path,
    )

    context = build_context(
        chunks
    )

    answer = generate_answer(
        standalone_question,
        context,
        history,
    )

    history.append(
        HumanMessage(
            content=question
        )
    )

    history.append(
        AIMessage(
            content=answer
        )
    )

    return {
        "answer": answer,
        "sources": chunks,
    }