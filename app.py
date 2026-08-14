import streamlit as st

from rag.chat import ask_question
from rag.config import APPROVED_EMAILS
from rag.db import delete_document_row, get_owned_document, list_documents
from rag.ingest import IngestError, ingest_pdf
from rag.storage import delete_object

st.set_page_config(page_title="Legal Lenz", page_icon=":material/gavel:", layout="wide")


def logged_in_email() -> str:
    return (st.user.get("email") or "").lower()


def source_caption(source: dict) -> str:
    metadata = source.get("metadata") or {}
    page_end = source.get("page_end")
    pages = f"pages {source['page']}-{page_end}" if page_end and page_end != source["page"] else f"page {source['page']}"
    if metadata.get("section_number"):
        legal_unit = f"Section {metadata['section_number']}"
        if metadata.get("chapter_number"):
            legal_unit += f" | Chapter {metadata['chapter_number']}"
    else:
        legal_unit = source["article"] or "No article"
    return f"[{source['citation']}] {source['document_name']} | {pages} | {legal_unit}"

st.markdown(
    """
<style>
.stApp { background: #111318; color: #f4f4f5; }
.block-container { max-width: 1120px; padding-top: 1.5rem; }
section[data-testid="stSidebar"] { background: #171a21; border-right: 1px solid #2f3440; }
.small-muted { color: #a1a1aa; font-size: 0.9rem; }
</style>
""",
    unsafe_allow_html=True,
)

email = logged_in_email()
if not email:
    st.title("Legal Lenz")
    st.caption("Controlled legal document assistant.")
    if st.button("Sign in with Google", type="primary"):
        st.login()
    st.stop()

if email not in APPROVED_EMAILS:
    st.error("This Google account is authenticated but not approved for Legal Lenz.")
    if st.button("Sign out"):
        st.logout()
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_docs" not in st.session_state:
    st.session_state.selected_docs = []

with st.sidebar:
    st.subheader("Documents")
    try:
        docs = list_documents(email)
    except Exception as exc:
        st.error("Database is not available. Check Cloud SQL configuration.")
        st.caption(str(exc))
        docs = []

    ready_docs = [doc for doc in docs if doc["status"] == "ready"]
    labels = {doc["id"]: f"{doc['original_filename']} ({doc['type']})" for doc in ready_docs}
    st.session_state.selected_docs = st.multiselect(
        "Selected uploads",
        options=[doc["id"] for doc in ready_docs if doc["type"] == "upload"],
        format_func=lambda doc_id: labels.get(doc_id, doc_id),
        default=[doc_id for doc_id in st.session_state.selected_docs if doc_id in labels],
    )

    upload = st.file_uploader("Upload PDF", type=["pdf"])
    if st.button("Ingest PDF", type="primary"):
        if not upload:
            st.warning("Choose a PDF first.")
        else:
            try:
                with st.spinner("Processing PDF..."):
                    ingest_pdf(upload.getvalue(), upload.name, "upload", email)
                st.success("PDF ingested.")
                st.rerun()
            except IngestError as exc:
                st.error(str(exc))
            except Exception:
                st.error("Upload failed. Check Storage, database, Gemini quota, and model access.")

    delete_id = st.selectbox(
        "Delete upload",
        options=[""] + [doc["id"] for doc in ready_docs if doc["type"] == "upload"],
        format_func=lambda doc_id: "Choose document" if not doc_id else labels.get(doc_id, doc_id),
    )
    if st.button("Delete permanently", disabled=not delete_id):
        owned = get_owned_document(email, delete_id)
        if owned:
            delete_object(owned["gcs_object"])
            delete_document_row(delete_id)
            st.session_state.selected_docs = [doc_id for doc_id in st.session_state.selected_docs if doc_id != delete_id]
            st.success("Document deleted permanently.")
            st.rerun()
        else:
            st.error("Document not found or not owned by this account.")

    st.divider()
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
    if st.button("Sign out"):
        st.logout()

st.title("Legal Lenz")
st.caption("Informational use only. This app does not provide legal advice.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.markdown(source_caption(source))
                    st.text(source["text"][:900])

question = st.chat_input("Ask about the Constitution, statutes, or selected uploads")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving context..."):
                response = ask_question(email, st.session_state.selected_docs, question, st.session_state.messages[:-1])
            st.markdown(response["answer"])
            if response["sources"]:
                with st.expander("Sources"):
                    for source in response["sources"]:
                        st.markdown(source_caption(source))
                        st.text(source["text"][:900])
        except Exception:
            response = {"answer": "Retrieval or model call failed. Check database, quota, and model access.", "sources": []}
            st.error(response["answer"])
    st.session_state.messages.append({"role": "assistant", "content": response["answer"], "sources": response["sources"]})
