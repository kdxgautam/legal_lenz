import streamlit as st

from rag.chat import ask_question
from rag.config import APPROVED_EMAILS
from rag.db import (
    ChatLimitError,
    append_chat_message,
    create_chat_session,
    delete_chat_session,
    delete_document_row,
    get_owned_document,
    list_chat_sessions,
    list_documents,
    load_chat_session,
    rename_chat_session,
    title_from_question,
    update_chat_selected_documents,
)
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

def load_chat(chat_id: str | None) -> None:
    if not chat_id:
        st.session_state.current_chat_id = None
        st.session_state.current_chat_title = ""
        st.session_state.messages = []
        st.session_state.selected_docs = []
        return
    chat = load_chat_session(email, chat_id)
    if not chat:
        load_chat(None)
        return
    st.session_state.current_chat_id = chat["id"]
    st.session_state.current_chat_title = chat["title"]
    st.session_state.messages = chat["messages"]
    st.session_state.selected_docs = chat["selected_document_ids"]


if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "current_chat_title" not in st.session_state:
    st.session_state.current_chat_title = ""
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_docs" not in st.session_state:
    st.session_state.selected_docs = []

with st.sidebar:
    st.subheader("Chats")
    try:
        chats = list_chat_sessions(email)
    except Exception as exc:
        st.error("Chats are not available. Run migrations and check database configuration.")
        st.caption(str(exc))
        chats = []

    chat_ids = [chat["id"] for chat in chats]
    if st.session_state.current_chat_id not in chat_ids:
        load_chat(chat_ids[0] if chat_ids else None)

    if chat_ids:
        chosen_chat = st.selectbox(
            "Chat session",
            options=chat_ids,
            format_func=lambda chat_id: next(chat["title"] for chat in chats if chat["id"] == chat_id),
            index=chat_ids.index(st.session_state.current_chat_id),
        )
        if chosen_chat != st.session_state.current_chat_id:
            load_chat(chosen_chat)
            st.rerun()

    if st.button("New chat", disabled=len(chats) >= 5):
        try:
            chat_id = create_chat_session(email, "New chat")
            load_chat(chat_id)
            st.rerun()
        except ChatLimitError as exc:
            st.warning(str(exc))
    if len(chats) >= 5:
        st.caption("Delete a chat before creating another.")

    if st.session_state.current_chat_id:
        new_title = st.text_input("Rename chat", value=st.session_state.current_chat_title)
        if st.button("Rename", disabled=not new_title.strip()):
            rename_chat_session(email, st.session_state.current_chat_id, new_title)
            st.session_state.current_chat_title = title_from_question(new_title)
            st.rerun()
        if st.button("Delete chat"):
            delete_chat_session(email, st.session_state.current_chat_id)
            load_chat(None)
            st.rerun()

    st.divider()
    st.subheader("Documents")
    try:
        docs = list_documents(email)
    except Exception as exc:
        st.error("Database is not available. Check Cloud SQL configuration.")
        st.caption(str(exc))
        docs = []

    ready_docs = [doc for doc in docs if doc["status"] == "ready"]
    labels = {doc["id"]: f"{doc['original_filename']} ({doc['type']})" for doc in ready_docs}
    selected_docs = st.multiselect(
        "Selected uploads",
        options=[doc["id"] for doc in ready_docs if doc["type"] == "upload"],
        format_func=lambda doc_id: labels.get(doc_id, doc_id),
        default=[doc_id for doc_id in st.session_state.selected_docs if doc_id in labels],
    )
    if selected_docs != st.session_state.selected_docs:
        st.session_state.selected_docs = selected_docs
        if st.session_state.current_chat_id:
            st.session_state.selected_docs = update_chat_selected_documents(
                email,
                st.session_state.current_chat_id,
                selected_docs,
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
    if st.button("Sign out"):
        load_chat(None)
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
                    if source.get("text"):
                        st.text(source["text"][:900])

question = st.chat_input("Ask about the Constitution, statutes, or selected uploads")
if question:
    if not st.session_state.current_chat_id:
        try:
            load_chat(create_chat_session(email, title_from_question(question), st.session_state.selected_docs))
        except ChatLimitError as exc:
            st.warning(str(exc))
            st.stop()
    elif not st.session_state.messages and st.session_state.current_chat_title == "New chat":
        rename_chat_session(email, st.session_state.current_chat_id, question)
        st.session_state.current_chat_title = title_from_question(question)

    st.session_state.messages.append({"role": "user", "content": question})
    append_chat_message(email, st.session_state.current_chat_id, "user", question)
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
                        if source.get("text"):
                            st.text(source["text"][:900])
        except Exception:
            response = {"answer": "Retrieval or model call failed. Check database, quota, and model access.", "sources": []}
            st.error(response["answer"])
    st.session_state.messages.append({"role": "assistant", "content": response["answer"], "sources": response["sources"]})
    append_chat_message(email, st.session_state.current_chat_id, "assistant", response["answer"], response["sources"])
