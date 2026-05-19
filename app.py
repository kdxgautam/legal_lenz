import setup_db
import os
import uuid
import time

import streamlit as st

from rag.ingest import ingest_pdf
from rag.chat import ask_question


# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="Legal Lenz",
    page_icon="⚖️",
    layout="wide",
)


# -----------------------------------
# SESSION ID
# -----------------------------------
if "session_id" not in st.session_state:

    st.session_state.session_id = (
        str(uuid.uuid4())
    )

SESSION_ID = (
    st.session_state.session_id
)


# -----------------------------------
# SESSION STATE
# -----------------------------------
if "messages" not in st.session_state:

    st.session_state.messages = []

if "latest_upload_db" not in st.session_state:

    st.session_state.latest_upload_db = None


# -----------------------------------
# CUSTOM CSS
# -----------------------------------
st.markdown(
    """
<style>

html, body, [class*="css"] {
    font-family: Inter, sans-serif;
}

.stApp {
    background: #0f1117;
    color: white;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

.main-title {
    font-size: 3rem;
    font-weight: 700;

    background: linear-gradient(
        90deg,
        #7c3aed,
        #06b6d4
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;

    margin-bottom: 0.3rem;
}

.subtitle {
    color: #9ca3af;
    font-size: 1rem;
    margin-bottom: 2rem;
}

.chat-card {
    background: rgba(255,255,255,0.04);

    border: 1px solid
        rgba(255,255,255,0.08);

    padding: 1rem;
    border-radius: 18px;

    backdrop-filter: blur(10px);
}

section[data-testid="stSidebar"] {
    background: #151821;

    border-right:
        1px solid
        rgba(255,255,255,0.05);
}

.sidebar-title {
    font-size: 1.4rem;
    font-weight: 700;
    margin-bottom: 1rem;
}

.stButton button {
    width: 100%;
    border-radius: 12px;
    height: 3rem;

    border: none;

    background: linear-gradient(
        90deg,
        #7c3aed,
        #2563eb
    );

    color: white;
    font-weight: 600;

    transition: 0.2s ease;
}

.stButton button:hover {

    transform: translateY(-2px);

    box-shadow:
        0px 8px 20px
        rgba(124,58,237,0.4);
}

.stChatInput textarea {
    border-radius: 16px !important;
}

.source-card {

    background:
        rgba(255,255,255,0.03);

    border:
        1px solid
        rgba(255,255,255,0.06);

    padding: 1rem;
    border-radius: 14px;

    margin-bottom: 1rem;
}

.source-title {
    font-size: 1rem;
    font-weight: 700;
    color: #60a5fa;
}

.source-meta {
    color: #9ca3af;
    font-size: 0.9rem;
    margin-bottom: 0.5rem;
}

</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------------
# HEADER
# -----------------------------------
st.markdown(
    """
<div class="main-title">
⚖️ Legal Lenz
</div>

<div class="subtitle">
AI Legal Assistant powered by RAG.
Ask questions about the Indian Constitution
or uploaded legal documents.
</div>
""",
    unsafe_allow_html=True,
)


# -----------------------------------
# SIDEBAR
# -----------------------------------
with st.sidebar:

    st.markdown(
        """
<div class="sidebar-title">
📄 Upload Legal PDF
</div>
""",
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Choose PDF",
        type="pdf",
    )

    # -----------------------------------
    # INGEST PDF
    # -----------------------------------
    if st.button("⚡ Ingest PDF"):

        if uploaded_file is not None:

            os.makedirs(
                "data/pdfs",
                exist_ok=True,
            )

            # -----------------------------
            # SAVE PDF
            # -----------------------------
            pdf_path = os.path.join(
                "data/pdfs",
                uploaded_file.name,
            )

            with open(pdf_path, "wb") as f:

                f.write(
                    uploaded_file.read()
                )

            # -----------------------------
            # UNIQUE UPLOAD DB
            # -----------------------------
            upload_id = str(
                int(time.time())
            )

            upload_db_path = (
                f"data/chroma/uploads/"
                f"{SESSION_ID}/"
                f"{upload_id}"
            )

            os.makedirs(
                upload_db_path,
                exist_ok=True,
            )

            # -----------------------------
            # INGEST
            # -----------------------------
            progress = st.progress(0)

            with st.spinner(
                "Processing PDF..."
            ):

                progress.progress(25)

                ingest_pdf(
                    pdf_path=pdf_path,
                    persist_directory=(
                        upload_db_path
                    ),
                    source_name=(
                        uploaded_file.name
                    ),
                )

                progress.progress(100)

            # -----------------------------
            # SAVE ACTIVE DB
            # -----------------------------
            st.session_state[
                "latest_upload_db"
            ] = upload_db_path

            st.success(
                "PDF ingested successfully!"
            )

        else:

            st.warning(
                "Please upload a PDF."
            )

    # -----------------------------------
    # CLEAR CHAT
    # -----------------------------------
    if st.button("🗑️ Clear Chat"):

        st.session_state.messages = []

        st.rerun()

    st.divider()

    st.markdown(
        """
### 💡 Suggested Questions

- What is Article 21?
- Explain Article 14
- Summarize this document
- What compensation is demanded?
"""
    )


# -----------------------------------
# DISPLAY CHAT
# -----------------------------------
for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            f"""
<div class="chat-card">
{message["content"]}
</div>
""",
            unsafe_allow_html=True,
        )


# -----------------------------------
# USER INPUT
# -----------------------------------
user_input = st.chat_input(
    "Ask a legal question..."
)


# -----------------------------------
# HANDLE QUERY
# -----------------------------------
if user_input:

    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
    })

    with st.chat_message("user"):

        st.markdown(
            f"""
<div class="chat-card">
{user_input}
</div>
""",
            unsafe_allow_html=True,
        )

    with st.chat_message("assistant"):

        with st.spinner(
            "Analyzing legal context..."
        ):

            response = ask_question(
                question=user_input,
                session_id=SESSION_ID,
                upload_db_path=(
                    st.session_state.get(
                        "latest_upload_db"
                    )
                ),
            )

            answer = response["answer"]

            sources = response["sources"]

            st.markdown(
                f"""
<div class="chat-card">
{answer}
</div>
""",
                unsafe_allow_html=True,
            )

            # -----------------------------
            # SOURCES
            # -----------------------------
            if sources:

                with st.expander(
                    "📚 View Sources"
                ):

                    for source in sources:

                        metadata = (
                            source.metadata
                        )

                        st.markdown(
                            f"""
<div class="source-card">

<div class="source-title">
{metadata.get("article")}
</div>

<div class="source-meta">
📄 {metadata.get("source_name")}
|
📍 Page {metadata.get("page")}
</div>

<div>
{source.page_content[:700]}...
</div>

</div>
""",
                            unsafe_allow_html=True,
                        )

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
    })