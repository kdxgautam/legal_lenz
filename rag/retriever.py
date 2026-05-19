from langchain_chroma import (
    Chroma,
)

from langchain_huggingface import (
    HuggingFaceEmbeddings,
)


# -----------------------------------
# EMBEDDING MODEL
# -----------------------------------
embedding_model = (
    HuggingFaceEmbeddings(
        model_name=(
            "sentence-transformers/"
            "all-MiniLM-L6-v2"
        )
    )
)


# -----------------------------------
# LOAD VECTORSTORE
# -----------------------------------
def load_vectorstore(
    persist_directory,
):

    return Chroma(
        persist_directory=(
            persist_directory
        ),
        embedding_function=(
            embedding_model
        ),
    )


# -----------------------------------
# RETRIEVE CHUNKS
# -----------------------------------
def retrieve_chunks(
    query,
    persist_directory,
    k=5,
):

    vectorstore = load_vectorstore(
        persist_directory
    )

    retriever = (
        vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": k,
            },
        )
    )

    return retriever.invoke(query)