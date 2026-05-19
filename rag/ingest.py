import re

from langchain_core.documents import (
    Document,
)

from langchain_community.document_loaders import (
    PyMuPDFLoader,
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

from langchain_huggingface import (
    HuggingFaceEmbeddings,
)

from langchain_chroma import (
    Chroma,
)


# -----------------------------------
# MAX CONSTITUTION ARTICLE NUMBER
# -----------------------------------
MAX_ARTICLE_NUMBER = 395


# -----------------------------------
# CLEAN TEXT
# -----------------------------------
def clean_text(
    text: str,
):

    # remove footer junk
    text = re.split(
        r"_{2,}",
        text,
    )[0]

    # normalize spaces ONLY
    # preserve line structure
    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    # normalize repeated newlines
    text = re.sub(
        r"\n+",
        "\n",
        text,
    )

    return text.strip()


# -----------------------------------
# STRICT ARTICLE EXTRACTION
# -----------------------------------
def extract_articles(
    text,
):

    articles = []

    lines = text.split("\n")

    current_article = None

    current_content = []

    for line in lines:

        line = line.strip()

        # -----------------------------------
        # MATCH ARTICLE FORMAT
        # Example:
        # 21. Protection of life...
        # -----------------------------------
        match = re.match(
            r"^(\d+[A-Z]?)\.\s",
            line,
        )

        if match:

            article_number = (
                match.group(1)
            )

            # -----------------------------------
            # VALIDATE ARTICLE NUMBER
            # -----------------------------------
            numeric_match = re.match(
                r"\d+",
                article_number,
            )

            if not numeric_match:
                continue

            numeric_value = int(
                numeric_match.group()
            )

            # -----------------------------------
            # IGNORE INVALID ARTICLES
            # -----------------------------------
            if numeric_value > MAX_ARTICLE_NUMBER:
                continue

            # -----------------------------------
            # SAVE PREVIOUS ARTICLE
            # -----------------------------------
            if (
                current_article
                and current_content
            ):

                articles.append({
                    "article": (
                        current_article
                    ),
                    "content": (
                        "\n".join(
                            current_content
                        )
                    ),
                })

            # -----------------------------------
            # START NEW ARTICLE
            # -----------------------------------
            current_article = (
                f"Article {article_number}"
            )

            current_content = [line]

        else:

            # -----------------------------------
            # CONTINUE CURRENT ARTICLE
            # -----------------------------------
            if current_article:

                current_content.append(
                    line
                )

    # -----------------------------------
    # FINAL ARTICLE
    # -----------------------------------
    if (
        current_article
        and current_content
    ):

        articles.append({
            "article": current_article,
            "content": (
                "\n".join(
                    current_content
                )
            ),
        })

    return articles


# -----------------------------------
# MAIN INGESTION FUNCTION
# -----------------------------------
def ingest_pdf(
    pdf_path,
    persist_directory,
    source_name="Uploaded Document",
):

    # -----------------------------------
    # LOAD PDF
    # -----------------------------------
    loader = PyMuPDFLoader(
        pdf_path
    )

    pages = loader.load()

    print(
        f"Loaded {len(pages)} pages"
    )

    documents = []

    # -----------------------------------
    # PROCESS PAGE-WISE
    # -----------------------------------
    for page in pages:

        page_number = (
            page.metadata.get(
                "page",
                0,
            )
        )

        cleaned_text = clean_text(
            page.page_content
        )

        # -----------------------------------
        # ONLY EXTRACT ARTICLES
        # FROM MAIN CONSTITUTION BODY
        # -----------------------------------
        if 30 <= page_number <= 220:

            extracted_articles = (
                extract_articles(
                    cleaned_text
                )
            )

        else:

            extracted_articles = []

        # -----------------------------------
        # ARTICLE-AWARE DOCUMENTS
        # -----------------------------------
        if extracted_articles:

            for article in (
                extracted_articles
            ):

                doc = Document(
                    page_content=(
                        article["content"]
                    ),
                    metadata={
                        "source_name": (
                            source_name
                        ),
                        "page": (
                            page_number
                        ),
                        "article": (
                            article["article"]
                        ),
                    },
                )

                documents.append(doc)

        # -----------------------------------
        # GENERIC DOCUMENT FALLBACK
        # -----------------------------------
        else:

            # skip tiny noisy chunks
            if len(cleaned_text) < 100:
                continue

            doc = Document(
                page_content=(
                    cleaned_text
                ),
                metadata={
                    "source_name": (
                        source_name
                    ),
                    "page": (
                        page_number
                    ),
                    "article": (
                        "N/A"
                    ),
                },
            )

            documents.append(doc)

    print(
        f"Created {len(documents)} documents"
    )

    # -----------------------------------
    # CHUNKING
    # -----------------------------------
    splitter = (
        RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
    )

    chunks = splitter.split_documents(
        documents
    )

    print(
        f"Created {len(chunks)} chunks"
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
    # VECTORSTORE
    # -----------------------------------
    vectorstore = Chroma(
        persist_directory=(
            persist_directory
        ),
        embedding_function=(
            embedding_model
        ),
    )

    # -----------------------------------
    # STORE CHUNKS
    # -----------------------------------
    vectorstore.add_documents(
        chunks
    )

    print(
        "Ingestion complete!"
    )