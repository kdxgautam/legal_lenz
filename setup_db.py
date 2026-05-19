import os

from rag.ingest import (
    ingest_pdf,
)


DB_PATH = "data/chroma/base"


# -----------------------------------
# BUILD CONSTITUTION DB
# -----------------------------------
if not os.path.exists(
    DB_PATH
):

    print(
        "Creating Constitution DB..."
    )

    os.makedirs(
        DB_PATH,
        exist_ok=True,
    )

    ingest_pdf(
        pdf_path=(
            "data/pdfs/"
            "constitution.pdf"
        ),
        persist_directory=DB_PATH,
        source_name=(
            "Indian Constitution"
        ),
    )

    print(
        "Constitution DB Created!"
    )