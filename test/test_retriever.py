from rag.retriever import retrieve_chunks


results = retrieve_chunks(
    "What is Article 2?"
)

print(f"Retrieved {len(results)} chunks")


for i, result in enumerate(results):

    print("\n")
    print("=" * 50)

    print(f"CHUNK {i+1}")

    print("\nMETADATA:")
    print(result.metadata)

    print("\nCONTENT:")
    print(result.page_content[:1000])