from rag.chat import ask_question


response1 = ask_question(
    "What is Article 2?",
    session_id="hexa"
)

print("\nQUESTION 1:\n")
print(response1["answer"])


response2 = ask_question(
    "Explain it simply",
    session_id="hexa"
)

print("\nQUESTION 2:\n")
print(response2["answer"])