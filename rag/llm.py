from dotenv import load_dotenv

from langchain_groq import ChatGroq


# -----------------------------------
# LOAD ENV VARIABLES
# -----------------------------------
load_dotenv()


# -----------------------------------
# LLM
# -----------------------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
)