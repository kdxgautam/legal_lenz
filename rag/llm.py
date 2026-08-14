from functools import cache
import json

from google import genai
from google.genai import types

from rag.config import EMBEDDING_DIM, EMBEDDING_MODEL, GENERATION_MODEL, LOCATION, PROJECT_ID


@cache
def client():
    return genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=LOCATION,
        http_options=types.HttpOptions(api_version="v1"),
    )


def embed_texts(texts: list[str], task_type: str) -> list[list[float]]:
    embeddings = []
    config = types.EmbedContentConfig(
        task_type=task_type,
        output_dimensionality=EMBEDDING_DIM,
    )
    for index, text in enumerate(texts, start=1):
        response = client().models.embed_content(
            model=EMBEDDING_MODEL,
            contents=types.Content(parts=[types.Part.from_text(text=text)]),
            config=config,
        )
        embeddings.extend(item.values for item in response.embeddings)
        if len(texts) > 25 and index % 25 == 0:
            print(f"Embedded {index}/{len(texts)} chunks")
    if len(embeddings) != len(texts):
        raise RuntimeError(f"Embedding count mismatch: got {len(embeddings)} for {len(texts)} texts.")
    return embeddings


def generate_text(prompt: str) -> str:
    response = client().models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0),
    )
    return (response.text or "").strip()


def generate_json(prompt: str, schema: dict) -> dict:
    response = client().models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_json_schema=schema,
        ),
    )
    return json.loads(response.text or "{}")
