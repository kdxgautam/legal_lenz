from rag import llm


def test_embed_texts_wraps_each_text_as_content(monkeypatch):
    seen = []

    class Models:
        def embed_content(self, model, contents, config):
            seen.append(contents)

            class Response:
                embeddings = [
                    type("Embedding", (), {"values": [float(len(seen))]}),
                ]

            return Response()

    monkeypatch.setattr(llm, "client", lambda: type("Client", (), {"models": Models()})())

    assert llm.embed_texts(["a", "b"], "RETRIEVAL_DOCUMENT") == [[1.0], [2.0]]
    assert len(seen) == 2
