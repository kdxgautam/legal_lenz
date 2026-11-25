import httpx
import pytest

from rag import ingest, ocr


def test_ocr_request_uses_zero_based_pages_and_cleans_markdown(monkeypatch):
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append(json)
        return httpx.Response(
            200,
            json={"pages": [{"index": 2, "markdown": "# Lease\n\n![img-0.jpeg](img-0.jpeg)\nTerm   ends $^{1}$[now]"}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(ocr, "MISTRAL_API_KEY", "key")
    monkeypatch.setattr(ocr.httpx, "post", fake_post)
    assert ocr.ocr_pdf_pages(b"%PDF-", [3]) == {3: "# Lease\nTerm ends 1[now]"}
    assert calls[0]["pages"] == [2]
    assert calls[0]["document"]["document_url"].startswith("data:application/pdf;base64,")


def test_ocr_http_failure_raises_ocr_error(monkeypatch):
    def fake_post(url, **_):
        return httpx.Response(429, request=httpx.Request("POST", url))

    monkeypatch.setattr(ocr, "MISTRAL_API_KEY", "key")
    monkeypatch.setattr(ocr.httpx, "post", fake_post)
    with pytest.raises(ocr.OcrError):
        ocr.ocr_pdf_pages(b"%PDF-", [1])


def test_only_pages_without_text_are_ocrd(monkeypatch):
    native = "x" * 150
    requested = []
    monkeypatch.setattr(ingest, "ocr_enabled", lambda: True)
    monkeypatch.setattr(
        ingest,
        "ocr_pdf_pages",
        lambda data, pages: requested.extend(pages) or {page: "scanned " * 20 for page in pages},
    )
    texts = ingest.with_ocr_text(b"%PDF-", {1: native, 2: "", 3: "12"})
    assert requested == [2, 3]
    assert texts[1] == native
    assert texts[2].startswith("scanned")


def test_ocr_skipped_when_disabled_or_not_needed(monkeypatch):
    monkeypatch.setattr(ingest, "ocr_pdf_pages", lambda *_: pytest.fail("OCR should not run"))
    monkeypatch.setattr(ingest, "ocr_enabled", lambda: False)
    assert ingest.with_ocr_text(b"%PDF-", {1: ""}) == {1: ""}
    monkeypatch.setattr(ingest, "ocr_enabled", lambda: True)
    assert ingest.with_ocr_text(b"%PDF-", {1: "x" * 150}) == {1: "x" * 150}
