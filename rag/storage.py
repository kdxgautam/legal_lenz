from google.cloud import storage
from google.api_core.exceptions import NotFound

from rag.config import UPLOAD_BUCKET


def _bucket():
    if not UPLOAD_BUCKET:
        raise RuntimeError("Set UPLOAD_BUCKET.")
    return storage.Client().bucket(UPLOAD_BUCKET)


def upload_pdf(data: bytes, object_name: str) -> None:
    blob = _bucket().blob(object_name)
    blob.upload_from_string(data, content_type="application/pdf")


def delete_object(object_name: str) -> None:
    try:
        _bucket().blob(object_name).delete()
    except NotFound:
        pass
