from google.cloud import storage
from google.api_core.exceptions import NotFound

from rag.config import PROJECT_ID, UPLOAD_BUCKET
from rag.google_auth import credentials


def _bucket():
    if not UPLOAD_BUCKET:
        raise RuntimeError("Set UPLOAD_BUCKET.")
    return storage.Client(project=PROJECT_ID, credentials=credentials()).bucket(UPLOAD_BUCKET)


def upload_pdf(data: bytes, object_name: str) -> None:
    blob = _bucket().blob(object_name)
    blob.upload_from_string(data, content_type="application/pdf")


def delete_object(object_name: str) -> None:
    try:
        _bucket().blob(object_name).delete()
    except NotFound:
        pass
