from functools import cache
import json
import os

from google.oauth2 import service_account


@cache
def credentials():
    raw = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not raw:
        return None
    return service_account.Credentials.from_service_account_info(
        json.loads(raw),
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
