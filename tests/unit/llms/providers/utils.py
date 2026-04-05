"""Shared helpers for LLM provider unit tests."""

import json
import secrets
from urllib.parse import quote

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa


def generate_service_account_json_string(
    project_id: str = "ols-gemini-ai-testing",
    client_email: str = "abc@def.com",
    client_id: str = "xyz123",
    private_key_id: str | None = None,
) -> str:
    """Build a JSON string shaped like a Google Cloud service account key file.

    Generates a fresh RSA private key for tests. The key is not registered in GCP.

    Args:
        project_id: GCP project id field.
        client_email: Service account email.
        client_id: Opaque numeric client id string.
        private_key_id: Key id; if omitted, a random hex string is used.

    Returns:
        Pretty-printed JSON with four-space indent and a trailing newline, using
        the same top-level keys as a Google Cloud service account key file.
    """
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    key_id = private_key_id if private_key_id is not None else secrets.token_hex(20)
    encoded_email = quote(client_email, safe="")
    payload = {
        "type": "service_account",
        "project_id": project_id,
        "private_key_id": key_id,
        "private_key": pem.decode(),
        "client_email": client_email,
        "client_id": client_id,
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": (
            f"https://www.googleapis.com/robot/v1/metadata/x509/{encoded_email}"
        ),
        "universe_domain": "googleapis.com",
    }
    return json.dumps(payload, indent=4) + "\n"
