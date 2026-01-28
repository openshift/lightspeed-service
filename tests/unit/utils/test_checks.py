"""Unit tests for checks utilities."""

from pathlib import Path

from ols.constants import NOOP_WITH_TOKEN_AUTHENTICATION_MODULE
from ols.utils.checks import resolve_headers


def test_resolve_headers_empty() -> None:
    """Test resolving empty authorization headers."""
    result = resolve_headers({})
    assert not result


def test_resolve_headers_with_file(tmp_path: Path) -> None:
    """Test resolving authorization headers from file."""
    # Create a temporary secret file
    secret_file = tmp_path / "secret.txt"
    secret_file.write_text("my-secret-token")

    headers = {"Authorization": str(secret_file)}
    result = resolve_headers(headers)

    assert result == {"Authorization": "my-secret-token"}


def test_resolve_headers_with_file_strips_whitespace(
    tmp_path: Path,
) -> None:
    """Test that resolving headers strips trailing whitespace from file content."""
    secret_file = tmp_path / "secret.txt"
    secret_file.write_text("  my-secret-token\n  ")

    headers = {"Authorization": str(secret_file)}
    result = resolve_headers(headers)

    # rstrip() only removes trailing whitespace, not leading
    assert result == {"Authorization": "  my-secret-token"}


def test_resolve_headers_with_nonexistent_file() -> None:
    """Test resolving headers with nonexistent file logs warning and skips."""
    headers = {"Authorization": "/nonexistent/path/to/secret.txt"}
    result = resolve_headers(headers)

    # Should return empty dict when file doesn't exist
    assert not result


def test_resolve_headers_client_token() -> None:
    """Test that client token keyword is preserved."""
    headers = {"Authorization": "client"}
    result = resolve_headers(headers)

    # Should keep "client" as-is for later substitution
    assert result == {"Authorization": "client"}


def test_resolve_headers_kubernetes_token() -> None:
    """Test that kubernetes keyword is preserved when k8s auth is configured."""
    headers = {"Authorization": "kubernetes"}
    result = resolve_headers(headers, auth_module="k8s")

    # Should keep "kubernetes" as-is for later substitution
    assert result == {"Authorization": "kubernetes"}


def test_resolve_headers_kubernetes_token_with_noop_with_token() -> None:
    """Test that kubernetes keyword is preserved when noop_with_token auth is configured."""
    headers = {"Authorization": "kubernetes"}
    result = resolve_headers(headers, auth_module=NOOP_WITH_TOKEN_AUTHENTICATION_MODULE)

    # Should keep "kubernetes" as-is for later substitution (for testing)
    assert result == {"Authorization": "kubernetes"}


def test_resolve_headers_multiple_headers(tmp_path: Path) -> None:
    """Test resolving multiple authorization headers."""
    # Create multiple secret files
    auth_file = tmp_path / "auth.txt"
    auth_file.write_text("auth-token")
    api_key_file = tmp_path / "api_key.txt"
    api_key_file.write_text("api-key-value")

    headers = {
        "Authorization": str(auth_file),
        "X-API-Key": str(api_key_file),
    }
    result = resolve_headers(headers)

    assert result == {
        "Authorization": "auth-token",
        "X-API-Key": "api-key-value",
    }


def test_resolve_headers_mixed_types(tmp_path: Path) -> None:
    """Test resolving mixed header types (file, client, kubernetes)."""
    # Create a secret file
    secret_file = tmp_path / "secret.txt"
    secret_file.write_text("file-secret")

    headers = {
        "Authorization": "client",
        "X-API-Key": str(secret_file),
        "X-K8s-Token": "kubernetes",
    }

    result = resolve_headers(headers, auth_module="k8s")

    # Special keywords should be preserved, file should be resolved
    assert result["Authorization"] == "client"
    assert result["X-API-Key"] == "file-secret"
    assert result["X-K8s-Token"] == "kubernetes"


def test_resolve_headers_file_read_error(tmp_path: Path) -> None:
    """Test handling of file read errors."""
    # Create a directory instead of a file to cause an error
    secret_dir = tmp_path / "secret_dir"
    secret_dir.mkdir()

    headers = {"Authorization": str(secret_dir)}
    result = resolve_headers(headers)

    # Should handle error gracefully and return empty dict
    assert not result


def test_resolve_headers_kubernetes_requires_k8s_auth(caplog) -> None:
    """Test that kubernetes placeholder logs warning and returns empty dict with non-k8s auth."""
    headers = {"Authorization": "kubernetes"}

    result = resolve_headers(headers, auth_module="azure")

    # Should return empty dict when kubernetes placeholder used with
    # non-k8s/non-noop_with_token auth
    assert result == {}
    assert "kubernetes" in caplog.text.lower()
    assert "k8s" in caplog.text
    assert "azure" in caplog.text
    assert "skipped" in caplog.text.lower()


def test_resolve_headers_kubernetes_with_no_auth_module(caplog) -> None:
    """Test that kubernetes placeholder logs warning when auth module is None."""
    headers = {"Authorization": "kubernetes"}

    result = resolve_headers(headers, auth_module=None)

    # Should return empty dict
    assert result == {}
    assert "kubernetes" in caplog.text.lower()
    assert "skipped" in caplog.text.lower()
