"""Unit tests for _ensure_cert_bundle in runner.py."""

from unittest.mock import patch

import pytest

from runner import _ensure_cert_bundle


class TestEnsureCertBundle:
    """Tests for _ensure_cert_bundle()."""

    def test_raises_when_ssl_cert_file_unset(self, monkeypatch):
        """Raise RuntimeError when SSL_CERT_FILE is not set."""
        monkeypatch.delenv("SSL_CERT_FILE", raising=False)
        with pytest.raises(RuntimeError, match="SSL_CERT_FILE is not set"):
            _ensure_cert_bundle()

    def _run_with_tmp_certs(self, monkeypatch, bundle_path):
        """Patch system CA then invoke _ensure_cert_bundle."""
        monkeypatch.setenv("SSL_CERT_FILE", str(bundle_path))
        with patch("runner._SYSTEM_CA_BUNDLE") as mock_system:
            mock_system.read_bytes.return_value = b"SYSTEM-ROOT-CA"
            _ensure_cert_bundle()

    def test_concatenates_certs_from_sibling_dirs(self, monkeypatch, tmp_path):
        """Gather .crt/.pem files from sibling directories into the bundle."""
        cert_root = tmp_path / "certs"
        bundle_dir = cert_root / "cert-bundle"
        bundle_dir.mkdir(parents=True)
        bundle_path = bundle_dir / "ols.pem"

        service_a = cert_root / "service-a"
        service_a.mkdir()
        (service_a / "ca.crt").write_bytes(b"CERT-A")

        service_b = cert_root / "service-b"
        service_b.mkdir()
        (service_b / "ca.pem").write_bytes(b"CERT-B")

        self._run_with_tmp_certs(monkeypatch, bundle_path)

        content = bundle_path.read_bytes()
        assert b"SYSTEM-ROOT-CA" in content
        assert b"CERT-A" in content
        assert b"CERT-B" in content

    def test_excludes_bundle_dir_from_iteration(self, monkeypatch, tmp_path):
        """The bundle output directory is skipped during iteration."""
        cert_root = tmp_path / "certs"
        bundle_dir = cert_root / "cert-bundle"
        bundle_dir.mkdir(parents=True)
        bundle_path = bundle_dir / "ols.pem"
        (bundle_dir / "stale.pem").write_bytes(b"STALE")

        service = cert_root / "service"
        service.mkdir()
        (service / "ca.crt").write_bytes(b"FRESH")

        self._run_with_tmp_certs(monkeypatch, bundle_path)

        content = bundle_path.read_bytes()
        assert b"STALE" not in content
        assert b"FRESH" in content

    def test_ignores_non_cert_extensions(self, monkeypatch, tmp_path):
        """Only .crt and .pem files are included."""
        cert_root = tmp_path / "certs"
        bundle_dir = cert_root / "cert-bundle"
        bundle_dir.mkdir(parents=True)
        bundle_path = bundle_dir / "ols.pem"

        service = cert_root / "service"
        service.mkdir()
        (service / "ca.crt").write_bytes(b"GOOD")
        (service / "readme.txt").write_bytes(b"IGNORED")
        (service / "config.json").write_bytes(b"IGNORED2")

        self._run_with_tmp_certs(monkeypatch, bundle_path)

        content = bundle_path.read_bytes()
        assert b"GOOD" in content
        assert b"IGNORED" not in content
