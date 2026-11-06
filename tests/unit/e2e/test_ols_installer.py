"""Test cluster installation orchestration without cluster access."""

from pathlib import Path
from threading import Barrier
from unittest.mock import Mock

import pytest
from kubernetes.client.exceptions import ApiException
from openshift.dynamic.exceptions import ConflictError, ForbiddenError

from tests.e2e.utils import cluster_api, ols_installer


@pytest.mark.parametrize("credentials_available", [False, True])
def test_ensure_azure_entra_id_secret_refreshes_credentials(
    monkeypatch: pytest.MonkeyPatch, credentials_available: bool
) -> None:
    """Replace stale credentials only when all new values are available."""
    values = {
        "AZUREOPENAI_ENTRA_ID_TENANT_ID": "tenant",
        "AZUREOPENAI_ENTRA_ID_CLIENT_ID": "client",
        "AZUREOPENAI_ENTRA_ID_CLIENT_SECRET": "new-secret",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    if not credentials_available:
        monkeypatch.delenv("AZUREOPENAI_ENTRA_ID_CLIENT_SECRET")
    calls = Mock()
    monkeypatch.setattr(cluster_api, "delete_resource", calls.delete_resource)
    monkeypatch.setattr(cluster_api, "create_secret", calls.create_secret)
    monkeypatch.setattr(cluster_api, "patch_resource", calls.patch_resource)
    ols_installer.ensure_azure_entra_id_secret()
    if not credentials_available:
        assert not calls.mock_calls
        return
    assert [call[0] for call in calls.mock_calls] == ["create_secret"]
    calls.delete_resource.assert_not_called()
    calls.create_secret.assert_called_once_with(
        "azure-entra-id",
        {"tenant_id": "tenant", "client_id": "client", "client_secret": "new-secret"},
        ignore_existing=False,
    )


@pytest.mark.parametrize("write_fails", [False, True])
def test_existing_entra_secret_is_updated_without_deletion(
    monkeypatch: pytest.MonkeyPatch, write_fails: bool
) -> None:
    """Keep the existing Secret intact if updating rotated credentials fails."""
    for key, value in {
        "AZUREOPENAI_ENTRA_ID_TENANT_ID": "tenant",
        "AZUREOPENAI_ENTRA_ID_CLIENT_ID": "client",
        "AZUREOPENAI_ENTRA_ID_CLIENT_SECRET": "rotated",
    }.items():
        monkeypatch.setenv(key, value)
    calls = Mock()
    calls.create_secret.side_effect = ConflictError(ApiException(status=409))
    if write_fails:
        calls.patch_resource.side_effect = ForbiddenError(ApiException(status=403))
    for operation in ("create_secret", "patch_resource", "delete_resource"):
        monkeypatch.setattr(cluster_api, operation, getattr(calls, operation))
    if write_fails:
        with pytest.raises(ForbiddenError):
            ols_installer.ensure_azure_entra_id_secret()
    else:
        ols_installer.ensure_azure_entra_id_secret()
    calls.delete_resource.assert_not_called()
    calls.create_secret.assert_called_once_with(
        "azure-entra-id",
        {"tenant_id": "tenant", "client_id": "client", "client_secret": "rotated"},
        ignore_existing=False,
    )
    calls.patch_resource.assert_called_once_with(
        "secret",
        "azure-entra-id",
        {
            "stringData": {
                "tenant_id": "tenant",
                "client_id": "client",
                "client_secret": "rotated",
            }
        },
    )


@pytest.mark.parametrize("external_mirrors", [None, "false", "true"])
def test_install_ols_mirror_management(
    monkeypatch: pytest.MonkeyPatch, external_mirrors: str | None
) -> None:
    """Only mutate guest mirrors when they are not externally managed."""
    create, apply = Mock(), Mock()
    monkeypatch.setattr(cluster_api, "create_resource", create)
    monkeypatch.setattr(cluster_api, "apply_file", apply)
    monkeypatch.setattr(ols_installer, "disconnected", False)
    monkeypatch.setattr(
        ols_installer,
        "_run_bundle_with_retry",
        Mock(side_effect=RuntimeError("bundle reached")),
    )
    monkeypatch.delenv("OLS_EXTERNAL_IMAGE_MIRRORS", raising=False)
    if external_mirrors is not None:
        monkeypatch.setenv("OLS_EXTERNAL_IMAGE_MIRRORS", external_mirrors)
    with pytest.raises(RuntimeError, match="bundle reached"):
        ols_installer.install_ols()
    assert create.call_args.args[0]["kind"] == "Namespace"
    if external_mirrors == "true":
        apply.assert_not_called()
    else:
        apply.assert_called_once_with(
            "tests/config/operator_install/imagedigestmirrorset.yaml"
        )


@pytest.mark.parametrize(
    "provider,suffix,inspection,url",
    [
        ("openai", "default", False, None),
        ("rhoai_vllm_lseval", "default", False, "https://model.example.test/v1"),
        ("openai", "mcp_inspection", True, None),
    ],
)
def test_install_ols_creates_olsconfig(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    suffix: str,
    inspection: bool,
    url: str | None,
) -> None:
    """Create the suffix-selected resource with environment substitution."""
    create = Mock(side_effect=RuntimeError("resource captured"))
    monkeypatch.setattr(cluster_api, "create_resource", create)
    monkeypatch.setattr(cluster_api, "delete_resource", Mock())
    monkeypatch.setattr(ols_installer, "disconnected", True)
    monkeypatch.setattr(
        ols_installer, "create_and_config_sas", lambda: ("token", "metrics")
    )
    monkeypatch.setattr(ols_installer, "create_secrets", Mock())
    monkeypatch.setattr(ols_installer, "warm_up_rhokp_image", Mock())
    monkeypatch.setattr(
        ols_installer, "retry_until_timeout_or_success", lambda *_a, **_k: True
    )
    for key, value in {
        "PROVIDER": provider,
        "PROVIDER_KEY_PATH": "unused",
        "KSVC_URL": "https://model.example.test",
        "OLS_CONFIG_SUFFIX": suffix,
    }.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(RuntimeError, match="resource captured"):
        ols_installer.install_ols()
    spec = create.call_args.args[0]["spec"]
    assert create.call_args.kwargs["ignore_existing"] is False
    assert spec["ols"]["guardrails"]["toolResultInspection"]["enabled"] is inspection
    if url:
        assert spec["llm"]["providers"][0]["url"] == url


@pytest.mark.parametrize(
    "provider,count,expected",
    [
        ("openai", 1, "llmcreds"),
        ("openai", 3, "openaicreds"),
        ("watsonx", 3, "watsonxcreds"),
    ],
)
def test_create_secrets_only_refreshes_owned_secret(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    provider: str,
    count: int,
    expected: str,
) -> None:
    """Keep other providers' credentials untouched."""
    path = tmp_path / "key"
    path.write_text("test-key\n")
    delete, create = Mock(), Mock()
    monkeypatch.setattr(cluster_api, "delete_resource", delete)
    monkeypatch.setattr(cluster_api, "create_secret", create)
    ols_installer.create_secrets(provider, str(path), count)
    delete.assert_called_once_with("secret", expected, ignore_missing=True)
    create.assert_called_once_with(expected, {"apitoken": "test-key\n"})


def test_create_secrets_empty_and_bedrock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Handle literal empty credentials and the Bedrock discriminator."""
    delete, create, bedrock = Mock(), Mock(), Mock()
    monkeypatch.setattr(cluster_api, "delete_resource", delete)
    monkeypatch.setattr(cluster_api, "create_secret", create)
    monkeypatch.setattr(ols_installer, "ensure_bedrock_iam_secret", bedrock)
    ols_installer.create_secrets("openai", "empty", 1)
    create.assert_called_once_with("llmcreds", {"apitoken": "empty"})
    ols_installer.create_secrets("bedrock", "iam", 3)
    bedrock.assert_called_once_with("iam")
    assert [call.args for call in delete.call_args_list] == [
        ("secret", "llmcreds"),
        ("secret", "llmcreds"),
    ]


def _deployment(image: str | None) -> dict:
    """Build the operator deployment containing the image argument."""
    return {
        "spec": {
            "template": {
                "spec": {
                    "containers": [
                        {"args": [f"--rhokp-image={image}"] if image else []}
                    ]
                }
            }
        }
    }


def test_warm_up_rhokp_image_concurrent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pull on every node concurrently with the extended timeout."""
    image = "registry.example.com/rhokp@sha256:abc123"
    read = Mock(
        side_effect=[
            _deployment(image),
            {
                "items": [
                    {"metadata": {"name": name}} for name in ("worker-a", "worker-b")
                ]
            },
        ]
    )
    barrier = Barrier(2)
    warm = Mock(side_effect=lambda *_a, **_k: barrier.wait(timeout=30))
    monkeypatch.setattr(cluster_api, "read_resource", read)
    monkeypatch.setattr(cluster_api, "warm_image", warm)
    monkeypatch.delenv("ARTIFACT_DIR", raising=False)
    ols_installer.warm_up_rhokp_image()
    assert sorted(call.args for call in warm.call_args_list) == [
        ("worker-a", image),
        ("worker-b", image),
    ]
    assert [call.kwargs for call in warm.call_args_list] == [
        {"timeout": 600},
        {"timeout": 600},
    ]


@pytest.mark.parametrize("old_image", [None, "old-image"])
def test_warm_up_rhokp_image_marker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, old_image: str | None
) -> None:
    """Create the artifact directory and rewarm only if the image changes."""
    image = "registry.example.com/rhokp@sha256:new"
    read = Mock(
        side_effect=lambda resource, *_a: (
            _deployment(image)
            if resource == "deployment"
            else {"items": [{"metadata": {"name": "worker-a"}}]}
        )
    )
    warm = Mock()
    monkeypatch.setattr(cluster_api, "read_resource", read)
    monkeypatch.setattr(cluster_api, "warm_image", warm)
    artifacts = tmp_path / "artifacts"
    marker = artifacts / ".rhokp_image_warmed"
    if old_image:
        artifacts.mkdir()
        marker.write_text(old_image)
    monkeypatch.setenv("ARTIFACT_DIR", str(artifacts))
    ols_installer.warm_up_rhokp_image()
    warm.assert_called_once_with("worker-a", image, timeout=600)
    assert marker.read_text() == image
    warm.reset_mock()
    ols_installer.warm_up_rhokp_image()
    warm.assert_not_called()


@pytest.mark.parametrize(
    "image,match", [(None, "--rhokp-image"), ("image", "cluster nodes")]
)
def test_warm_up_rhokp_image_missing_inputs(
    monkeypatch: pytest.MonkeyPatch, image: str | None, match: str
) -> None:
    """Reject absent image arguments and empty node lists."""
    monkeypatch.delenv("ARTIFACT_DIR", raising=False)
    monkeypatch.setattr(
        cluster_api,
        "read_resource",
        Mock(side_effect=[_deployment(image), {"items": []}]),
    )
    with pytest.raises(RuntimeError, match=match):
        ols_installer.warm_up_rhokp_image()
