"""Test RHOAI vLLM model-profile loading."""

import os
import shlex
import subprocess
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).parents[3]
DEPLOY_SCRIPT = PROJECT_ROOT / "tests/rhoai/scripts/deploy-vllm.sh"
PROFILE_SCRIPT = PROJECT_ROOT / "tests/rhoai/scripts/model-profile.sh"
RUNTIME_MANIFEST = PROJECT_ROOT / "tests/rhoai/manifests/vllm/vllm-runtime-gpu.yaml"
INFERENCE_SERVICE_MANIFEST = (
    PROJECT_ROOT / "tests/rhoai/manifests/vllm/vllm-inference-service-gpu.yaml"
)
PROFILE_VARIABLES = (
    "VLLM_MODEL",
    "VLLM_TOOL_CALL_PARSER",
    "VLLM_CHAT_TEMPLATE_URL",
    "VLLM_CHAT_TEMPLATE_CONFIGMAP",
    "VLLM_CHAT_TEMPLATE_KEY",
    "VLLM_GPU_COUNT",
    "VLLM_TENSOR_PARALLEL_SIZE",
    "VLLM_MAX_MODEL_LEN",
    "VLLM_GPU_MEMORY_UTILIZATION",
    "VLLM_RUNTIME_CPU_REQUEST",
    "VLLM_RUNTIME_CPU_LIMIT",
    "VLLM_RUNTIME_MEMORY_REQUEST",
    "VLLM_RUNTIME_MEMORY_LIMIT",
    "VLLM_INFERENCE_SERVICE_CPU_REQUEST",
    "VLLM_INFERENCE_SERVICE_CPU_LIMIT",
    "VLLM_INFERENCE_SERVICE_MEMORY_REQUEST",
    "VLLM_INFERENCE_SERVICE_MEMORY_LIMIT",
)


def conflicting_value(variable: str) -> str:
    """Return a value that a selected profile must replace for ``variable``."""
    return f"host-value-for-{variable.lower()}"


def profile_environment(profile_name: str | None = None) -> dict[str, str]:
    """Build an isolated environment with conflicting profile output values."""
    environment = {
        "PATH": os.defpath,
        "VLLM_IMAGE": "example.invalid/vllm:test",
        **{variable: conflicting_value(variable) for variable in PROFILE_VARIABLES},
    }
    if profile_name is not None:
        environment["VLLM_MODEL_PROFILE"] = profile_name
    return environment


def load_profile(profile_name: str | None = None) -> dict[str, str]:
    """Load a profile in isolation and return every shell-escaped exported value."""
    result = subprocess.run(  # noqa: S603
        [
            "/bin/bash",
            "-c",
            'source "$1" && load_vllm_model_profile && '
            'for variable in "${@:2}"; do '
            'value="$(printenv "$variable")" || exit 1; '
            'printf "%s=%q\\n" "$variable" "$value"; '
            "done",
            "/bin/bash",
            str(PROFILE_SCRIPT),
            *PROFILE_VARIABLES,
        ],
        check=True,
        capture_output=True,
        cwd=PROJECT_ROOT,
        env=profile_environment(profile_name),
        text=True,
    )
    return {
        name: shlex.split(value)[0]
        for line in result.stdout.splitlines()
        for name, value in [line.split("=", maxsplit=1)]
    }


def assert_profile_overwrites_host_values(profile: dict[str, str]) -> None:
    """Assert that the loader exports each contract value instead of inheriting it."""
    assert set(profile) == set(PROFILE_VARIABLES)
    for variable, value in profile.items():
        assert value != conflicting_value(variable)


def render_manifest(
    manifest: Path, profile_name: str | None = None
) -> dict[str, object]:
    """Load a profile and render one manifest without inheriting host configuration."""
    result = subprocess.run(  # noqa: S603
        [
            "/bin/bash",
            "-c",
            'source "$1" && load_vllm_model_profile && envsubst < "$2"',
            "/bin/bash",
            str(PROFILE_SCRIPT),
            str(manifest),
        ],
        check=True,
        capture_output=True,
        cwd=PROJECT_ROOT,
        env=profile_environment(profile_name),
        text=True,
    )
    return yaml.safe_load(result.stdout)


def argument_value(arguments: list[str], argument_name: str) -> str:
    """Return the value immediately following a vLLM command-line argument."""
    return arguments[arguments.index(argument_name) + 1]


def assert_resources(
    resources: dict[str, object],
    *,
    gpu_count: int,
    cpu_request: str,
    memory_request: str,
    cpu_limit: str,
    memory_limit: str,
) -> None:
    """Assert a container resource configuration matches its selected profile."""
    assert resources == {
        "limits": {
            "nvidia.com/gpu": gpu_count,
            "cpu": cpu_limit,
            "memory": memory_limit,
        },
        "requests": {
            "nvidia.com/gpu": gpu_count,
            "cpu": cpu_request,
            "memory": memory_request,
        },
    }


def test_default_profile_preserves_llama_configuration() -> None:
    """Load every existing Llama value and an intentional single-GPU parallelism default."""
    profile = load_profile()

    assert_profile_overwrites_host_values(profile)
    assert profile["VLLM_MODEL"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert profile["VLLM_TOOL_CALL_PARSER"] == "llama3_json"
    assert (
        profile["VLLM_CHAT_TEMPLATE_URL"]
        == "https://raw.githubusercontent.com/vllm-project/vllm/main/examples/tool_chat_template_llama3.1_json.jinja"
    )
    assert profile["VLLM_GPU_COUNT"] == "1"
    assert profile["VLLM_TENSOR_PARALLEL_SIZE"] == "1"
    assert profile["VLLM_MAX_MODEL_LEN"] == "35936"
    assert profile["VLLM_GPU_MEMORY_UTILIZATION"] == "0.9"
    assert profile["VLLM_CHAT_TEMPLATE_CONFIGMAP"] == "vllm-chat-template"
    assert profile["VLLM_CHAT_TEMPLATE_KEY"] == "tool_chat_template_llama3.1_json.jinja"
    assert profile["VLLM_RUNTIME_CPU_LIMIT"] == "6"
    assert profile["VLLM_RUNTIME_MEMORY_LIMIT"] == "20Gi"
    assert profile["VLLM_RUNTIME_CPU_REQUEST"] == "4"
    assert profile["VLLM_RUNTIME_MEMORY_REQUEST"] == "16Gi"
    assert profile["VLLM_INFERENCE_SERVICE_CPU_LIMIT"] == "4"
    assert profile["VLLM_INFERENCE_SERVICE_MEMORY_LIMIT"] == "20Gi"
    assert profile["VLLM_INFERENCE_SERVICE_CPU_REQUEST"] == "2"
    assert profile["VLLM_INFERENCE_SERVICE_MEMORY_REQUEST"] == "16Gi"


def test_gemma_profile_loads_gemma_configuration() -> None:
    """Load the Gemma 4 profile's model and GPU topology without host overrides."""
    profile = load_profile("gemma-4-31b")

    assert_profile_overwrites_host_values(profile)
    assert profile["VLLM_MODEL"] == "google/gemma-4-31B-it"
    assert profile["VLLM_TOOL_CALL_PARSER"] == "gemma4"
    assert (
        profile["VLLM_CHAT_TEMPLATE_URL"]
        == "https://huggingface.co/google/gemma-4-31B-it/raw/main/chat_template.jinja"
    )
    assert profile["VLLM_CHAT_TEMPLATE_KEY"] == "gemma-4-chat-template.jinja"
    assert profile["VLLM_GPU_COUNT"] == "4"
    assert profile["VLLM_TENSOR_PARALLEL_SIZE"] == "4"
    assert profile["VLLM_MAX_MODEL_LEN"] == "16000"
    assert profile["VLLM_GPU_MEMORY_UTILIZATION"] == "0.9"
    assert profile["VLLM_RUNTIME_CPU_REQUEST"] == "32"
    assert profile["VLLM_RUNTIME_MEMORY_REQUEST"] == "128Gi"
    assert profile["VLLM_RUNTIME_CPU_LIMIT"] == "40"
    assert profile["VLLM_RUNTIME_MEMORY_LIMIT"] == "160Gi"
    assert profile["VLLM_INFERENCE_SERVICE_CPU_REQUEST"] == "32"
    assert profile["VLLM_INFERENCE_SERVICE_MEMORY_REQUEST"] == "128Gi"
    assert profile["VLLM_INFERENCE_SERVICE_CPU_LIMIT"] == "40"
    assert profile["VLLM_INFERENCE_SERVICE_MEMORY_LIMIT"] == "160Gi"


def test_rendered_llama_manifests_preserve_current_configuration() -> None:
    """Render the default profile with the current Llama serving configuration."""
    runtime = render_manifest(RUNTIME_MANIFEST)
    inference_service = render_manifest(INFERENCE_SERVICE_MANIFEST)

    container = runtime["spec"]["containers"][0]
    arguments = container["args"]
    assert argument_value(arguments, "--model") == "meta-llama/Llama-3.1-8B-Instruct"
    assert argument_value(arguments, "--tool-call-parser") == "llama3_json"
    assert argument_value(arguments, "--tensor-parallel-size") == "1"
    assert (
        argument_value(arguments, "--chat-template")
        == "/mnt/chat-template/chat_template.jinja"
    )
    assert argument_value(arguments, "--max-model-len") == "35936"
    assert argument_value(arguments, "--gpu-memory-utilization") == "0.9"
    assert_resources(
        container["resources"],
        gpu_count=1,
        cpu_request="4",
        memory_request="16Gi",
        cpu_limit="6",
        memory_limit="20Gi",
    )
    assert runtime["spec"]["volumes"][0]["configMap"] == {
        "name": "vllm-chat-template",
        "items": [
            {
                "key": "tool_chat_template_llama3.1_json.jinja",
                "path": "chat_template.jinja",
            }
        ],
    }
    assert_resources(
        inference_service["spec"]["predictor"]["model"]["resources"],
        gpu_count=1,
        cpu_request="2",
        memory_request="16Gi",
        cpu_limit="4",
        memory_limit="20Gi",
    )


def test_rendered_gemma_manifests_use_selected_profile_values() -> None:
    """Render Gemma model, parser, GPU topology, resources, and chat template values."""
    runtime = render_manifest(RUNTIME_MANIFEST, "gemma-4-31b")
    inference_service = render_manifest(INFERENCE_SERVICE_MANIFEST, "gemma-4-31b")

    container = runtime["spec"]["containers"][0]
    arguments = container["args"]
    assert argument_value(arguments, "--model") == "google/gemma-4-31B-it"
    assert argument_value(arguments, "--tool-call-parser") == "gemma4"
    assert argument_value(arguments, "--tensor-parallel-size") == "4"
    assert (
        argument_value(arguments, "--chat-template")
        == "/mnt/chat-template/chat_template.jinja"
    )
    assert_resources(
        container["resources"],
        gpu_count=4,
        cpu_request="32",
        memory_request="128Gi",
        cpu_limit="40",
        memory_limit="160Gi",
    )
    assert runtime["spec"]["volumes"][0]["configMap"] == {
        "name": "vllm-chat-template",
        "items": [
            {"key": "gemma-4-chat-template.jinja", "path": "chat_template.jinja"}
        ],
    }
    assert_resources(
        inference_service["spec"]["predictor"]["model"]["resources"],
        gpu_count=4,
        cpu_request="32",
        memory_request="128Gi",
        cpu_limit="40",
        memory_limit="160Gi",
    )


def test_deploy_renders_profile_values_in_every_applied_manifest(
    tmp_path: Path,
) -> None:
    """Apply manifests only after substituting every selected profile value."""
    applied_manifests = tmp_path / "applied-manifests.yaml"
    mock_oc = tmp_path / "oc"
    mock_oc.write_text(
        "#!/bin/bash\n"
        'if [[ "$1" == "apply" && "$2" == "-f" ]]; then\n'
        f'  printf "%s\\n" "--- manifest ---" >> "{applied_manifests}"\n'
        '  if [[ "$3" == "-" ]]; then\n'
        f'    cat >> "{applied_manifests}"\n'
        "  else\n"
        f'    cat "$3" >> "{applied_manifests}"\n'
        "  fi\n"
        "fi\n"
        'printf "ready\\n"\n'
    )
    mock_oc.chmod(0o755)
    mock_sleep = tmp_path / "sleep"
    mock_sleep.write_text("#!/bin/bash\n")
    mock_sleep.chmod(0o755)

    subprocess.run(  # noqa: S603
        ["/bin/bash", str(DEPLOY_SCRIPT), str(PROJECT_ROOT / "tests/rhoai")],
        check=True,
        capture_output=True,
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "VLLM_IMAGE": "example.invalid/vllm:test",
            "VLLM_MODEL_PROFILE": "gemma-4-31b",
        },
        text=True,
        timeout=10,
    )

    applied = applied_manifests.read_text()
    assert applied.count("--- manifest ---") == 2
    assert "${VLLM_" not in applied
    assert "nvidia.com/gpu: 4" in applied


def test_deploy_rejects_unsupported_profile_before_calling_oc(tmp_path: Path) -> None:
    """Reject an invalid profile before attempting any cluster operation."""
    oc_log = tmp_path / "oc.log"
    mock_oc = tmp_path / "oc"
    mock_oc.write_text(
        f'#!/bin/bash\nprintf "%s\\n" "$*" >> "{oc_log}"\nprintf "ready\\n"\n'
    )
    mock_oc.chmod(0o755)

    result = subprocess.run(  # noqa: S603
        ["/bin/bash", str(DEPLOY_SCRIPT), str(PROJECT_ROOT / "tests/rhoai")],
        capture_output=True,
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "VLLM_MODEL_PROFILE": "unsupported",
        },
        text=True,
        timeout=10,
    )

    assert result.returncode != 0
    assert "Unsupported VLLM model profile" in result.stderr
    assert not oc_log.exists()


def test_unsupported_profile_lists_supported_names() -> None:
    """Reject a profile name that has no vLLM configuration in an isolated environment."""
    result = subprocess.run(  # noqa: S603
        [
            "/bin/bash",
            "-c",
            'source "$1" && load_vllm_model_profile',
            "/bin/bash",
            str(PROFILE_SCRIPT),
        ],
        capture_output=True,
        cwd=PROJECT_ROOT,
        env=profile_environment("unsupported"),
        text=True,
    )

    assert result.returncode != 0
    assert "llama-3.1-8b" in result.stderr
    assert "gemma-4-31b" in result.stderr
