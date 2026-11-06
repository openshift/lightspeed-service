"""Functions to install the service onto an OCP cluster using the OLS operator."""

import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml
from openshift.dynamic.exceptions import ConflictError, DynamicApiError

from ols.constants import DEFAULT_CONFIGURATION_FILE
from tests.e2e.utils import cluster as cluster_utils
from tests.e2e.utils import cluster_api
from tests.e2e.utils.data_collector_control import configure_exporter_for_e2e_tests
from tests.e2e.utils.olsconfig_cr import apply_e2e_olsconfig
from tests.e2e.utils.retry import retry_until_timeout_or_success
from tests.e2e.utils.wait_for_ols import wait_for_ols

CLUSTER_RETRY_COUNT = 120
CLUSTER_RETRY_DELAY = 5

disconnected = os.getenv("DISCONNECTED", "")


def _app_server_deployment_exists() -> bool:
    """Return whether the OLS API Deployment exists."""
    return cluster_utils.deployment_exists("lightspeed-app-server")


def _wait_for_operator_controller_ready() -> None:
    """Wait until the operator manager pod exists and all containers are ready."""
    r = retry_until_timeout_or_success(
        60,
        5,
        lambda: (
            pods := cluster_utils.get_pod_by_prefix(
                prefix="lightspeed-operator-controller-manager", fail_not_found=False
            )
        )
        and all(
            status == "true"
            for status in cluster_utils.get_container_ready_status(pods[0])
        ),
        "Waiting for operator controller to be ready",
    )
    if not r:
        raise Exception(
            "Timed out waiting for operator controller manager to become ready"
        )


def _dump_ols_install_debug() -> None:
    """Print cluster state to logs when install waits fail (Konflux / Azure debugging)."""
    for resource, name in (
        ("olsconfig", "cluster"),
        ("deployment", None),
        ("clusterserviceversion", None),
    ):
        try:
            print(f"--- {resource} (debug) ---")
            print(yaml.safe_dump(cluster_api.read_resource(resource, name)))
        except Exception as e:
            print(f"Could not get {resource}: {e}")


def create_and_config_sas() -> tuple[str, str]:
    """Create and provide access to service accounts for testing.

    Returns:
        tuple containing token and metrics token.
    """
    cluster_utils.create_user("test-user", ignore_existing_resource=True)
    cluster_utils.create_user("metrics-test-user", ignore_existing_resource=True)
    token = cluster_utils.get_token_for("test-user")
    metrics_token = cluster_utils.get_token_for("metrics-test-user")
    print("created test service account users")

    # grant the test service accounts permission to query ols and retrieve metrics
    cluster_utils.grant_sa_user_access("test-user", "lightspeed-operator-query-access")
    cluster_utils.grant_sa_user_access(
        "metrics-test-user", "lightspeed-operator-ols-metrics-reader"
    )
    print("test service account permissions granted")

    # grant pod listing permission to test-user - to test the tools,
    # more specifically the we need the test-user be able to see pods
    # in the namespace
    cluster_api.configure_pod_reader()

    print("Granted test-user permission to list pods.")

    return token, metrics_token


def update_ols_config() -> None:
    """Create the ols config configmap with log and collector config for e2e tests.

    Returns:
        Nothing.
    """
    # modify olsconfig configmap
    configmap = cluster_api.read_resource("configmap", "olsconfig")
    olsconfig = yaml.safe_load(configmap["data"][DEFAULT_CONFIGURATION_FILE])

    # one of our libs logs a secrets in debug mode which causes the pod
    # logs beying redacted/removed completely - we need log at info level
    olsconfig["ols_config"]["logging_config"]["lib_log_level"] = "INFO"

    # The operator auto-generates reference_content pointing to a RAG index
    # that doesn't exist in the CI image; strip it for cluster_updates tests
    # which evaluate MCP tool calling, not RAG document retrieval.
    if os.getenv("OLS_CONFIG_SUFFIX") == "cluster_updates":
        olsconfig["ols_config"].pop("reference_content", None)

    # patch reference content config for new format
    # Todo: remove this when the operator PR is merged:
    # https://github.com/openshift/lightspeed-operator/pull/668
    if (
        "reference_content" in olsconfig["ols_config"]
        and "indexes" not in olsconfig["ols_config"]["reference_content"]
    ):
        old_rag_config = olsconfig["ols_config"]["reference_content"]
        product_docs_index_id = ""
        product_docs_index_path = ""
        if "product_docs_index_id" in old_rag_config:
            product_docs_index_id = old_rag_config["product_docs_index_id"]
        if "product_docs_index_path" in old_rag_config:
            product_docs_index_path = old_rag_config["product_docs_index_path"]
        olsconfig["ols_config"]["reference_content"] = {
            "indexes": [
                {
                    "product_docs_index_id": product_docs_index_id,
                    "product_docs_index_path": product_docs_index_path,
                }
            ],
        }

    configmap["data"][DEFAULT_CONFIGURATION_FILE] = yaml.dump(olsconfig)
    cluster_api.patch_resource(
        "configmap",
        "olsconfig",
        {
            "data": {
                DEFAULT_CONFIGURATION_FILE: configmap["data"][
                    DEFAULT_CONFIGURATION_FILE
                ]
            }
        },
    )


def replace_ols_image(ols_image: str) -> None:
    """Replace the existing ols image with a new one.

    Args:
        ols_image (str): the new ols image to be added to the server pod.

    Returns:
        Nothing.
    """
    print(f"Updating deployment to use OLS image {ols_image}")

    # Ensure the operator controller manager pod is gone before touching anything else
    retry_until_timeout_or_success(
        CLUSTER_RETRY_COUNT,
        CLUSTER_RETRY_DELAY,
        lambda: not cluster_utils.get_pod_by_prefix(
            "lightspeed-operator-controller-manager", fail_not_found=False
        ),
        "Waiting for operator controller manager pod to be gone",
    )

    # scale down the ols api server so we can ensure no pods
    # are still running the unsubstituted image
    cluster_api.scale_deployment("lightspeed-app-server", 0)

    # wait for the old ols api pod to go away due to deployment being scaled down
    retry_until_timeout_or_success(
        CLUSTER_RETRY_COUNT,
        CLUSTER_RETRY_DELAY,
        lambda: not cluster_utils.get_pod_by_prefix(fail_not_found=False),
        "Waiting for OLS API pod to be scaled down",
    )

    # update the OLS deployment to use the new image from CI/OLS_IMAGE env var
    cluster_api.patch_resource(
        "deployment",
        "lightspeed-app-server",
        [
            {
                "op": "replace",
                "path": "/spec/template/spec/containers/0/image",
                "value": ols_image,
            }
        ],
        patch_type="json",
    )


_AZURE_ENTRA_ENV_KEYS: tuple[str, ...] = (
    "AZUREOPENAI_ENTRA_ID_TENANT_ID",
    "AZUREOPENAI_ENTRA_ID_CLIENT_ID",
    "AZUREOPENAI_ENTRA_ID_CLIENT_SECRET",
)


def ensure_azure_entra_id_secret() -> None:
    """Create openshift-lightspeed/azure-entra-id when Entra credentials are available."""
    values = {k: os.getenv(k) for k in _AZURE_ENTRA_ENV_KEYS}
    missing = [k for k in _AZURE_ENTRA_ENV_KEYS if not values[k]]
    if missing:
        print(
            "Skipping azure-entra-id secret creation (unset or empty): "
            f"{', '.join(missing)}. "
            "Azure OpenAI via Entra will not work until these CI/Vault/Prow env vars "
            "are set."
        )
        return

    tenant_id = values["AZUREOPENAI_ENTRA_ID_TENANT_ID"]
    client_id = values["AZUREOPENAI_ENTRA_ID_CLIENT_ID"]
    client_secret = values["AZUREOPENAI_ENTRA_ID_CLIENT_SECRET"]

    print("Ensuring azure-entra-id secret exists...")
    credentials = {
        "tenant_id": tenant_id,
        "client_id": client_id,
        "client_secret": client_secret,
    }
    try:
        cluster_api.create_secret("azure-entra-id", credentials, ignore_existing=False)
    except ConflictError:
        cluster_api.patch_resource(
            "secret", "azure-entra-id", {"stringData": credentials}
        )


_BEDROCK_IAM_ENV_KEYS: tuple[str, ...] = (
    "BEDROCK_AWS_ACCESS_KEY_ID",
    "BEDROCK_AWS_SECRET_ACCESS_KEY",
)

_BEDROCK_ROLE_ENV_KEYS: tuple[str, ...] = (
    "BEDROCK_ROLE_AWS_ACCESS_KEY_ID",
    "BEDROCK_ROLE_AWS_SECRET_ACCESS_KEY",
    "BEDROCK_ROLE_ARN",
)


def ensure_bedrock_iam_secret(creds: str) -> None:
    """Create openshift-lightspeed/llmcreds with AWS IAM credentials for Bedrock.

    Bedrock does not use an API token file like other providers. Instead, the
    ``PROVIDER_KEY_PATH`` run_suite argument carries a discriminator string
    (``"iam"`` or ``"iam_role"``) that selects which set of env vars to read.

    Args:
        creds: discriminator passed via ``PROVIDER_KEY_PATH`` —
               ``"iam"`` for direct IAM credentials,
               ``"iam_role"`` for STS assume-role credentials.
    """
    if creds == "iam_role":
        env_keys = _BEDROCK_ROLE_ENV_KEYS
    elif creds == "iam":
        env_keys = _BEDROCK_IAM_ENV_KEYS
    else:
        raise ValueError(f"Unsupported Bedrock credentials mode: {creds}")

    values = {k: os.getenv(k) for k in env_keys}
    missing = [k for k in env_keys if not values[k]]
    if missing:
        raise RuntimeError(
            f"Bedrock IAM secret cannot be created — required env vars "
            f"are unset or empty: {', '.join(missing)}. "
            f"Set these CI/Vault/Prow env vars before running Bedrock e2e tests."
        )

    prefix = "BEDROCK_ROLE_" if creds == "iam_role" else "BEDROCK_"
    data = {
        "aws_access_key_id": values[prefix + "AWS_ACCESS_KEY_ID"],
        "aws_secret_access_key": values[prefix + "AWS_SECRET_ACCESS_KEY"],
    }
    if creds == "iam_role":
        data["role_arn"] = values["BEDROCK_ROLE_ARN"]
    print(f"Ensuring Bedrock IAM secret exists (mode={creds})...")
    cluster_api.create_secret("llmcreds", data)


def _delete_secret(secret_name: str) -> None:
    """Delete a secret, tolerating its absence."""
    cluster_api.delete_resource("secret", secret_name, ignore_missing=True)


def create_secrets(provider_name: str, creds: str, provider_size: int) -> None:
    """Create Kubernetes secrets needed for an LLM provider (API creds).

    Azure Entra ID secret ``azure-entra-id`` is ensured separately when the provider
    list includes ``azure_openai`` (see ``install_ols`` and ``adapt_ols_config``).

    Args:
        provider_name (str): the name of the provider.
        creds (str): string containing credentials for provider
        provider_size (int): size of the provider to create llm creds whenever we have only one

    Returns:
        Nothing.
    """
    if provider_name.startswith("bedrock"):
        _delete_secret("llmcreds")
        ensure_bedrock_iam_secret(creds)
        return
    if creds == "empty":
        _delete_secret("llmcreds")
        cluster_api.create_secret("llmcreds", {"apitoken": creds})
    elif provider_size == 1:
        _delete_secret("llmcreds")
        cluster_api.create_secret(
            "llmcreds", {"apitoken": Path(creds).read_text(encoding="utf-8")}
        )
    else:
        # Only refresh this provider's own secret: other providers in the
        # list own theirs (including llmcreds for bedrock), so deleting a
        # foreign secret here would break a multi-provider deployment.
        secret_name = provider_name.replace("_", "-") + "creds"
        _delete_secret(secret_name)
        cluster_api.create_secret(
            secret_name, {"apitoken": Path(creds).read_text(encoding="utf-8")}
        )


def _run_operator_sdk_bundle(bundle_image: str) -> None:
    """Run ``operator-sdk run bundle`` for the given image."""
    subprocess.run(  # noqa: S603
        [  # noqa: S607
            "operator-sdk",
            "run",
            "bundle",
            "--timeout=20m",
            "-n",
            "openshift-lightspeed",
            bundle_image,
            "--verbose",
        ],
        capture_output=True,
        text=True,
        check=True,
    )


def _cleanup_failed_bundle() -> bool:
    """Clean up a failed operator bundle install so it can be retried.

    Returns True if cleanup succeeded, False otherwise.
    """
    print("Cleaning up failed operator bundle install...")
    try:
        result = subprocess.run(
            [  # noqa: S607
                "operator-sdk",
                "cleanup",
                "lightspeed-operator",
                "--timeout=5m",
                "-n",
                "openshift-lightspeed",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            print(
                f"operator-sdk cleanup failed (rc={result.returncode}),"
                f" stdout: {result.stdout}, stderr: {result.stderr}"
            )
            return False
        print("operator-sdk cleanup succeeded")
        return True
    except OSError as e:
        print(f"operator-sdk cleanup error (non-fatal): {e}")
        return False


def _run_bundle_with_retry(bundle_image: str) -> None:
    """Install the operator bundle, retrying once on failure after cleanup."""
    try:
        _run_operator_sdk_bundle(bundle_image)
    except subprocess.CalledProcessError as first_err:
        print(
            f"operator-sdk run bundle failed: {first_err},"
            f" stdout: {first_err.output}, stderr: {first_err.stderr}"
        )
        if not _cleanup_failed_bundle():
            print("Cleanup failed, skipping retry")
            raise
        time.sleep(30)
        print("Retrying operator-sdk run bundle...")
        try:
            _run_operator_sdk_bundle(bundle_image)
        except subprocess.CalledProcessError as retry_err:
            print(
                f"operator-sdk run bundle retry failed: {retry_err},"
                f" stdout: {retry_err.output}, stderr: {retry_err.stderr}"
            )
            raise


def _rhokp_warmup_marker() -> Path | None:
    """Return the once-per-job RHOKP warm-up marker path, if recordable."""
    artifact_dir = os.getenv("ARTIFACT_DIR")
    if not artifact_dir:
        return None
    return Path(artifact_dir) / ".rhokp_image_warmed"


def warm_up_rhokp_image() -> None:
    """Pull the operator's RHOKP image onto every cluster node.

    Runs at most once per CI job: ARTIFACT_DIR is shared by every suite in
    a job, so a marker file records the warmed image and later suites skip
    the pull (each pull uses a short-lived pod on every node).
    """
    operator_deployment = cluster_api.read_resource(
        "deployment", "lightspeed-operator-controller-manager"
    )
    rhokp_image = next(
        (
            argument.partition("=")[2]
            for container in operator_deployment["spec"]["template"]["spec"][
                "containers"
            ]
            for argument in container.get("args", [])
            if argument.startswith("--rhokp-image=")
        ),
        None,
    )
    if rhokp_image is None:
        raise RuntimeError(
            "Could not find --rhokp-image in the operator controller Deployment"
        )

    marker = _rhokp_warmup_marker()
    if marker is not None and marker.is_file():
        if marker.read_text(encoding="utf-8").strip() == rhokp_image:
            print(f"RHOKP image already warmed in this job: {rhokp_image}")
            return

    nodes = cluster_api.read_resource("node")["items"]
    if not nodes:
        raise RuntimeError("Could not find any cluster nodes to warm up")

    with ThreadPoolExecutor(max_workers=len(nodes)) as executor:
        pulls = []
        for node in nodes:
            node_name = node["metadata"]["name"]
            print(f"Pulling RHOKP image on node {node_name}: {rhokp_image}")
            pulls.append(
                executor.submit(
                    cluster_api.warm_image,
                    node_name,
                    rhokp_image,
                    timeout=600,
                )
            )
        for pull in pulls:
            pull.result()

    if marker is not None:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(rhokp_image, encoding="utf-8")


def install_ols() -> tuple[str, str, str]:  # pylint: disable=R0915, R0912  # noqa: C901
    """Install OLS onto an OCP cluster using the OLS operator."""
    if not disconnected:
        print("Setting up for on cluster test execution")
        bundle_image = os.getenv(
            "BUNDLE_IMAGE",
            "quay.io/openshift-lightspeed/lightspeed-operator-bundle:latest",
        )
        # setup the lightspeed namespace
        cluster_api.create_resource(
            {
                "apiVersion": "v1",
                "kind": "Namespace",
                "metadata": {"name": "openshift-lightspeed"},
            },
            ignore_existing=True,
        )
        print("created OLS project")

        # HyperShift mirrors are provisioned through HostedCluster; guest-side
        # ImageDigestMirrorSet mutations are rejected by admission policy.
        if os.getenv("OLS_EXTERNAL_IMAGE_MIRRORS", "false") != "true":
            cluster_api.apply_file(
                "tests/config/operator_install/imagedigestmirrorset.yaml"
            )

        # install the operator from bundle
        print("Installing OLS operator from bundle")
        _run_bundle_with_retry(bundle_image)
    token, metrics_token = create_and_config_sas()

    # wait for the operator to install
    # time.sleep(3)  # not sure if it is needed but it fails sometimes
    r = retry_until_timeout_or_success(
        120,
        10,
        lambda: any(
            csv.get("status", {}).get("phase") == "Succeeded"
            for csv in cluster_api.read_resource("clusterserviceversion")["items"]
        ),
        "Waiting for OLS operator to install",
    )
    if not r:
        msg = "Timed out waiting for OLS operator to install successfully"
        print(msg)
        raise Exception(msg)
    print("Operator installed successfully")

    provider = os.getenv("PROVIDER", "openai")
    creds = os.getenv("PROVIDER_KEY_PATH", "empty")
    # create the llm api key secret ols will mount
    provider_list = provider.split()
    creds_list = creds.split()
    for i, prov in enumerate(provider_list):
        create_secrets(prov, creds_list[i], len(provider_list))

    if "azure_openai" in provider_list:
        ensure_azure_entra_id_secret()

    warm_up_rhokp_image()

    # create the olsconfig operand
    cluster_api.delete_resource("olsconfig", "cluster", ignore_missing=True)

    try:
        crd_yml_file = "tests/config/operator_install/olsconfig.crd.evaluation.yaml"
        ols_config_suffix = os.getenv("OLS_CONFIG_SUFFIX", "default")

        if len(provider_list) == 1:
            crd_yml_name = f"olsconfig.crd.{provider}"
            if ols_config_suffix != "default":
                crd_yml_name += f"_{ols_config_suffix}"
            crd_yml_file = f"tests/config/operator_install/{crd_yml_name}.yaml"

        print(f"CRD path: {crd_yml_file}.")
        substitute_environment = "rhoai_vllm_lseval" in crd_yml_file
        if substitute_environment:
            print("Running envsubst on CR YAML (rhoai_vllm_lseval template)...")
            if not os.environ.get("KSVC_URL"):
                raise RuntimeError(
                    "KSVC_URL environment variable is not set; "
                    "required for rhoai_vllm_lseval CR template"
                )

        apply_e2e_olsconfig(
            crd_yml_file,
            "create",
            suffix=ols_config_suffix,
            substitute_environment=substitute_environment,
            ignore_existing_resource=False,
        )

    except DynamicApiError as e:
        csv = cluster_api.read_resource("clusterserviceversion")
        print(f"Error creating olsconfig: {e}, csv: {csv}")
        raise
    cluster_api.scale_deployment("lightspeed-operator-controller-manager", 1)
    _wait_for_operator_controller_ready()

    # wait for the ols api server deployment to be created
    r = retry_until_timeout_or_success(
        CLUSTER_RETRY_COUNT,
        CLUSTER_RETRY_DELAY,
        _app_server_deployment_exists,
        "Waiting for OLS API server deployment to be created",
    )
    if not r:
        msg = "Timed out waiting for OLS deployment to be created"
        print(msg)
        _dump_ols_install_debug()
        raise Exception(msg)
    print("OLS deployment created")

    # Ensure ols pod exists so it gets deleted during the scale down to zero, otherwise
    # there may be a race condition.
    retry_until_timeout_or_success(
        CLUSTER_RETRY_COUNT,
        CLUSTER_RETRY_DELAY,
        lambda: cluster_utils.get_pod_by_prefix(fail_not_found=False),
    )

    # get the name of the OLS image from CI so we can substitute it in
    new_ols_image = os.getenv("OLS_IMAGE", "")

    # scale down the operator controller manager to avoid it interfering with the tests
    cluster_api.scale_deployment("lightspeed-operator-controller-manager", 0)

    if new_ols_image != "":
        replace_ols_image(new_ols_image)

    # Scale down server pod. If image is replaced, it won't do anything
    # otherwise, it enables the config modification and subsequent
    # scaling up
    cluster_api.scale_deployment("lightspeed-app-server", 0)
    update_ols_config()
    # scale the ols app server up
    cluster_api.scale_deployment("lightspeed-app-server", 1)
    print("Deployment updated, waiting for new pod to be ready")
    # Wait for the pod to start being created and then wait for it to start running.
    cluster_utils.wait_for_running_pod(wait_http_ready=False)

    print("-" * 50)
    print("OLS pod seems to be ready")
    print("All pods")
    print(yaml.safe_dump(cluster_api.read_resource("pod")))
    print("Running pods")
    print(cluster_utils.get_running_pods())
    pod_name = cluster_utils.get_pod_by_prefix()[0]
    print(f"Found new running OLS pod {pod_name}")
    print("-" * 50)

    # Print the deployment so we can confirm the configuration is what we
    # expect it to be (must-gather will also collect this)
    print(
        yaml.safe_dump(cluster_api.read_resource("deployment", "lightspeed-app-server"))
    )
    print("-" * 50)
    if not disconnected:
        # Configure exporter for e2e tests with proper settings
        try:
            print("Configuring exporter for e2e tests...")
            configure_exporter_for_e2e_tests(
                interval_seconds=3600,  # 1 hour to prevent interference
                ingress_env="stage",
                log_level="debug",
                data_dir="/app-root/ols-user-data",
            )
            print("Exporter configured successfully")
        except Exception as e:
            print(f"Warning: Could not configure exporter: {e}")
            print("Tests may experience interference from data collector")

    cluster_api.delete_resource("route", "ols", ignore_missing=True)
    cluster_api.apply_file("tests/config/operator_install/route.yaml")
    ols_url = cluster_utils.get_ols_url("ols")
    wait_for_ols(ols_url)
    return ols_url, token, metrics_token
