"""Functions to install the service onto an OCP cluster using the OLS operator."""

import os
import subprocess

import yaml

from ols.constants import DEFAULT_CONFIGURATION_FILE
from tests.e2e.utils import cluster as cluster_utils
from tests.e2e.utils.constants import OLS_COLLECTOR_DISABLING_FILE
from tests.e2e.utils.retry import retry_until_timeout_or_success
from tests.e2e.utils.wait_for_ols import wait_for_ols

OC_COMMAND_RETRY_COUNT = 120
OC_COMMAND_RETRY_DELAY = 5


def create_and_config_sas() -> tuple[str, str]:
    """Create and provide access to service accounts for testing.

    Returns:
        tuple containing token and metrics token.
    """
    cluster_utils.run_oc(
        ["project", "openshift-lightspeed"], ignore_existing_resource=True
    )
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
    cluster_utils.run_oc(
        [
            "create",
            "role",
            "pod-reader",
            "--verb=get,list",
            "--resource=pods",
            "--namespace=openshift-lightspeed",
        ],
        ignore_existing_resource=True,
    )

    cluster_utils.run_oc(
        [
            "create",
            "rolebinding",
            "test-user-pod-reader",
            "--role=pod-reader",
            "--serviceaccount=openshift-lightspeed:test-user",
            "--namespace=openshift-lightspeed",
        ],
        ignore_existing_resource=True,
    )

    print("Granted test-user permission to list pods.")

    return token, metrics_token


def update_ols_config() -> None:
    """Create the ols config configmap with log and collector config for e2e tests.

    Returns:
        Nothing.
    """
    # modify olsconfig configmap
    configmap_yaml = cluster_utils.run_oc(["get", "cm/olsconfig", "-o", "yaml"]).stdout
    configmap = yaml.safe_load(configmap_yaml)
    olsconfig = yaml.safe_load(configmap["data"][DEFAULT_CONFIGURATION_FILE])

    # one of our libs logs a secrets in debug mode which causes the pod
    # logs beying redacted/removed completely - we need log at info level
    olsconfig["ols_config"]["logging_config"]["lib_log_level"] = "INFO"

    # add collector config for e2e tests
    olsconfig["user_data_collector_config"] = {
        "data_storage": "/app-root/ols-user-data",
        "log_level": "debug",
        "collection_interval": 10,
        "run_without_initial_wait": True,
        "ingress_env": "stage",
        "cp_offline_token": os.getenv("CP_OFFLINE_TOKEN", ""),
    }

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
        embeddings_model_path = ""
        if "embeddings_model_path" in old_rag_config:
            embeddings_model_path = old_rag_config["embeddings_model_path"]
        if "product_docs_index_id" in old_rag_config:
            product_docs_index_id = old_rag_config["product_docs_index_id"]
        if "product_docs_index_path" in old_rag_config:
            product_docs_index_path = old_rag_config["product_docs_index_path"]
        olsconfig["ols_config"]["reference_content"] = {
            "embeddings_model_path": embeddings_model_path,
            "indexes": [
                {
                    "product_docs_index_id": product_docs_index_id,
                    "product_docs_index_path": product_docs_index_path,
                }
            ],
        }

    configmap["data"][DEFAULT_CONFIGURATION_FILE] = yaml.dump(olsconfig)
    updated_configmap = yaml.dump(configmap)

    cluster_utils.run_oc(["delete", "configmap", "olsconfig"])
    cluster_utils.run_oc(["apply", "-f", "-"], command=updated_configmap)


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
        OC_COMMAND_RETRY_COUNT,
        OC_COMMAND_RETRY_DELAY,
        lambda: not cluster_utils.get_pod_by_prefix(
            "lightspeed-operator-controller-manager", fail_not_found=False
        ),
        "Waiting for operator controller manager pod to be gone",
    )

    # scale down the ols api server so we can ensure no pods
    # are still running the unsubstituted image
    cluster_utils.run_oc(
        [
            "scale",
            "deployment/lightspeed-app-server",
            "--replicas",
            "0",
        ]
    )

    # wait for the old ols api pod to go away due to deployment being scaled down
    retry_until_timeout_or_success(
        OC_COMMAND_RETRY_COUNT,
        OC_COMMAND_RETRY_DELAY,
        lambda: not cluster_utils.get_pod_by_prefix(fail_not_found=False),
        "Waiting for OLS API pod to be scaled down",
    )

    # update the OLS deployment to use the new image from CI/OLS_IMAGE env var
    patch = f"""[{{"op": "replace", "path": "/spec/template/spec/containers/0/image", "value":"{ols_image}"}}]"""  # noqa: E501
    cluster_utils.run_oc(
        ["patch", "deployment/lightspeed-app-server", "--type", "json", "-p", patch]
    )

    patch = f"""[{{"op": "replace", "path": "/spec/template/spec/containers/1/image", "value":"{ols_image}"}}]"""  # noqa: E501
    cluster_utils.run_oc(
        ["patch", "deployment/lightspeed-app-server", "--type", "json", "-p", patch]
    )


def create_secrets(provider_name: str, creds: str, provider_size: int) -> None:
    """Create secrets for models.

    Args:
        provider_name (str): the name of the provider.
        creds (str): string containing credentials for provider
        provider_size (int): size of the provider to create llm creds whenever we have only one

    Returns:
        Nothing.
    """
    try:
        cluster_utils.run_oc(
            [
                "delete",
                "secret",
                provider_name + "creds",
            ],
        )
    except subprocess.CalledProcessError:
        print("llmcreds secret does not yet exist. Creating it.")
    if provider_size == 1:
        cluster_utils.run_oc(
            [
                "create",
                "secret",
                "generic",
                "llmcreds",
                f"--from-file=apitoken={creds}",
            ],
            ignore_existing_resource=True,
        )
    else:
        cluster_utils.run_oc(
            [
                "create",
                "secret",
                "generic",
                provider_name.replace("_", "-") + "creds",
                f"--from-file=apitoken={creds}",
            ],
            ignore_existing_resource=True,
        )


def install_ols() -> tuple[str, str, str]:  # pylint: disable=R0915, R0912  # noqa: C901
    """Install OLS onto an OCP cluster using the OLS operator."""
    disconnected = os.getenv("DISCONNECTED", "")
    if not disconnected:
        print("Setting up for on cluster test execution")
        bundle_image = os.getenv(
            "BUNDLE_IMAGE",
            "quay.io/openshift-lightspeed/lightspeed-operator-bundle:latest",
        )
        # setup the lightspeed namespace
        cluster_utils.run_oc(
            ["create", "ns", "openshift-lightspeed"], ignore_existing_resource=True
        )
        cluster_utils.run_oc(
            ["project", "openshift-lightspeed"], ignore_existing_resource=True
        )
        print("created OLS project")

        # install the ImageDigestMirrorSet to mirror images
        # from "registry.redhat.io/openshift-lightspeed-beta"
        # to "quay.io/redhat-user-workloads/crt-nshift-lightspeed-tenant/ols"
        cluster_utils.run_oc(
            ["create", "-f", "tests/config/operator_install/imagedigestmirrorset.yaml"],
            ignore_existing_resource=True,
        )

        # install the operator from bundle
        print("Installing OLS operator from bundle")
        cluster_utils.run_oc(
            [
                "apply",
                "-f",
                "tests/config/operator_install/imagedigestmirrorset.yaml",
            ],
            ignore_existing_resource=True,
        )
        try:
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
        # TODO: add run_command func
        except subprocess.CalledProcessError as e:
            print(
                f"Error running operator-sdk: {e}, stdout: {e.output}, stderr: {e.stderr}"
            )
            raise

    token, metrics_token = create_and_config_sas()

    # wait for the operator to install
    # time.sleep(3)  # not sure if it is needed but it fails sometimes
    r = retry_until_timeout_or_success(
        120,
        10,
        lambda: cluster_utils.run_oc(
            [
                "get",
                "clusterserviceversion",
                "-o",
                "jsonpath={.items[0].status.phase}",
            ]
        ).stdout
        == "Succeeded",
        "Waiting for OLS operator to install",
    )
    if not r:
        msg = "Timed out waiting for OLS operator to install successfully"
        print(msg)
        raise Exception(msg)
    print("Operator installed successfully")

    provider = os.getenv("PROVIDER", "openai")
    creds = os.getenv("PROVIDER_KEY_PATH", "")
    # create the llm api key secret ols will mount
    provider_list = provider.split()
    creds_list = creds.split()
    for i, prov in enumerate(provider_list):
        create_secrets(prov, creds_list[i], len(provider_list))

    if provider == "azure_openai":
        # create extra secrets with Entra ID
        cluster_utils.run_oc(
            [
                "create",
                "secret",
                "generic",
                "azure-entra-id",
                f"--from-literal=tenant_id={os.environ['AZUREOPENAI_ENTRA_ID_TENANT_ID']}",
                f"--from-literal=client_id={os.environ['AZUREOPENAI_ENTRA_ID_CLIENT_ID']}",
                f"--from-literal=client_secret={os.environ['AZUREOPENAI_ENTRA_ID_CLIENT_SECRET']}",
            ],
            ignore_existing_resource=True,
        )

    # create the olsconfig operand
    try:
        cluster_utils.run_oc(
            [
                "delete",
                "olsconfig",
                "cluster",
            ],
        )
    except subprocess.CalledProcessError:
        print("olsconfig does not yet exist. Creating it.")

    crd_yml_name = f"olsconfig.crd.{provider}"
    tool_calling_enabled = os.getenv("TOOL_CALLING_ENABLED", "n") == "y"
    try:
        if len(provider_list) == 1:
            if tool_calling_enabled:
                print("Cluster tool_calling is enabled.")
                crd_yml_name += "_tool_calling"
            # OLS-1711: temp solution until operator changes to support mcp_servers
            print("DEBUG: creating introspection CM")
            cluster_utils.run_oc(
                [
                    "create",
                    "-f",
                    f"tests/config/operator_install/{crd_yml_name}.yaml",
                ],
                ignore_existing_resource=True,
            )
            print("DEBUG: introspection CM created")
        else:
            cluster_utils.run_oc(
                [
                    "create",
                    "-f",
                    "tests/config/operator_install/olsconfig.crd.evaluation.yaml",
                ],
                ignore_existing_resource=True,
            )
    except subprocess.CalledProcessError as e:
        csv = cluster_utils.run_oc(
            ["get", "clusterserviceversion", "-o", "jsonpath={.items[0].status}"]
        )
        print(
            f"Error creating olsconfig: {e}, stdout: {e.output}, stderr: {e.stderr}, csv: {csv}"
        )
        raise
    cluster_utils.run_oc(
        [
            "scale",
            "deployment/lightspeed-operator-controller-manager",
            "--replicas",
            "1",
        ]
    )

    # wait for the ols api server deployment to be created
    r = retry_until_timeout_or_success(
        30,
        6,
        lambda: cluster_utils.run_oc(
            [
                "get",
                "deployment",
                "lightspeed-app-server",
                "--ignore-not-found",
                "-o",
                "name",
            ]
        ).stdout
        == "deployment.apps/lightspeed-app-server\n",
        "Waiting for OLS API server deployment to be created",
    )
    if not r:
        msg = "Timed out waiting for OLS deployment to be created"
        print(msg)
        raise Exception(msg)
    print("OLS deployment created")

    # Ensure ols pod exists so it gets deleted during the scale down to zero, otherwise
    # there may be a race condition.
    retry_until_timeout_or_success(
        OC_COMMAND_RETRY_COUNT,
        OC_COMMAND_RETRY_DELAY,
        lambda: cluster_utils.get_pod_by_prefix(fail_not_found=False),
    )

    # get the name of the OLS image from CI so we can substitute it in
    new_ols_image = os.getenv("OLS_IMAGE", "")

    # scale down the operator controller manager to avoid it interfering with the tests
    cluster_utils.run_oc(
        [
            "scale",
            "deployment/lightspeed-operator-controller-manager",
            "--replicas",
            "0",
        ]
    )

    if new_ols_image != "":
        replace_ols_image(new_ols_image)

    # Scale down server pod. If image is replaced, it won't do anything
    # otherwise, it enables the config modification and subsequent
    # scaling up
    cluster_utils.run_oc(
        [
            "scale",
            "deployment/lightspeed-app-server",
            "--replicas",
            "0",
        ]
    )
    update_ols_config()
    # scale the ols app server up
    cluster_utils.run_oc(
        [
            "scale",
            "deployment/lightspeed-app-server",
            "--replicas",
            "1",
        ]
    )
    print("Deployment updated, waiting for new pod to be ready")
    # Wait for the pod to start being created and then wait for it to start running.
    cluster_utils.wait_for_running_pod()

    print("-" * 50)
    print("OLS pod seems to be ready")
    print("All pods")
    print(cluster_utils.run_oc(["get", "pods"]).stdout)
    print("Running pods")
    print(
        cluster_utils.run_oc(
            ["get", "pods", "--field-selector=status.phase=Running"]
        ).stdout
    )
    pod_name = cluster_utils.get_pod_by_prefix()[0]
    print(f"Found new running OLS pod {pod_name}")
    print("-" * 50)

    # Print the deployment so we can confirm the configuration is what we
    # expect it to be (must-gather will also collect this)
    print(
        cluster_utils.run_oc(
            ["get", "deployment", "lightspeed-app-server", "-o", "yaml"]
        ).stdout
    )
    print("-" * 50)

    # disable collector script by default to avoid running during all
    # tests (collecting/sending data)
    cluster_utils.create_file(pod_name, OLS_COLLECTOR_DISABLING_FILE, "")

    try:
        cluster_utils.run_oc(
            [
                "delete",
                "route",
                "ols",
            ],
        )
    except subprocess.CalledProcessError:
        print("No route exists, creating it.")
    # create a route so tests can access OLS directly
    cluster_utils.run_oc(["create", "-f", "tests/config/operator_install/route.yaml"])

    url = cluster_utils.run_oc(
        ["get", "route", "ols", "-o", "jsonpath='{.spec.host}'"]
    ).stdout.strip("'")
    ols_url = f"https://{url}"
    wait_for_ols(ols_url)
    return ols_url, token, metrics_token
