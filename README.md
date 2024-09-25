# About The Project

OpenShift LightSpeed (OLS) is an AI powered assistant that runs on OpenShift
and provides answers to product questions using backend LLM services. Currently
[OpenAI](https://openai.com/), [Azure
OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service),
[OpenShift
AI](https://www.redhat.com/en/technologies/cloud-computing/openshift/openshift-ai),
[RHEL
AI](https://www.redhat.com/en/technologies/linux-platforms/enterprise-linux/ai),
and [Watsonx](https://www.ibm.com/watsonx) are officially supported as
backends. Other providers, even ones that are not fully supported, can be used
as well. For example, it is possible to use BAM (IBM's research environment).
It is also possible to run [InstructLab](https://instructlab.ai/) locally,
configure model, and connect to it.


<!-- the following line is used by tool to autogenerate Table of Content when the document is changed -->
<!-- vim-markdown-toc GFM -->

* [Prerequisites](#prerequisites)
* [Installation](#installation)
    * [1. Clone the repo](#1-clone-the-repo)
    * [2. Install python packages](#2-install-python-packages)
    * [3. Get API keys](#3-get-api-keys)
        * [OpenAI](#openai)
        * [Azure OpenAI](#azure-openai)
        * [WatsonX](#watsonx)
        * [OpenShift AI](#openshift-ai)
        * [RHEL AI](#rhel-ai)
        * [BAM (not officially supported)](#bam-not-officially-supported)
        * [Locally running InstructLab](#locally-running-instructlab)
    * [4. Store local copies of API keys securely](#4-store-local-copies-of-api-keys-securely)
    * [5. Configure OpenShift LightSpeed (OLS)](#5-configure-openshift-lightspeed-ols)
    * [6. Configure LLM providers](#6-configure-llm-providers)
        * [OpenAI provider](#openai-provider)
        * [Azure OpenAI](#azure-openai-1)
        * [WatsonX](#watsonx-1)
        * [RHEL AI provider](#rhel-ai-provider)
        * [Red Hat OpenShift AI](#red-hat-openshift-ai)
        * [Local *ollama* server](#local-ollama-server)
    * [7. Configure OLS Authentication](#7-configure-ols-authentication)
    * [8. Configure OLS TLS communication](#8-configure-ols-tls-communication)
    * [9. (Optional) Configure the local document store](#9-optional-configure-the-local-document-store)
    * [10. (Optional) Configure conversation cache](#10-optional-configure-conversation-cache)
    * [11. (Optional) Incorporating additional CA(s). You have the option to include an extra TLS certificate into the OLS trust store as follows.](#11-optional-incorporating-additional-cas-you-have-the-option-to-include-an-extra-tls-certificate-into-the-ols-trust-store-as-follows)
    * [12. Registering new LLM provider](#12-registering-new-llm-provider)
* [Usage](#usage)
    * [Deployments](#deployments)
        * [Local Deployment](#local-deployment)
            * [Run the server](#run-the-server)
        * [Optionally run with podman](#optionally-run-with-podman)
        * [Optionally run inside an OpenShift environment](#optionally-run-inside-an-openshift-environment)
    * [Communication with the service](#communication-with-the-service)
        * [Query the server](#query-the-server)
        * [Swagger UI](#swagger-ui)
        * [OpenAPI](#openapi)
        * [Metrics](#metrics)
        * [Gradio UI](#gradio-ui)
        * [Swagger UI](#swagger-ui-1)
    * [Deploying OLS on OpenShift](#deploying-ols-on-openshift)
* [Project structure](#project-structure)
    * [Overall architecture](#overall-architecture)
    * [Sequence diagram](#sequence-diagram)
    * [Token truncation algorithm](#token-truncation-algorithm)
* [Contributing](#contributing)
* [License](#license)

<!-- the following line is used by tool to autogenerate Table of Content when the document is changed -->
<!-- vim-markdown-toc -->

# Prerequisites

* Python 3.11
    - please note that currently Python 3.12 is not officially supported, because OLS LightSpeed depends on some packages that can not be used in this Python version
* Git, pip and [PDM](https://github.com/pdm-project/pdm?tab=readme-ov-file#installation)
* An LLM API key or API secret (in case of Azure OpenAI)
* (Optional) extra certificates to access LLM API

# Installation

## 1. Clone the repo
   ```sh
   git clone https://github.com/openshift/lightspeed-service.git
   cd lightspeed-service
   ```
## 2. Install python packages
   ```sh
   make install-deps
   ```
## 3. Get API keys

   This step depends on provider type

### OpenAI

Please look into ([OpenAI api key](https://platform.openai.com/api-keys))

### Azure OpenAI

Please look at following articles describing how to retrieve API key or secret from Azure: [Get subscription and tenant IDs in the Azure portal](https://learn.microsoft.com/en-us/azure/azure-portal/get-subscription-tenant-id) and [How to get client id and client secret in Azure Portal](https://azurelessons.com/how-to-get-client-id-and-client-secret-in-azure-portal/). Currently it is possible to use both ways to auth. to Azure OpenAI: by API key or by using secret

### WatsonX

Please look at into [Generating API keys for authentication](https://www.ibm.com/docs/en/watsonx/watsonxdata/1.0.x?topic=started-generating-api-keys)

### OpenShift AI

(TODO: to be updated)

### RHEL AI

(TODO: to be updated)

### BAM (not officially supported)
    1. Get a BAM API Key at [https://bam.res.ibm.com](https://bam.res.ibm.com)
        * Login with your IBM W3 Id credentials.
        * Copy the API Key from the Documentation section.
        ![BAM API Key](docs/bam_api_key.png)
    2. BAM API URL: https://bam-api.res.ibm.com

### Locally running InstructLab

Depends on configuration, but usually it is not needed to generate or use API key.


## 4. Store local copies of API keys securely

   Here is a proposed scheme for storing API keys on your development workstation. It is similar to how private keys are stored for OpenSSH.
   It keeps copies of files containing API keys from getting scattered around and forgotten:

   ```
   $ cd <lightspeed-service local git repo root>
   $ find ~/.openai -ls
   72906922      0 drwx------   1 username username        6 Feb  6 16:45 /home/username/.openai
   72906953      4 -rw-------   1 username username       52 Feb  6 16:45 /home/username/.openai/key
   $ ls -l openai_api_key.txt
   lrwxrwxrwx. 1 username username 26 Feb  6 17:41 openai_api_key.txt -> /home/username/.openai/key
   $ grep openai_api_key.txt olsconfig.yaml
    credentials_path: openai_api_key.txt
   ```

## 5. Configure OpenShift LightSpeed (OLS)

   OLS configuration is in YAML format. It is loaded from a file referred to by the `OLS_CONFIG_FILE` environment variable and defaults to `olsconfig.yaml` in the current directory.
   You can find a example configuration in the [examples/olsconfig.yaml](examples/olsconfig.yaml) file in this repository.

## 6. Configure LLM providers

   The example configuration file defines providers for six LLM providers: BAM, OpenAI, Azure OpenAI, Watsonx, OpenShift AI VLLM (RHOAI VLLM), and RHELAI (RHEL AI), but defines BAM as the default provider. If you prefer to use a different LLM provider than BAM, such as OpenAI, ensure that the provider definition points to a file containing a valid OpenAI, Watsonx etc. API key, and change the `default_model` and `default_provider` values to reference the selected provider and model.

   The example configuration also defines locally running provider InstructLab which is OpenAI-compatible and can use
   several models. Please look at [instructlab pages](https://github.com/instructlab/instructlab/tree/main) for detailed
   information on how to set up and run this provider.

   API credentials are in turn loaded from files specified in the config YAML by the `credentials_path` attributes. If these paths are relative,
   they are relative to the current working directory. To use the example olsconfig.yaml as is, place your BAM API Key into a file named `bam_api_key.txt` in your working directory.

   Note: there are two supported methods to provide credentials for Azure OpenAI. The first method is compatible with other providers, i.e. `credentials_path` contains a directory name containing one file with API token. In the second method, that directory should contain three files named `tenant_id`, `client_id`, and `client_secret`. Please look at following articles describing how to retrieve this information from Azure: [Get subscription and tenant IDs in the Azure portal](https://learn.microsoft.com/en-us/azure/azure-portal/get-subscription-tenant-id) and [How to get client id and client secret in Azure Portal](https://azurelessons.com/how-to-get-client-id-and-client-secret-in-azure-portal/).

### OpenAI provider

   Multiple models can be configured, but `default_model` will be used, unless specified differently via REST API request:


  ```yaml
    type: openai
    url: "https://api.openai.com/v1"
    credentials_path: openai_api_key.txt
    models:
      - name: gpt-4-1106-preview
      - name: gpt-3.5-turbo
  ```

### Azure OpenAI

   Make sure the `url` and `deployment_name` are set correctly.

  ```yaml
  - name: my_azure_openai
    type: azure_openai
    url: "https://myendpoint.openai.azure.com/"
    credentials_path: azure_openai_api_key.txt
    deployment_name: my_azure_openai_deployment_name
    models:
      - name: gpt-3.5-turbo
  ```

### WatsonX

   Make sure the `project_id` is set up correctly.

  ```yaml
  - name: my_watsonx
    type: watsonx
    url: "https://us-south.ml.cloud.ibm.com"
    credentials_path: watsonx_api_key.txt
    project_id: XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX
    models:
      - name: ibm/granite-13b-chat-v2
  ```

### RHEL AI provider

   It is possible to use RHELAI as a provider too. That provider is OpenAI-compatible
   and can be configured the same way as other OpenAI providers. For example if
   RHEL AI is running as EC2 instance and `granite-7b-lab` model is deployed, the
   configuration might look like:

  ```yaml
      - name: my_rhelai
        type: openai
        url: "http://{PATH}.amazonaws.com:8000/v1/"
        credentials_path: openai_api_key.txt
        models:
          - name: granite-7b-lab
  ```

### Red Hat OpenShift AI

   To use RHOAI (Red Hat OpenShiftAI) as provider, the following
   configuration can be utilized (`mistral-7b-instruct` model is supported by
   RHOAI, as well as other models):

  ```yaml
      - name: my_rhoai
        type: openai
        url: "http://{PATH}:8000/v1/"
        credentials_path: openai_api_key.txt
        models:
          - name: mistral-7b-instruct
  ```

### Local *ollama* server

   It is possible to configure the service to use local *ollama* server.
   Please look into an
   [examples/olsconfig-local-ollama.yaml](examples/olsconfig-local-ollama.yaml)
   file that describes all required steps.

   1. Common providers configuration options

       - `name`: unique name, can be any proper YAML literal
       - `type`: provider type: any of `bam`, `openai`, `azure_openai`, `rhoai_vllm`, `rhelai_vllm`, or `watsonx`
       - `url`: URL to be used to call LLM via REST API
       - `api_key`: path to secret (token) used to call LLM via REST API
       - `models`: list of models configuration (model name + model-specific parameters)

            Notes: 
            - `Context window size` varies based on provider/model.
            - `Max response tokens` depends on user need and should be in reasonable proportion to context window size. If value is too less then there is a risk of response truncation. If we set it too high then we will reserve too much for response & truncate history/rag context unnecessarily.
            - These are optional setting, if not set; then default will be used (which may be incorrect and may cause truncation & potentially error by exceeding context window).

   2. Specific configuration options for WatsonX

       - `project_id`: as specified on WatsonX AI page

   3. Specific configuration options for Azure OpenAI

       - `deployment_name`: as specified in AzureAI project settings

   4. Default provider and default model
       - one provider and its model needs to be selected as default. When no
         provider+model is specified in REST API calls, the default provider and model are used:

         ```yaml
            ols_config:
              default_provider: my_bam
              default_model: ibm/granite-13b-chat-v2
         ```


## 7. Configure OLS Authentication

   NOTE: Currently, only K8S-based authentication can be used. In future versions, more authentication mechanisms will be configurable.

   This section provides guidance on how to configure authentication within OLS. It includes instructions on enabling or disabling authentication, configuring authentication through OCP RBAC, overriding authentication configurations, and specifying a static authentication token in development environments.

   1. Enabling and Disabling Authentication
   
      Authentication is enabled by default in OLS. To disable authentication, modify the `dev_config` in your configuration file as shown below:

      ```yaml
         dev_config:
            disable_auth: true
      ```

   2. Configuring Authentication with OCP RBAC

      OLS utilizes OCP RBAC for authentication, necessitating connectivity to an OCP cluster. It automatically selects the configuration from the first available source, either an in-cluster configuration or a KubeConfig file.

   3. Overriding Authentication Configuration

      You can customize the authentication configuration by overriding the default settings. The configurable options include:

      - **Kubernetes Cluster API URL (`k8s_cluster_api`):** The URL of the K8S/OCP API server where tokens are validated.
      - **CA Certificate Path (`k8s_ca_cert_path`):** Path to a CA certificate for clusters with self-signed certificates.
      - **Skip TLS Verification (`skip_tls_verification`):** If true, the Kubernetes client skips TLS certificate validation for the OCP cluster.

      To apply any of these overrides, update your configuration file as follows:

      ```yaml
         ols_config:
            authentication_config:
               k8s_cluster_api: "https://api.example.com:6443"
               k8s_ca_cert_path: "/Users/home/ca.crt"
               skip_tls_verification: false
      ```

   4. Providing a Static Authentication Token in Development Environments

      For development environments, you may wish to use a static token for authentication purposes. This can be configured in the `dev_config` section of your configuration file:

      ```yaml
         dev_config:
            k8s_auth_token: your-user-token
      ```
      **Note:** using static token will require you to set the `k8s_cluster_api` mentioned in section 6.4, as this will disable the loading of OCP config from in-cluster/kubeconfig.

## 8. Configure OLS TLS communication

   This section provides instructions on configuring TLS (Transport Layer Security) for the OLS Application, enabling secure connections via HTTPS. TLS is enabled by default; however, if necessary, it can be disabled through the `dev_config` settings.


   1. Enabling and Disabling TLS
   
      By default, TLS is enabled in OLS. To disable TLS, adjust the `dev_config` in your configuration file as shown below:

      ```yaml
         dev_config:
            disable_tls: false
      ```

   2. Configuring TLS in local Environments:

      1. Generate Self-Signed Certificates: To generate self-signed certificates, run the following command from the project's root directory:
         ```bash
            ./scripts/generate-certs.sh
         ``` 
      2. Update OLS Configuration: Modify your config.yaml to include paths to your certificate and its private key:
         ```yaml
            ols_config:
               tls_config:
                  tls_certificate_path: /full/path/to/certs/cert.pem
                  tls_key_path: /full/path/to/certs/key.pem
         ```
      3. Launch OLS with HTTPS: After applying the above configurations, OLS will run over HTTPS.
   
   3. Configuring OLS in OpenShift:

      For deploying in OpenShift, Service-Served Certificates can be utilized. Update your ols-config.yaml as shown below, based on the example provided in the examples directory:

      ```yaml
         ols_config:
            tls_config:
               tls_certificate_path: /app-root/certs/cert.pem
               tls_key_path: /app-root/certs/key.pem
      ```
   4. Using a Private Key with a Password
      If your private key is encrypted with a password, specify a path to a file that contains the key password as follows:
      ```yaml
         ols_config:
            tls_config:
               tls_key_password_path: /app-root/certs/password.txt
      ```

## 9. (Optional) Configure the local document store
   ```sh
   make get-rag
   ```

## 10. (Optional) Configure conversation cache
   Conversation cache can be stored in memory (it's content will be lost after shutdown) or in PostgreSQL database. It is possible to specify storage type in `olsconfig.yaml` configuration file.
   
   1. Cache stored in memory:
         ```yaml
         ols_config:
            conversation_cache:
               type: memory
               memory:
               max_entries: 1000
         ```
   2. Cache stored in PostgreSQL:
         ```yaml
         conversation_cache:
            type: postgres
            postgres:
               host: "foobar.com"
               port: "1234"
               dbname: "test"
               user: "user"
               password_path: postgres_password.txt
               ca_cert_path: postgres_cert.crt
               ssl_mode: "require"
         ```
         In this case, file `postgres_password.txt` contains password required to connect to PostgreSQL. Also CA certificate can be specified using `postgres_ca_cert.crt` to verify trusted TLS connection with the server. All these files needs to be accessible. 

## 11. (Optional) Incorporating additional CA(s). You have the option to include an extra TLS certificate into the OLS trust store as follows.
```yaml
      ols_config:
         extra_ca:
            - "path/to/cert_1.crt"
            - "path/to/cert_2.crt"
```

 > This action may be required for self-hosted LLMs.

## 12. Registering new LLM provider
   Please look [here](https://github.com/openshift/lightspeed-service/blob/main/CONTRIBUTING.md#adding-a-new-providermodel) for more info.

# Usage

## Deployments

### Local Deployment

OLS service can be started locally. In this case GradIO web UI is used to
interact with the service. Alternatively the service can be accessed through
REST API.

#### Run the server

If Python virtual environment is setup already, it is possible to start the service by following command:

```sh
make run
```

It is also possible to initialize virtual environment and start the service by using just one command:

```sh
pdm start
```


### Optionally run with podman
There is an all-in-one image that has the document store included already.

1. Follow steps above to create your config yaml and your API key file(s). 
1. Place your config yaml and your API key file(s) in a known location (eg:
`/path/to/config`)
1. Make sure your config yaml references the config folder for the path to your
key file(s) (eg: `credentials_path: config/openai_api_key.txt`)
1. Run the all-in-one-container. Example invocation:

   ```sh
    podman run -it --rm -v `/path/to/config:/app-root/config:Z \
    -e OLS_CONFIG_FILE=/app-root/config/olsconfig.yaml -p 8080:8080 \
    quay.io/openshift-lightspeed/lightspeed-service-api:latest
    ```

### Optionally run inside an OpenShift environment
In the `examples` folder is a set of YAML manifests,
`openshift-lightspeed.yaml`. This includes all the resources necessary to get
OpenShift Lightspeed running in a cluster. It is configured expecting to only
use OpenAI as the inference endpoint, but you can easily modify these manifests,
looking at the `olsconfig.yaml` to see how to alter it to work with BAM as the
provider.

There is a commented-out OpenShift Route with TLS Edge termination available if
you wish to use it.

To deploy, assuming you already have an OpenShift environment to target and that
you are logged in with sufficient permissions:

1. Make the change to your API keys and/or provider configuration in the
manifest file
2. Create a namespace/project to hold OLS
3. `oc apply -f examples/openshift-lightspeed-tls.yaml -n created-namespace`

Once deployed, it is probably easiest to `oc port-forward` into the pod where
OLS is running so that you can access it from your local machine.


## Communication with the service

### Query the server

To send a request to the server you can use the following curl command:
```sh
curl -X 'POST' 'http://127.0.0.1:8080/v1/query' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"query": "write a deployment yaml for the mongodb image"}'
```

### Swagger UI

Web page with Swagger UI has the standard `/docs` endpoint. If the service is running on localhost on port 8080, Swagger UI can be accessed on address `http://localhost:8080/docs`.

### OpenAPI

OpenAPI schema is available [docs/openapi.json](docs/openapi.json). It is possible to re-generate the document with schema by using:

```
make schema
```

When the OLS service is started OpenAPI schema is available on `/openapi.json` endpoint. For example, for service running on localhost on port 8080, it can be accessed and pretty printed by using following command:

```sh
curl 'http://127.0.0.1:8080/openapi.json' | jq .
```


### Metrics

Service exposes metrics in Prometheus format on `/metrics` endpoint. Scraping them is straightforward:

```sh
curl 'http://127.0.0.1:8080/metrics'
```

### Gradio UI

There is a minimal Gradio UI you can use when running the OLS server locally.  To use it, it is needed to enable UI in `olsconfig.yaml` file:

```yaml
dev_config:
  enable_dev_ui: true
```

Then start the OLS server per [Run the server](#run-the-server) and then browse to the built in Gradio interface at http://localhost:8080/ui

By default this interface will ask the OLS server to retain and use your conversation history for subsequent interactions.  To disable this behavior, expand the `Additional Inputs` configuration at the bottom of the page and uncheck the `Use history` checkbox.  When not using history each message you submit to OLS will be treated independently with no context of previous interactions.

###  Swagger UI

OLS API documentation is available at http://localhost:8080/docs


## Deploying OLS on OpenShift

A Helm chart is available for installing the service in OpenShift.

Before installing the chart, you must configure the `auth.key` parameter in the [Values](helm/values.yaml) file

To install the chart with the release name `ols-release` in the namespace `openshift-lightspeed`:

```shell
helm upgrade --install ols-release helm/ --create-namespace --namespace openshift-lightspeed
```

The command deploys the service in the default configuration.

The default configuration contains OLS fronting with a [kube-rbac-proxy](https://github.com/brancz/kube-rbac-proxy).

To uninstall/delete the chart with the release name `ols-release`:

```shell
helm delete ols-release --namespace openshift-lightspeed
```

Chart customization is available using the [Values](helm/values.yaml) file.


# Project structure

1. REST API handlers
1. Configuration loader
1. LLM providers registry
1. LLM loader
1. Interface to LLM providers
1. Doc retriever from vector storage
1. Question validator
1. Docs summarizer
1. Conversation cache
1. (Local) Web-based user interface



## Overall architecture

Overall architecture with all main parts is displayed below:

![Architecture diagram](docs/architecture_diagram.png)

OpenShift LightSpeed service is based on FastAPI framework (Uvicorn) with Langchain for LLM interactions. The service is split into several parts described below.

### FastAPI server

Handles REST API requests from clients (mainly from UI console, but can be any REST API-compatible tool), handles requests queue, and also exports Prometheus metrics. Uvicorn framework is used as FastAPI implementation.

### Authorization checker

Manages authentication flow for REST API endpoints. Currently K8S/OCL-based authorization is used, but in the future it will be implemented in a more modular way to allow registering other auth. checkers.

### Query handler

Retrieves user queries, validates them, redacts them, calls LLM, and summarizes feedback.

### Redactor

Redacts the question based on the regex filters provided in the configuration file.

### Question validator

Validates question and provides one-word responses. It is optional component.

### Document summarizer

Summarizes documentation context.

### Conversation history cache interface

Unified interface used to store and retrieve conversation history with optionally defined maximum length.

### Conversation history cache implementations

Currently there exist three conversation history cache implementations:
1. in-memory cache
1. Redis cache
1. Postgres cache

Entries stored in cache have compound keys that consist of `user_id` and `conversation_id`. It is possible for one user to have multiple conversations and thus multiple `conversation_id` values at the same time. Global cache capacity can be specified. The capacity is measured as number of entries; entries size are ignored in this computation.

#### In-memory cache

In-memory cache is implemented as a queue with defined maximum capacity specified as number of entries that can be stored in a cache. That number is limit for all cache entries, not matter how many users are using LLM. When new entry is put into the cache and if the maximum capacity is reached, oldest entry is removed from the cache.

#### Redis cache

Entries are stored in Redis as dictionary. LRU policy can be specified that allows Redis to automatically remove oldest entries.

#### Postgres cache

Entries are stored in one Postgres table with following schema:

```
     Column      |            Type             | Nullable | Default | Storage  |
-----------------+-----------------------------+----------+---------+----------+
 user_id         | text                        | not null |         | extended |
 conversation_id | text                        | not null |         | extended |
 value           | bytea                       |          |         | extended |
 updated_at      | timestamp without time zone |          |         | plain    |
Indexes:
    "cache_pkey" PRIMARY KEY, btree (user_id, conversation_id)
    "cache_key_key" UNIQUE CONSTRAINT, btree (key)
    "timestamps" btree (updated_at)
Access method: heap
```

During new record insertion the maximum number of entries is checked and when the defined capacity is reached, oldest entry is deleted.



### LLM providers registry

Manages LLM providers implementations. If a new LLM provider type needs to be added, it is registered by this machinery and its libraries are loaded to be used later.

### LLM providers interface implementations

Currently there exist the following LLM providers implementations:
1. OpenAI
1. Azure OpenAI
1. RHEL AI
1. OpenShift AI
1. WatsonX
1. BAM
1. Fake provider (to be used by tests and benchmarks)


## Sequence diagram

Sequence of operations performed when user asks a question:

![Sequence diagram](docs/sequence_diagram.png)



## Token truncation algorithm

The context window size is limited for all supported LLMs which means that token truncation algorithm needs to be performed for longer queries, queries with long conversation history etc. Current truncation logic/context window token check:

1. Tokens for current prompt system instruction + user query + attachment (if any) + tokens reserved for response (default 512) should not be greater than model context window size, otherwise OLS will raise an error.
1. Let’s say above tokens count as default tokens that will be used all the time. If any token is left after default usage then RAG context will be used completely or truncated depending upon how much tokens are left.
1. Finally if we have further available tokens after using complete RAG context, then history will be used (or will be truncated)
1. There is a flag set to True by the service, if history is truncated due to tokens limitation.

![Token truncation](docs/token_truncation.png)




# Contributing

* See [contributors](CONTRIBUTING.md) guide.

* See the [open issues](https://github.com/openshift/lightspeed-service/issues) for a full list of proposed features (and known issues).

# License
Published under the Apache 2.0 License

