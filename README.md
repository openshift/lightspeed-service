
<!-- PROJECT LOGO -->



<!-- ABOUT THE PROJECT -->
## About The Project




### Built With

This project is built using IBM Watson machine learning.



<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple example steps.

### Prerequisites


* Python 3.11 or newer
* Git and pip installed
* BAM account 
    - BAM_API_KEY
    - BAM_URL

### Installation

_Below is an example of how you can instruct your audience on installing and setting up your app. This template doesn't rely on any external dependencies or services._

1. Get a BAM API Key at [https://bam.res.ibm.com](https://bam.res.ibm.com)
   * Login with your IBM W3 Id credentials.
   * Copy the API Key from the Documentation section.
     ![BAM API Key](docs/bam_api_key.png)
2. BAM URL: https://bam.res.ibm.com
3. Clone the repo
   ```sh
   git clone <project gti>
   ```
4. Install python packages
   ```sh
   pip install -r requirements.txt
   ```
5. Create a new file `.env` from the `default.env` example and enter your BAM_API_KEY in the top line

<!-- USAGE EXAMPLES -->
## Usage

### Local Deployment

#### Run the server
in order to run the API service  
```sh
uvicorn app.main:app --reload --port 8080
```

#### Query the server

To send a request to the server you can use the following curl command:
```sh
curl -X 'POST' 'http://127.0.0.1:8080/ols' -H2 'accept: application/json' -H 'Content-Type: application/json' -d '{"query": "write a deployment yaml for the mongodb image"}'
```

### Gradio UI

There is a minimal Gradio UI you can use when running the OLS server locally.  To use it, first start the OLS server per [Run the server](#run-the-server) and then browse to the built in Gradio interface at http://localhost:8080/ui

By default this interface will ask the OLS server to retain and use your conversation history for subsequent interactions.  To disable this behavior, expand the `Additional Inputs` configuration at the bottom of the page and uncheck the `Use history` checkbox.  When not using history each message you submit to OLS will be treated independently with no context of previous interactions.

### Deploying OLS on OpenShift

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

<!-- ROADMAP -->
## Roadmap

- [ ] 


See the [open issues](https://github.com/othneildrew/Best-README-Template/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

See [contributors](CONTRIBUTING.md) guide.


<!-- LICENSE -->
## License
Published under the Apache 2.0 License
