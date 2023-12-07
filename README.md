
<!-- PROJECT LOGO -->



<!-- ABOUT THE PROJECT -->
## About The Project




### Built With
this project is built using IBM watson machine learning 



<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple example steps.

### Prerequisites


* python 
* BAM account 
    - BAM_API_KEY
    - BAM_URL

### Installation

_Below is an example of how you can instruct your audience on installing and setting up your app. This template doesn't rely on any external dependencies or services._

1. Get a BAM API Key at [https://ibmmid.com](https://www.ibm.com/account/reg/us-en/signup?formid=urx-19776&target=https%3A%2F%2Flogin.ibm.com%2Foidc%2Fendpoint%2Fdefault%2Fauthorize%3FqsId%3D5b380573-b65b-469c-b3d3-e60c31c89011%26client_id%3DMyIBMDallasProdCI)
2. Clone the repo
   ```sh
   git clone <project gti>
   ```
3. Install python packages
   ```sh
   pip install -r requirements.txt
   ```
4. Create a new file `.env` from the `default.env` example and enter your BAM_API_KEY in the top line

<!-- USAGE EXAMPLES -->
## Usage

### Local Deployment

#### Run the server
in order to run the API service  
```sh
uvicorn app.main:ols --reload
```

#### Query the server

To send a request to the server you can use the following curl command:
```sh
curl -X 'POST' 'http://127.0.0.1:8000/ols' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"query": "write a deployment yaml for the mongodb image"}'
```

### Gradio UI

There is a minimal Gradio UI you can use when running the OLS server locally.  To use it, first start the OLS server per [Run the server](#run-the-server) and then browse to the built in gradio interface at http://localhost:8080/ui

By default this interface will ask the OLS server to retain and use your conversation history for subsequent interactions.  To disable this behavior, expand the `Additional Inputs` configuration at the bottom of the page and uncheck the `Use history` checkbox.  When not using history each message you submit to OLS will be treated independently with no context of previous interactions.

### In Cluster Deployment
Deploying OLS on an openshift cluster is fairly easy with the configuration files we have in [manifests](./manifests) folder.

You can use the existing image built from the latest code via this image pullspec: quay.io/openshift/lightspeed-service-api:latest

If you need to build your own image, you can use the following commands:

```
podman build -f Containerfile -t=<your-image-pullspec> .
podman push <your-image-pullspec>
```

Once we have our image ready, export it as an ENV and use the below [kustomize](https://kustomize.io/) command to deploy resources.
```
export OLS_IMAGE=<image-pullspec>
kustomize build . | envsubst | oc apply -f -
``` 
This should deploy ols fronting with a [kube-rbac-proxy](https://github.com/brancz/kube-rbac-proxy) along with a sample [client](./config/ols-client-test.yaml) that makes requests to one of the ols endpoints demonstrating client usage of our service.

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
