
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
   pip install -r requirement.txt
   ```
4. Create a new file `.env` from the `default.env` example and enter your BAM_API_KEY in the top line

<!-- USAGE EXAMPLES -->
## Usage

### Run the server
in order to run the API service  
   ```sh
        uvicorn ols:app --reload
   ```

### Query the server

To send a request to the server you can use the following curl command:
   ```sh
      curl -X 'POST' 'http://127.0.0.1:8000/ols' -H2 'accept: application/json' -H 'Content-Type: application/json' -d '{"query": "write a deployment yaml for the mongodb image"}'
   ```


<!-- ROADMAP -->
## Roadmap

- [ ] 


See the [open issues](https://github.com/othneildrew/Best-README-Template/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing


<!-- LICENSE -->
## License
Published under the Apache 2.0 License

