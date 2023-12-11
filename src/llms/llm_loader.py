# workaround to disable UserWarning
import warnings
warnings.simplefilter("ignore", UserWarning)

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager

import os, inspect

## append path of current caller when __main__
if __name__ == '__main__':
    import sys 
    sys.path.insert(0,'')

from utils.logger import Logger

class LLMLoader:
    """
    Note: This class loads the LLM backend librearies if the specific LLM is loaded.
    Known caveats: Currently supports a single instance/model per backend
    """
    def __init__(self, llm_backends=set(),
                 inference_url=None,
                 prompt_type=None,
                 api_key=None,
                 model=None, logger=None) -> None:
        self.logger = logger if logger is not None else Logger("llm_loader").logger
        # a set of backend LLMs to activate
        self.llm_backends = set(os.environ.get('LLM_BACKENDS', {'ollama'})) if len(llm_backends) == 0 else llm_backends
        # defalt LLM backend to use or use the first index from loaded ones
        self.llm_default = os.environ.get('LLM_DEFAULT', list(self.llm_backends)[0])
        self.llm = {}
        self._set_llm_instance()

    def _set_llm_instance(self, inference_url=None, api_key=None, model=None):
        self.logger.debug(f"[{inspect.stack()[0][3]}] Loading LLM instances with default {self.llm_default}")

        for backend in self.llm_backends:
            self.logger.debug(f"Loading backend LLM {backend}")
            match backend:
                case 'openai':
                    self._openai_llm_instance()
                case 'ollama':
                    self._ollama_llm_instance()
                case 'tgi':
                    self._tgi_llm_instance()
                case 'watson':
                    self._watson_llm_instance()
                case 'bam':
                    self._bam_llm_instance()
                case _:
                    self.logger.warning(f"WARNING: Unsupported LLM {backend}")

    def _openai_llm_instance(self):
        self.logger.debug(
            f"[{inspect.stack()[0][3]}] Creating OpenAI LLM instance")
        try:
            import openai
            from langchain.llms import OpenAI 
        except e:
            self.logger.error(f"ERROR: Missing openai libraries. Skipping loading backend LLM.")
            return
        params = {
            'base_url': os.environ.get('OPENAI_API_URL', 'https://api.openai.com/v1'),
            'api_key': os.environ.get('OPENAI_API_KEY', None),
            'model': os.environ.get('OPENAI_MODEL', None),
            'model_kwargs': {}, # TODO: add model args
            'organization': os.environ.get('OPENAI_ORGANIZATION', None),
            'timeout': os.environ.get('OPENAI_TIMEOUT', None),
            'cache': None,
            'streaming': True,            
            'temperature': 0.01,
            'max_tokens': 512,
            'top_p': 0.95,
            'frequency_penalty': 1.03,
            'verbose': False
        }
        self.llm['openai']=OpenAI(**params)
        self.logger.debug(
            f"[{inspect.stack()[0][3]}] OpenAI LLM instance {self.llm['openai']}")

    def _ollama_llm_instance(self):
        self.logger.debug(
            f"[{inspect.stack()[0][3]}] Creating Ollama LLM instance")
        try:
            from langchain.llms import Ollama 
        except e:
            self.logger.error(f"ERROR: Missing ollama libraries. Skipping loading backend LLM.")
            return
        params = {
            'base_url': os.environ.get('OLLAMA_API_URL', "http://127.0.0.1:11434"),
            'model': os.environ.get('OLLAMA_MODEL', 'Mistral'),
            'cache': None,
            'temperature': 0.01,
            'top_k': 10,
            'top_p': 0.95,
            'repeat_penalty': 1.03,
            'verbose': False,
            'callback_manager': CallbackManager([StreamingStdOutCallbackHandler()])
        }
        self.llm['ollama'] = Ollama(**params)
        self.logger.debug(
            f"[{inspect.stack()[0][3]}] Ollama LLM instance {self.llm}")

    def _tgi_llm_instance(self):
        """
        Note: TGI does not support specifying the model, it is an instance per model.
        """
        self.logger.debug(
            f"[{inspect.stack()[0][3]}] Creating Hugging Face TGI LLM instance")
        try:
            from langchain.llms import HuggingFaceTextGenInference
        except e:
            self.logger.error(f"ERROR: Missing HuggingFaceTextGenInference libraries. Skipping loading backend LLM.")
            return
        params = {
            'inference_server_url': os.environ.get('TGI_API_URL', None),
            'model_kwargs': {}, # TODO: add model args
            'max_new_tokens': 512,
            'cache': None,
            'temperature': 0.01,
            'top_k': 10,
            'top_p': 0.95,
            'repetition_penalty': 1.03,
            'streaming': True,
            'verbose': False,
            'callback_manager': CallbackManager([StreamingStdOutCallbackHandler()])
        }        
        self.llm['tgi'] = HuggingFaceTextGenInference(**params)
        self.logger.debug(
            f"[{inspect.stack()[0][3]}] Hugging Face TGI LLM instance {self.llm}")

    def _bam_llm_instance(self):
        """BAM Research Lab"""
        self.logger.debug(f"[{inspect.stack()[0][3]}] BAM LLM instance")
        try:
            # BAM Research lab
            from genai.extensions.langchain import LangChainInterface
            from genai.credentials import Credentials
            from genai.model import Model
            from genai.schemas import GenerateParams
        except e:
            self.logger.error(f"ERROR: Missing ibm-generative-ai libraries. Skipping loading backend LLM.")
            return
        # BAM Research lab
        creds = Credentials(
                        api_key=os.environ.get('BAM_API_KEY', None),
                        api_endpoint=os.environ.get('BAM_API_URL', 'https://bam-api.res.ibm.com')
                        )
        params = GenerateParams(decoding_method="sample",
                                max_new_tokens=512,
                                min_new_tokens=1,
                                random_seed=42,
                                top_k=10,
                                top_p=0.95,
                                repetition_penalty=1.03,
                                temperature=0.05)
        self.llm['bam'] = LangChainInterface(
                                        model=os.environ.get('BAM_MODEL', None),
                                        params=params,
                                        credentials=creds
                                        )
        self.logger.debug(
            f"[{inspect.stack()[0][3]}] BAM LLM instance {self.llm}")

    def _watson_llm_instance(self):
        self.logger.debug(f"[{inspect.stack()[0][3]}] Watson LLM instance")
        # watsonX (requires WansonX libraries)
        try:
            from ibm_watson_machine_learning.foundation_models import Model
            from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
            from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
        except e:
            self.logger.error(f"ERROR: Missing ibm_watson_machine_learning libraries. Skipping loading backend LLM.")
            return
        #
        creds = {
            # example from https://heidloff.net/article/watsonx-langchain/
            "url": os.environ.get('WATSON_API_URL', None),
            "apikey": os.environ.get('WATSON_API_KEY', None)
        }
        params = {
            GenParams.DECODING_METHOD: "sample",
            GenParams.MIN_NEW_TOKENS: 1,
            GenParams.MAX_NEW_TOKENS: 512,
            GenParams.RANDOM_SEED: 42,
            GenParams.TEMPERATURE: 0.05,
            GenParams.TOP_K: 10,
            GenParams.TOP_P: 0.95,
            # https://www.ibm.com/docs/en/watsonx-as-a-service?topic=models-parameters
            GenParams.REPETITION_PENALTY: 1.03
        }
        llm_model = Model(model_id=os.environ.get('WATSON_MODEL', None),
                          credentials=creds,
                          params=params,
                          project_id=os.environ.get('WATSON_PROJECT_ID', None)
                          )
        self.llm['watson'] = WatsonxLLM(model=llm_model)
        self.logger.debug(
            f"[{inspect.stack()[0][3]}] Watson LLM instance {self.llm}")

    def status(self):
        import json
        return json.dumps(self.llm, indent=4)

###
# FOR LOCAL DEVELOPMENT
###
if __name__ == '__main__':
    # only load for local development
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from utils.config import Config

    config = Config()
    #logger = config.logger

    # prompt="What is Kubernetes?"
    prompt = PromptTemplate(
        input_variables=["question"],
        template="""\
            {question}
            Instruction:
            - You are a helpful assistant with expertise in OpenShift and Kubernetes.
            - Do not address questions unrelated to Kubernetes or OpenShift.
            - Refuse to participate in anything that could harm a human.
            - Provide the answer for the question based on the given context.
            - Refuse to answer questions unrelated to topics in Kubernetes or OpenShift.
            - Prefer succinct answers with YAML examples.
            Answer:
            """,
    )

    llm_backends = {'ollama','tgi','openai','bam','watson'}
    print(f"Loading LLM backends {llm_backends}")
    llm_config = LLMLoader(llm_backends=llm_backends)

    for backend in llm_backends:
        print(f"{'#'*40}\n#### Testing LLM backend {backend}")

        llm_chain = LLMChain(llm=llm_config.llm[backend], prompt=prompt)

        q1 = "How to build an application in OpenShift"
        print(f"# Test Prompt 1: {q1}")
        print(llm_chain.run(q1))

        # system should reject this request
        q2 = "Tell me a joke"
        print(f"# Test Prompt 2: {q2}")
        print(llm_chain.run(q2))

        print(f"#### Completed LLM backend {backend} ###")
