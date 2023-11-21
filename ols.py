# base python things
import uuid
from typing import Union
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Request

# auxiliary functionalities
from src.config import Config

# internal modules
from modules.question_validator import QuestionValidator
from modules.yaml_generator import YamlGenerator
from modules.happy_response_generator import HappyResponseGenerator
from modules.docs_summarizer import DocsSummarizer
from modules.model_context import get_watsonx_predictor
from modules.conversation_cache import LRUCache

# gradio ui
from modules.gradio_ui import gradioUI

# TODO: Move to external file
class LLMRequest(BaseModel):
    query: str
    conversation_id: Union[str, None] = None
    response: Union[str, None] = None

class FeedbackRequest(BaseModel):
    conversation_id: int # required
    feedback_object: str # a json blob

# common globals
app = FastAPI()

config = Config()
logger = config.logger

# determine if need to enable local UI
if config.enable_ui:
    app=gradioUI().mount_ui(app)
else:
    logger.info("Embedded Gradio UI is disabled. To enable set OLS_ENABLE_UI=True")

# application global variables
base_completion_model = config.base_completion_model
conversation_cache = LRUCache(100)

def get_suid():
    return str(uuid.uuid4().hex)

@app.get("/healthz")
@app.get("/readyz")
def read_root():
    return {"status": "1"}

@app.get("/")
@app.get("/status")
def root(request: Request):
    """
    TODO: In the future should respond
    """
    return {
        "message" : "This is the default endpoint for OLS",
        "status" : "running"
        }

@app.post("/ols")
def ols_request(llm_request: LLMRequest):
    """
    This endpoint is for the alternative flow
        1. validate whether the query is about k8s/ocp
        2. pass to yaml generator
        3. filter/clean/lint
        4. RAG for supporting documentation
        5. user-friendly summary
    """
    previous_input=None

    conversation=llm_request.conversation_id
    if conversation==None:   
        # generate a new unique UUID for the request:
        conversation = get_suid()
        logger.info(conversation + " New conversation")
    else:
        previous_input=conversation_cache.get(conversation)
        logger.info(conversation + " Previous conversation input: " + previous_input)

    llm_response = LLMRequest(query=llm_request.query,conversation_id=conversation)

    # TODO: some kind of logging module that includes the conversation automatically?
    logger.info(conversation + " Incoming request: " + llm_request.query)

    # determine if the query is about OpenShift or Kubernetes
    question_validator = QuestionValidator()

    is_valid = question_validator.validate_question(conversation, llm_request.query)
    if is_valid[0] == "INVALID":
        logger.info(conversation + " question was determined to not be k8s/ocp, so rejecting")
        llm_response.response = ("Sorry, I can only answer questions about "
                                 "OpenShift and Kubernetes. This does not look "
                                 "like something I know how to handle.")
        raise HTTPException(status_code=422, detail=llm_response.dict())
    if is_valid[0] == "VALID":
        logger.info(conversation + " question is about k8s/ocp")
        # the LLM thought the question was valid, so decide if it's about YAML or not

        # generate a user-friendly response to wrap the YAML and/or the supporting information
        response_wrapper = HappyResponseGenerator()
        wrapper = response_wrapper.generate(conversation, llm_request.query)

        if is_valid[1] == "NOYAML":
            logger.info(conversation + " question is not about yaml, so send for generic info")

            docs_summarizer = DocsSummarizer()
            summary, referenced_documents = docs_summarizer.summarize(conversation, llm_request.query)

            llm_response.response = wrapper + "\n" + summary
            return llm_response
        elif is_valid[1] == "YAML":
            logger.info(conversation + " question is about yaml, so send to the YAML generator")
            yaml_generator = YamlGenerator()
            generated_yaml = yaml_generator.generate_yaml(conversation, llm_request.query, previous_input)

            if generated_yaml == "some failure":
                # we didn't get any kind of yaml markdown block back from the model
                llm_response.response = (
                    "Sorry, something bad happened internally. Please try again."
                )
                raise HTTPException(status_code=500, detail=llm_response.dict())

            # we got some kind of valid yaml back from the yaml generator, so proceed

            # filter/clean/lint the YAML response

            # RAG for supporting documentation

            llm_response.response = wrapper + "\n" + generated_yaml
            conversation_cache.upsert(conversation,llm_request.query+"\n\n"+llm_response.response)
            return llm_response

        else:
            # something weird happened, so generate an internal error
            # something bad happened with the validation
            llm_response.response = (
                "Sorry, something bad happened internally. Please try again."
            )
        raise HTTPException(status_code=500, detail=llm_response.dict())
    else:
        # something bad happened with the validation
        llm_response.response = (
            "Sorry, something bad happened internally. Please try again."
        )
        raise HTTPException(status_code=500, detail=llm_response.dict())


@app.post("/ols/raw_prompt")
@app.post("/base_llm_completion")
def base_llm_completion(llm_request: LLMRequest):
    """
    Raw pass through to backend LLM
    """
    conversation = get_suid()

    llm_response = LLMRequest(query=llm_request.query)
    llm_response.conversation_id = conversation

    logger.info(conversation + " New conversation")

    logger.info(conversation + " Incoming request: " + llm_request.query)
    bare_llm = get_watsonx_predictor(model=base_completion_model)

    response = bare_llm(llm_request.query)

    # TODO: make the removal of endoftext some kind of function
    clean_response = response.split("<|endoftext|>")[0]
    llm_response.response = clean_response

    logger.info(conversation + " Model returned: " + llm_response.response)

    return llm_response

@app.post("/feedback")
def feedback_request(feedback_request: FeedbackRequest):
    """
    Endpoint to collect direct feedback on conversation
    """

    conversation = str(feedback_request.conversation_id)
    logger.info(conversation + " New feedback received")
    logger.info(conversation + " Feedback blob: " + feedback_request.feedback_object)

    return {"status":"feedback received"}
