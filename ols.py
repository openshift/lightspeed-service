from typing import Union
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

from pydantic import BaseModel
from modules.task_breakdown import TaskBreakdown
from modules.task_processor import TaskProcessor
from modules.question_validator import QuestionValidator
from modules.yaml_generator import YamlGenerator
from modules.happy_response_generator import HappyResponseGenerator
from modules.docs_summarizer import DocsSummarizer
import logging, sys, os

logging.basicConfig(
    stream=sys.stdout,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    level=logging.INFO,
)

import uuid

## internal stuff
from modules.model_context import get_watsonx_predictor

load_dotenv()

instruct_model = os.getenv("INSTRUCT_MODEL", "ibm/granite-13b-instruct-v1")
rag_model = os.getenv("RAG_MODEL", "ibm/granite-13b-chat-grounded-v01")

# TODO: env for verbose chains


class LLMRequest(BaseModel):
    query: str
    conversation_id: Union[int, None] = None
    response: Union[str, None] = None


app = FastAPI()


def get_suid():
    return str(uuid.uuid4().hex)


@app.get("/healthz")
def read_root():
    return {"status": "1"}


@app.post("/ols")
def ols_request(llm_request: LLMRequest):
    # TODO: probably should setup all models to use here to avoid setting them
    # up over and over

    # generate a unique UUID for the request:
    conversation = get_suid()
    llm_response = LLMRequest(query=llm_request.query)
    llm_response.conversation_id = conversation

    logging.info(conversation + " New conversation")

    # TODO: some kind of logging module that includes the conversation automatically?
    logging.info(conversation + " Incoming request: " + llm_request.query)

    # determine what tasks are required to perform the query
    task_breakdown = TaskBreakdown()

    # TODO: make a real task breakdown call
    # task_list, referenced_documents = task_breakdown.breakdown_tasks(
    #   conversation, llm_request.query
    # )

    # TODO: remove fake response
    task_list = [
        "1. Create a ClusterAutoscaler YAML that specifies the maximum size of the cluster"
    ]

    # task_list = [
    #    "1. Create a ClusterAutoscaler YAML that specifies the maximum size of the cluster",
    #    "2. Create a MachineAutoscaler YAML object to specify which MachineSet should be scaled and the minimum and maximum number of replicas."
    # ]

    logging.info(conversation + " Task list: " + str(task_list))

    # determine if the various tasks can be performed with the information given
    # TODO: should this maybe be called task validator?
    task_processor = TaskProcessor()
    processed = task_processor.process_tasks(conversation, task_list, llm_request.query)

    # when the 0th element is 0, the task processor has encountered an error, likely that it can't figure out what to do
    # with the provided information in the query. alert the user
    if processed[0] == 0:
        llm_response.response = processed[1]
        raise HTTPException(status_code=500, detail=llm_response.dict())

    # TODO: handle a response of a 9 which means there was some other issue in understanding the task

    llm_response.response = processed[1]

    # if we got this far, all of the tasks can be completed, so do something
    return llm_response


@app.post("/ols2")
def ols2_request(llm_request: LLMRequest):
    # this endpoint is for the alternative flow
    # 1. validate whether the query is about k8s/ocp
    # 2. pass to yaml generator
    # 3. filter/clean/lint
    # 4. RAG for supporting documentation
    # 5. user-friendly summary

    # generate a unique UUID for the request:
    conversation = get_suid()
    llm_response = LLMRequest(query=llm_request.query)
    llm_response.conversation_id = conversation

    logging.info(conversation + " New conversation")

    # TODO: some kind of logging module that includes the conversation automatically?
    logging.info(conversation + " Incoming request: " + llm_request.query)

    # determine if the query is about OpenShift or Kubernetes
    question_validator = QuestionValidator()

    is_valid = question_validator.validate_question(conversation, llm_request.query)
    if is_valid[0] == "INVALID":
        logging.info(conversation + " question was determined to not be k8s/ocp, so rejecting")
        llm_response.response = ("Sorry, I can only answer questions about "
                                 "OpenShift and Kubernetes. This does not look "
                                 "like something I know how to handle.")
        raise HTTPException(status_code=422, detail=llm_response.dict())
    if is_valid[0] == "VALID":
        logging.info(conversation + " question is about k8s/ocp")
        # the LLM thought the question was valid, so decide if it's about YAML or not

        # generate a user-friendly response to wrap the YAML and/or the supporting information
        response_wrapper = HappyResponseGenerator()
        wrapper = response_wrapper.generate(conversation, llm_request.query)

        if is_valid[1] == "NOYAML":
            logging.info(conversation + " question is not about yaml, so send for generic info")

            docs_summarizer = DocsSummarizer()
            summary, referenced_documents = docs_summarizer.summarize(conversation, llm_request.query)

            llm_response.response = wrapper + "\n" + summary
            return llm_response
        elif is_valid[1] == "YAML":
            logging.info(conversation + " question is about yaml, so send to the YAML generator")
            yaml_generator = YamlGenerator()
            generated_yaml = yaml_generator.generate_yaml(conversation, llm_request.query)

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


@app.post("/base_llm_completion")
def base_llm_completion(llm_request: LLMRequest):
    conversation = get_suid()
    llm_response = LLMRequest(query=llm_request.query)
    llm_response.conversation_id = conversation

    logging.info(conversation + " New conversation")

    logging.info(conversation + " Incoming request: " + llm_request.query)
    bare_llm = get_watsonx_predictor(model=instruct_model)

    response = bare_llm(llm_request.query)

    # TODO: make the removal of endoftext some kind of function
    clean_response = response.split("<|endoftext|>")[0]
    llm_response.response = clean_response

    logging.info(conversation + " Model returned: " + llm_response.response)

    return llm_response
