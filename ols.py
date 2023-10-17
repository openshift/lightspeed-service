from typing import Union
from fastapi import FastAPI, HTTPException

from pydantic import BaseModel
from task_breakdown import TaskBreakdown
from task_processor import TaskProcessor

import logging, sys, os

logging.basicConfig(
    stream=sys.stdout,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    level=logging.INFO,
)

import uuid

## internal stuff
from model_context import get_watsonx_predictor

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

    # task_list, referenced_documents = task_breakdown.breakdown_tasks(
    #   conversation, llm_request.query
    # )

    task_list = [
        "1. Create a ClusterAutoscaler YAML that specifies the maximum size of the cluster"]

    #task_list = [
    #    "1. Create a ClusterAutoscaler YAML that specifies the maximum size of the cluster",
    #    "2. Create a MachineAutoscaler YAML object to specify which MachineSet should be scaled and the minimum and maximum number of replicas."
    #]

    # task_list = [
    #    "1. Determine the maximum number of nodes desired for the cluster.",
    #    "2. Create a ClusterAutoscaler that specifies the size of the cluster.",
    #    "3. Create a MachineAutoscaler object to specify which MachineSet should be scale and the minimum and maximum number of replicas.",
    # ]

    logging.info(conversation + " Task list: " + str(task_list))

    # determine if the various tasks can be performed with the information given
    # TODO: should this maybe be called task validator?
    task_processor = TaskProcessor()
    processed = task_processor.process_tasks(
        conversation, task_list, llm_request.query
    )

    # when the 0th element is 0, the task processor has encountered an error, likely that it can't figure out what to do
    # with the provided information in the query. alert the user
    if processed[0] == 0:
        llm_response.response = processed[1]
        raise HTTPException(status_code=500, detail=llm_response.dict())

    # TODO: handle a response of a 9 which means there was some other issue in understanding the task

    llm_response.response = processed[1]

    # if we got this far, all of the tasks can be completed, so do something
    return llm_response


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
