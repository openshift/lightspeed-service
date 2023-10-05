import logging
import sys
from model_context import get_watsonx_predictor

def task_processor(conversation, tasklist, original_query):
    logging.info(conversation + ' Beginning task processing')
    for task in tasklist:
        logging.info(conversation + " task: " + task)

        # determine if we have enough information to answer the task
        # first, construct the overall query

        task_query = (
            "Given the question: " + original_query +
            " do you have enough information perform the task: " + task
        )
        logging.info(conversation + ' task query: ' + task_query)

        bare_llm = get_watsonx_predictor(model="ibm/granite-13b-instruct-v1")
        response = bare_llm.complete(task_query)

        logging.info(conversation + ' task response: ' + str(response))

    return