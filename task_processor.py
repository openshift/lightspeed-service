import logging
import sys
from string import Template
from model_context import get_watsonx_predictor
from yes_no_classifier import YesNoClassifier

class TaskProcessor:
    def process_tasks(self, conversation, model, tasklist, original_query):
        prompt_instructions = Template(
            """Instructions:
        - You are a helpful assistant.
        - You are an expert in Kubernetes and OpenShift.
        - Respond to questions about topics other than Kubernetes and OpenShift with: "I can only answer questions about Kubernetes and OpenShift"
        - Refuse to participate in anything that could harm a human.
        - Your job is to look at the following description and provide a response.
        - Base your answer on the provided task and query and not on prior knowledge.

        TASK:
        ${task}
        QUERY:
        ${query}

        Question:
        Does the above query contain enough information about the task? Provide a yes or no answer with explanation.

        Response:
        """
        )

        logging.info(conversation + " Beginning task processing")
        # iterate over the tasks and figure out if we should abort and request more information

        # build a dictionary of stuff to use later
        to_do_stuff = list()

        for task in tasklist:
            logging.info(conversation + " task: " + task)

            # determine if we have enough information to answer the task
            task_query = prompt_instructions.substitute(task=task, query=original_query)

            logging.info(conversation + " task query: " + task_query)

            logging.info(conversation + " usng model: " + model)
            bare_llm = get_watsonx_predictor(model=model, min_new_tokens=5)
            response = str(bare_llm(task_query))

            logging.info(conversation + " task response: " + response)

            # strip <|endoftext|> from the reponse
            clean_response = response.split("<|endoftext|>")[0]

            yes_no_classifier = YesNoClassifier()

            # check if the response was a yes or no answer
            # TODO: need to handle when this fails to return an integer
            response_status = int(yes_no_classifier.classify(conversation, model, clean_response))

            logging.info(conversation + " response status: " + str(response_status))

            if response_status == 0:
                logging.info(
                    conversation
                    + " Aborting task processing for no response - need details"
                )
                resolution_request = str(
                    "In trying to answer your question, we were unable to determine how to proceed."
                    " The step we failed on was the following:\n "
                    + task
                    + " The failure message was:\n "
                    + clean_response
                    + " Please try rephrasing your request to include information that might help complete the task."
                )
                logging.info(
                    conversation + " resolution request: " + resolution_request
                )
                return [response_status, resolution_request]
            elif response_status == 1:
                # we have enough information for the task, so put the task into a list for later
                to_do_stuff.append(task)
                logging.info(conversation + " Continuing task processing")
            else:
                logging.info(conversation + " Unknown response status")
                return [response_status, "Unknown error occurred"]

            return [1, to_do_stuff]

# TODO: make executable directly with name main