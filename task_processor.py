import logging
import sys
from string import Template
from model_context import get_watsonx_predictor
from yes_no_classifier import YesNoClassifier
from task_performer import TaskPerformer

class TaskProcessor:
    def __init__(self):
        logging.basicConfig(
            stream=sys.stdout,
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            level=logging.INFO,
        )
        self.logger = logging.getLogger("task_processor")

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

        self.logger.info(conversation + " Beginning task processing")
        # iterate over the tasks and figure out if we should abort and request more information

        # build a dictionary of stuff to use later
        to_do_stuff = list()

        for task in tasklist:
            self.logger.info(conversation + " task: " + task)

            # determine if we have enough information to answer the task
            task_query = prompt_instructions.substitute(task=task, query=original_query)

            self.logger.info(conversation + " task query: " + task_query)

            self.logger.info(conversation + " usng model: " + model)
            bare_llm = get_watsonx_predictor(model=model, min_new_tokens=5)
            response = str(bare_llm(task_query))

            self.logger.info(conversation + " task response: " + response)

            # strip <|endoftext|> from the reponse
            clean_response = response.split("<|endoftext|>")[0]

            yes_no_classifier = YesNoClassifier()

            # check if the response was a yes or no answer
            # TODO: need to handle when this fails to return an integer
            response_status = int(yes_no_classifier.classify(conversation, model, clean_response))

            self.logger.info(conversation + " response status: " + str(response_status))

            if response_status == 0:
                self.logger.info(
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
                self.logger.info(
                    conversation + " resolution request: " + resolution_request
                )
                return [response_status, resolution_request]
            elif response_status == 1:
                # we have enough information for the task, so go ahead and try to perform it
                to_do_stuff.append(task)
                task_performer = TaskPerformer()
                task_performer.perform_task(conversation, model, task, original_query)
            else:
                self.logger.info(conversation + " Unknown response status")
                return [response_status, "Unknown error occurred"]

            return [1, to_do_stuff]

if __name__ == "__main__":
    task_breakdown = TaskProcessor()
    # arg 1 is the conversation id
    # arg 2 is the desired model
    # arg 3 is a string represnting a python list with the tasks
    # arg 3 is a quoted string to pass as the query
    task_breakdown.process_tasks(sys.argv[1], 
                                   sys.argv[2], 
                                   sys.argv[3].split(','),
                                   sys.argv[4])
