# base python things
import os
from dotenv import load_dotenv

# external deps
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# internal modules
from modules.model_context import get_watsonx_predictor
from modules.yes_no_classifier import YesNoClassifier
from modules.task_performer import TaskPerformer
from modules.task_rephraser import TaskRephraser

# internal tools
from tools.ols_logger import OLSLogger

load_dotenv()

DEFAULT_MODEL = os.getenv("TASK_PROCESSOR_MODEL", "ibm/granite-13b-chat-v1")


class TaskProcessor:
    def __init__(self):
        self.logger = OLSLogger("task_processor").logger

    def process_tasks(self, conversation, tasklist, original_query, **kwargs):
        if "model" in kwargs:
            model = kwargs["model"]
        else:
            model = DEFAULT_MODEL

        if "verbose" in kwargs:
            if kwargs["verbose"] == "True" or kwargs["verbose"] == "true":
                verbose = True
            else:
                verbose = False
        else:
            verbose = False

        settings_string = f"conversation: {conversation}, query: {original_query},model: {model}, verbose: {verbose}"
        self.logger.info(
            conversation
            + " call settings: "
            + settings_string
        )

        prompt_instructions = PromptTemplate.from_template(
            """
Instructions:
- You are a helpful assistant.
- You are an expert in Kubernetes and OpenShift.
- Respond to questions about topics other than Kubernetes and OpenShift with: "I can only answer questions about Kubernetes and OpenShift"
- Refuse to participate in anything that could harm a human.
- Your job is to look at the following description and provide a response.
- Base your answer on the provided task and query and not on prior knowledge.

TASK:
{task}
QUERY:
{query}

Question:
Does the above query contain enough background information to complete the task? Provide a yes or no answer with explanation.

Response:
"""
        )

        self.logger.info(conversation + " Beginning task processing")
        # iterate over the tasks and figure out if we should abort and request more information

        # build a list of stuff to use later
        outputs = list()

        self.logger.info(conversation + " using model: " + model)
        bare_llm = get_watsonx_predictor(model=model, min_new_tokens=5)
        llm_chain = LLMChain(llm=bare_llm, prompt=prompt_instructions, verbose=verbose)

        for task in tasklist:
            self.logger.info(conversation + " task: " + task)

            # determine if we have enough information to answer the task
            task_query = prompt_instructions.format(task=task, query=original_query)

            self.logger.info(conversation + " task query: " + task_query)

            response = llm_chain(inputs={"task": task, "query": original_query})

            self.logger.info(conversation + " task response: " + str(response))

            # TODO: doesn't look like it's needed any more
            # strip <|endoftext|> from the reponse
            clean_response = response["text"].split("<|endoftext|>")[0]

            yes_no_classifier = YesNoClassifier()

            # check if the response was a yes or no answer
            # TODO: need to handle when this fails to return an integer
            response_status = int(
                yes_no_classifier.classify(conversation, clean_response)
            )

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
                # we have enough information for the task

                # rephrase the task and query
                task_rephraser = TaskRephraser()
                rephrased_task = task_rephraser.rephrase_task(
                    conversation, clean_response, original_query
                )
                # to_do_stuff.append(task)

                task_performer = TaskPerformer()
                outputs.append(
                    task_performer.perform_task(
                        conversation, model, rephrased_task, original_query
                    )
                )
                self.logger.info(conversation + " current outputs: " + str(outputs))

            else:
                self.logger.info(conversation + " Unknown response status")
                return [response_status, "Unknown error occurred"]

            return [1, outputs]


if __name__ == "__main__":
    """to execute, from the repo root, use python -m modules.task_processor.py"""
    import argparse

    parser = argparse.ArgumentParser(description="Process a list of tasks")
    parser.add_argument(
        "-c",
        "--conversation-id",
        default="1234",
        type=str,
        help="A short identifier for the conversation",
    )
    parser.add_argument(
        "-t",
        "--tasklist",
        default="task1,task2",
        type=str,
        help="A comma-separated list of tasks to process eg: 'task1,task2'",
    )
    parser.add_argument(
        "-q",
        "--query",
        default="What is the weather like today?",
        type=str,
        help="The user query to use",
    )
    parser.add_argument(
        "-m", "--model", default=DEFAULT_MODEL, type=str, help="The model to use"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=False,
        help="Set Verbose status of langchains [True/False]",
    )

    args = parser.parse_args()

    task_breakdown = TaskProcessor()
    task_breakdown.process_tasks(
        args.conversation_id,
        args.tasklist.split(","),
        args.query,
        model=args.model,
        verbose=args.verbose,
    )
