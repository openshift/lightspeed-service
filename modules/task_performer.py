# base python things
import os
from dotenv import load_dotenv

# internal tools
from tools.ols_logger import OLSLogger

load_dotenv()

DEFAULT_MODEL = os.getenv("TASK_PERFORMER_MODEL", "ibm/granite-13b-chat-v1")


class TaskPerformer:
    def __init__(self):
        self.logger = OLSLogger("task_performer").logger

    def perform_task(self, conversation, task, **kwargs):
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


        settings_string = f"conversation: {conversation}, task: {task},model: {model}, verbose: {verbose}"
        self.logger.info(
            conversation
            + " call settings: "
            + settings_string
        )

        # determine if this should go to a general LLM, the YAML generator, or elsewhere

        # send to the right tool

        # output the response
        response = """
apiVersion: "autoscaling.openshift.io/v1"
kind: "ClusterAutoscaler"
metadata:
  name: "default"
spec:
  resourceLimits:
    maxNodesTotal: 10
  scaleDown: 
    enabled: true 
"""

        self.logger.info(conversation + " response: " + response)

        # return the response

        return response


if __name__ == "__main__":
    """to execute, from the repo root, use python -m modules.task_performer.py"""
    import argparse

    parser = argparse.ArgumentParser(description="Perform a task")
    parser.add_argument(
        "-c",
        "--conversation-id",
        default="1234",
        type=str,
        help="A short identifier for the conversation",
    )
    parser.add_argument(
        "-t",
        "--task",
        default="Create an autoscaler YAML that scales the cluster up to 10 nodes",
        type=str,
        help="A task to perform",
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

    task_performer = TaskPerformer()
    task_performer.perform_task(
        args.conversation_id, args.task, model=args.model, verbose=args.verbose
    )
