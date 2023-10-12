import logging
import sys
from string import Template
from model_context import get_watsonx_predictor


class TaskPerformer:
    def perform_task(self, conversation, model, task, original_query):
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
        BACKGROUND INFORMATION:
        ${query}

        Question:
        Please complete the task using the background information.

        Response:
        """
        )
        logging.info(conversation + " Performing the task: " + task)
        task_query = prompt_instructions.substitute(task=task, query=original_query)

        logging.info(conversation + " task query: " + task_query)

        logging.info(conversation + " usng model: " + model)
        bare_llm = get_watsonx_predictor(model=model, min_new_tokens=5)
        response = str(bare_llm(task_query))
        logging.info(conversation + " task response: " + response)
        return
