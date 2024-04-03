# There is no need for enforcing line length in this file,
# as these are mostly special purpose constants.
# ruff: noqa: E501
"""Prompt templates/constants."""

from ols.constants import SUBJECT_ALLOWED, SUBJECT_REJECTED

# TODO: Fine-tune system prompt
QUERY_SYSTEM_PROMPT = """You are an assistant for question-answering tasks \
related to the openshift and kubernetes container orchestration platforms. \
"""

USE_PREVIOUS_HISTORY = """
Use the previous chat history to interact and help the user.
"""

USE_RETRIEVED_CONTEXT = """
Use the following pieces of retrieved context to answer the question.

{context}
"""

# {{query}} is escaped because it will be replaced as a parameter at time of use
QUESTION_VALIDATOR_PROMPT_TEMPLATE = f"""
Instructions:
- You are a question classifying tool
- You are an expert in kubernetes and openshift
- Your job is to determine where or a user's question is related to kubernetes and/or openshift technologies and to provide a one-word response
- If a question appears to be related to kubernetes or openshift technologies, answer with the word {SUBJECT_ALLOWED}, otherwise answer with the word {SUBJECT_REJECTED}
- Do not explain your answer, just provide the one-word response


Example Question:
Why is the sky blue?
Example Response:
{SUBJECT_REJECTED}

Example Question:
Can you help configure my cluster to automatically scale?
Example Response:
{SUBJECT_ALLOWED}

Example Question:
How do I accomplish $task in openshift?
Example Response:
{SUBJECT_ALLOWED}

Question:
{{query}}
Response:
"""

SUMMARIZATION_TEMPLATE = """
The following context contains several pieces of documentation. Please answer the user's question based on this context.
Documentation context:
{context_str}

User query:
{query_str}

Summary:

"""

SUMMARY_TASK_BREAKDOWN_TEMPLATE = (
    """
The following documentation contains a task list. Your job is to extract the list of tasks. """
    """If the user-supplied query seems unrelated to the list of tasks, please reply that you do not know what to do with the query and the summary documentation. """
    """Use only the supplied content and extract the task list.

Summary document:
{context_str}

User query:
{query_str}

What are the tasks?
"""
)

TASK_PERFORMER_PROMPT_TEMPLATE = """
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

TASK_REPHRASER_PROMPT_TEMPLATE = """
Instructions:
- You are a helpful assistant.
- Your job is to combine the information from the task and query into a single, new task.
- Base your answer on the provided task and query and not on prior knowledge.

TASK:
{task}
QUERY:
{query}

Please combine the information from the task and query into a single, new task.

Response:
"""

YES_OR_NO_CLASSIFIER_PROMPT_TEMPLATE = """
Instructions:
- determine if a statement is a yes or a no
- return a 1 if the statement is a yes statement
- return a 0 if the statement is a no statement
- return a 9 if you cannot determine if the statement is a yes or no

Examples:
Statement: Yes, that sounds good.
Response: 1

Statement: No, I don't think that is wise.
Response: 0

Statement: Apples are red.
Response: 9

Statement: {statement}
Response:
"""
