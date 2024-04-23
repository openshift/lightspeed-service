# There is no need for enforcing line length in this file,
# as these are mostly special purpose constants.
# ruff: noqa: E501
"""Prompt templates/constants."""

from ols.constants import SUBJECT_ALLOWED, SUBJECT_REJECTED

# TODO: OLS-503 Fine tune system prompt

# Note::
# Right now templates are somewhat alligned to make granite work better.
# GPT still works well with this. Ideally we should have model specific tags.
# For history we can laverage ChatPromptTemplate from langchain,
# but that is not done as granite was adding role tags like `Human:` in the response.
# With PromptTemplate, we have more control how we want to structure the prompt.

QUERY_SYSTEM_INSTRUCTION = """
You are an assistant for question-answering tasks \
related to the openshift and kubernetes container orchestration platforms.

"""

USE_CONTEXT_INSTRUCTION = """
Use the retrieved document to answer the question.
"""

CONTEXT_PLACEHOLDER = """

[DOCUMENT]
{context}
[END]

"""

USE_HISTORY_INSTRUCTION = """
Use the previous chat history to interact and help the user.
"""

HISTORY_PLACEHOLDER = """

[HISTORY]
{chat_history}
[END]

"""

QUERY_PLACEHOLDER = """
<|user|>
{query}
<|assistant|>

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
