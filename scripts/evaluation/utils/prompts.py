# ruff: noqa: E501
"""Prompt templates/constants."""

# Use below as system instruction to override OLS instruction.
# Instruction for RAG/Context is still used with proper document format.
BASIC_PROMPT = """
You are a helpful assistant.
"""

# Below is inspired by both ragas & langchain internal/example prompts.
ANSWER_RELEVANCY_PROMPT = """You are an helpful assistant. Your task is to analyze answer and come up with questions from the given answer.
Given the following answer delimited by three backticks please generate {num_questions} questions.
A question should be concise and based explicitly on the information present in answer. It should be asking about one thing at a time.
Give Valid as 1 if the answer is valid and 0 if the answer is invalid. An invalid answer is one that is evasive, vague, or ambiguous. For example, "I don't know" or "I'm not sure" are noncommittal answers.
When formulating a question, don't include text like "from the provided context", "as described in the document", "according to the given document" or anything similar. Also don't add sequence number in question.

Use below json format for your response. Do not add any additional text apart from json output.
{{
    "Question": [
        QUESTION 1,
        QUESTION 2,
    ],
    "Valid": 0 or 1
}}

```
{answer}
```
"""

ANSWER_SIMILARITY_PROMPT = """You are an expert professor specialized in grading students' answers to questions.
You are grading the following question:
{question}
Here is the real answer:
{answer}
You are grading the following predicted answer:
{response}
What grade do you give from 0 to 10, where 0 is the lowest (very low similarity) and 10 is the highest (very high similarity)?
Only give the score value.
"""
