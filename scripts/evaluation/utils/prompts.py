# ruff: noqa: E501
"""Prompt templates/constants."""

# Below is inspired by both ragas & langchain internal/example prompts.
ANSWER_RELEVANCY_PROMPT = """
You are an helpful assistant. Your task is to analyze answer and come up with questions from the given answer.
Given the following answer delimited by three backticks please generate {num_questions} questions.
A question should be concise and based explicitly on the information present in answer. It should be asking about one thing at a time.
Give Valid as 1 if the answer is valid and 0 if the answer is invalid. An invalid answer is one that is evasive, vague, or ambiguous. For example, "I don't know" or "I'm not sure" are noncommittal answers.
When formulating a question, don't include text like "from the provided context", "as described in the document", "according to the given document" or anything similar. Also don't add sequence number in question.

Use below json format for your response. Do not add any additional text apart from json output.
{{
    Question: [
        QUESTION 1,
        QUESTION 2,
    ],
    Valid: 0 or 1
}}

```
{answer}
```
"""
