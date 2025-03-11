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
Consider the following criteria when grading:
1. Semantic similarity: How closely the predicted answer matches the meaning of the real answer.
2. Factual accuracy: Whether the predicted answer is factually correct.
3. Relevance: Whether the predicted answer is relevant to the question.
4. Completeness: Whether the predicted answer covers all key points of the real answer.

Only give the score value as final response.
"""

# TODO: Will add alternate approach, once we have expected context.
RAG_RELEVANCY_PROMPT1 = """You are an expert in validating search result.
Your task is to evaluate different search results against actual search query.

What score do you give from 0 to 10, where 0 is the lowest and 10 is the highest ?
Provide score for each of the below aspects:
* Relevance: The search result contains relevant information to address the query.
* Completeness: The search result contains complete information.
* Conciseness: The search result contains only related information.

Provide scores & summarized explaination.
Use below json format for your response. Do not add any additional text apart from json output.

Output format:
{{
    "explaination": [Summarized explaination for search result 1, Summarized explaination for search result 2 ...],
    "relevance_score": [Relevance score for search result 1, Relevance score for search result 2 ...],
    "completeness_score": [Completeness score for search result 1, Completeness score for search result 2 ...],
    "conciseness_score": [Conciseness score for search result 1, Conciseness score for search result 2 ...]
}}

Actual search query: "{query}"
You are evaluating below {n_results} search results:
```
{retrieval_texts}
```
"""

# Keeping it simpler
RAG_RELEVANCY_PROMPT2 = """You are an expert in validating search result.
Your task is to evaluate different search results against actual search query.

What score do you give from 0 to 10, where 0 is the lowest and 10 is the highest ?
Consider below aspects to give a final score :
* Relevance: The search result contains relevant information to address the query.
* Completeness: The search result contains complete information.

Provide final score & your explaination.
Use below json format for your response. Do not add any additional text apart from json output.
{{
    "explaination": [Explaination behind the score for search result 1, Explaination behind the score for search result 2 ...],
    "final_score": [Final score for search result 1, Final score for search result 2 ...]
}}

Actual search query: "{query}"
You are evaluating below {n_results} search results:
```
{retrieval_texts}
```
"""
