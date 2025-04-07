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

TAXONOMY_CONTEXT_RELEVANCY = """You are an expert in validating data used question & answer generation.
You are given context, question and answer pair generated from that context.

Your tasks are below.
is the given context relevant for the given question ? Give a valid_flag with either 1 or 0.
Does the context contain all information present in the Answer. Give a context_relevancy score between 0 to 1.

Rule for valid_flag:
This is the evaluation of the context against the question.
Set the valid_flag as 1, if the context contains enough information to answer the question. Other wise set it to 0.

Rule for context_relevancy score:
This is the evaluation of the context against the answer.
Give relevancy_score as 1, when every information in answer present in the given context.
Give the score as 0, when no information matches with the context.
Give a score between 0 to 1, where details in answer partially matches with the context.

Use below json format for your response. Do not add any additional text apart from json output.
{{
    "valid_flag": 0 or 1,
    "relevancy_score": between 0 to 1
}}

Question: "{question}"
Answer:
```
{answer}
```
Context:
```
{context}
```
"""

GROUNDNESS_PROMPT = """You are an expert in grading answer for a given question.
You are given reference data, question and answer pair generated from that reference data.
Your task is to grade answer based on based on question and given context.

Grade against question:
Grade and give a relevancy_score for answer between 0 to 1 based on how accurately it answers the question.
Best answer will be given a score of 1 and worst answer will be given 0.

Grade against context:
Grade and give a groundness_score for answer between 0 to 1 based on how factually it is correct based on given context.
When answer is fully supported by given reference data then give a score of 1.

Use below json format for your response. Do not add any additional text apart from json output.
{{
    "relevancy_score": between 0 to 1,
    "groundness_score": between 0 to 1
}}

Question: "{question}"
Answer:
```
{answer}
```
Reference Data:
```
{context}
```
"""
