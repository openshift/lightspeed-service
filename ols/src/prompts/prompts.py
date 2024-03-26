"""prompt handler/constants."""

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
