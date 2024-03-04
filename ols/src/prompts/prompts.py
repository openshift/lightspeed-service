"""prompt handler/constants."""

from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

# TODO: Fine-tune system prompt
QUERY_SYSTEM_PROMPT = """You are an assistant for question-answering tasks \
related to the openshift and kubernetes container orchestration platforms. \
Use the following pieces of retrieved context to answer the question.

{context}"""

# TODO: Add placeholder for history
CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(QUERY_SYSTEM_PROMPT),
        HumanMessagePromptTemplate.from_template("{query}"),
    ]
)
