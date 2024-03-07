"""prompt handler/constants."""

from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)

# TODO: Fine-tune system prompt
QUERY_SYSTEM_PROMPT = """You are an assistant for question-answering tasks \
related to the openshift and kubernetes container orchestration platforms. \
Use the previous chat history to interact and help the user.
Use the following pieces of retrieved context to answer the question.

{context}
"""

CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(QUERY_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{query}"),
    ]
)
