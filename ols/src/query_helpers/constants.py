"""Constants used for LLM chain."""

from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)

from ols import constants

summary_prompt_for_langchain = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(constants.SUMMARIZATION_TEMPLATE),
        # The `variable_name` here is what must align with memory
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{query}"),
    ]
)
