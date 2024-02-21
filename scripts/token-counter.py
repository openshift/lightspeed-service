"""Utility script for counting tokens in a LlamaIndex index."""

import sys

import tiktoken
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.llms.openai import OpenAI

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: python token-counter.py <docs_dir> <llm>")
        sys.exit(1)

    docs_dir = sys.argv[1]
    llm = sys.argv[2]

    token_counter = TokenCountingHandler(
        tokenizer=tiktoken.encoding_for_model(llm).encode
    )
    Settings.llm = OpenAI(model=llm, temperature=0.2)
    Settings.callback_manager = CallbackManager([token_counter])

    documents = SimpleDirectoryReader(docs_dir, recursive=True).load_data()

    index = VectorStoreIndex.from_documents(documents)
    print(token_counter.total_embedding_token_count)
