# Imports
import llama_index
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.storage.storage_context import StorageContext
from model_context import get_watsonx_context

import logging, sys, os
#logging.basicConfig(stream=sys.stdout, level=logging.INFO)
#logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

llama_index.set_global_handler("simple")

# Select Model
service_context = get_watsonx_context(model="ibm/granite-13b-chat-v1")

# Load data
filename_fn = lambda filename: {'file_name': filename}

storage_context = StorageContext.from_defaults()

summary_documents = SimpleDirectoryReader('data/summary-docs', file_metadata=filename_fn).load_data()
summary_index = VectorStoreIndex.from_documents(summary_documents, storage_context=storage_context, service_context=service_context)
summary_index.set_index_id("summary")
storage_context.persist(persist_dir="./vector-db")

print("Done indexing!")

