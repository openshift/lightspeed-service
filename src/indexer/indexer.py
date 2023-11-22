import os
from dotenv import load_dotenv
import llama_index
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.storage.storage_context import StorageContext
from utils.model_context import get_watsonx_context
import src.constants as constants

llama_index.set_global_handler("simple")

load_dotenv()
model = os.getenv("INDEXER_MODEL", "ibm/granite-13b-chat-v1")


# Select Model
## check if we are using remote embeddings via env
tei_embedding_url = os.getenv("TEI_SERVER_URL", None)

if tei_embedding_url != None:
    service_context = get_watsonx_context(
        model=model, tei_embedding_model="BAAI/bge-base-en-v1.5", url=tei_embedding_url
    )
else:
    service_context = get_watsonx_context(model=model)

print("Using embed model: " + str(service_context.embed_model))

# Load data
filename_fn = lambda filename: {"file_name": filename}

storage_context = StorageContext.from_defaults()

# index the summary documents
print("Indexing summary documents...")
summary_documents = SimpleDirectoryReader(
    "data/summary-docs", file_metadata=filename_fn
).load_data()
summary_index = VectorStoreIndex.from_documents(
    summary_documents,
    storage_context=storage_context,
    service_context=service_context,
    show_progress=True,
)
summary_index.set_index_id(constants.SUMMARY_INDEX)
storage_context.persist(persist_dir=constants.SUMMARY_DOCS_PERSIST_DIR)

# index the product documentation
print("Indexing product documents...")
product_documents = SimpleDirectoryReader(
    "data/ocp-product-docs-md", file_metadata=filename_fn
).load_data()
product_index = VectorStoreIndex.from_documents(
    product_documents,
    storage_context=storage_context,
    service_context=service_context,
    show_progress=True,
)
product_index.set_index_id(constants.PRODUCT_INDEX)
storage_context.persist(persist_dir=constants.PRODUCT_DOCS_PERSIST_DIR)

print("Done indexing!")
