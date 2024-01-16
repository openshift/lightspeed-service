"""Document indexer."""

import os

import llama_index
from dotenv import load_dotenv
from llama_index import ServiceContext, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings import TextEmbeddingsInference
from llama_index.storage.storage_context import StorageContext

from ols import constants

llama_index.set_global_handler("simple")

load_dotenv()

# Select Model
embed_model: TextEmbeddingsInference | str

## check if we are using remote embeddings via env
url = os.getenv("TEI_SERVER_URL", "local")
if url != "local":
    embed_model = TextEmbeddingsInference(
        model_name="BAAI/bge-base-en-v1.5",
        base_url=url,
    )
else:
    embed_model = "local:BAAI/bge-base-en"

service_context = ServiceContext.from_defaults(
    chunk_size=1024, llm=None, embed_model=embed_model
)

print("Using embed model: " + str(service_context.embed_model))


# Load data
def filename_fn(filename):
    """Constructs file metadata with filename."""
    return {"file_name": filename}


storage_context = StorageContext.from_defaults()

# index the summary documents
# https://gitlab.cee.redhat.com/openshift/lightspeed-rag-documents/-/tree/main/summary-docs?ref_type=heads
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
# https://gitlab.cee.redhat.com/openshift/lightspeed-rag-documents/-/tree/main/ocp-product-docs-plaintext?ref_type=heads
print("Indexing product documents...")
product_documents = SimpleDirectoryReader(
    input_dir="data/ocp-product-docs-plaintext",
    recursive=True,
    file_metadata=filename_fn,
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
