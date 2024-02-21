"""Utility script to generate embeddings."""

import argparse
import json
import os
import time

import faiss
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from llama_index import ServiceContext, SimpleDirectoryReader, VectorStoreIndex
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore

PRODUCT_INDEX = "product"

if __name__ == "__main__":

    start_time = time.time()

    parser = argparse.ArgumentParser(description="embedding cli for task execution")
    parser.add_argument("-f", "--folder", help="Plain text folder path")
    parser.add_argument(
        "-m",
        "--model",
        default="embeddings_model",
        help="LLM model used for embeddings [local, llama2, or any other supported by llama_index]",
    )
    parser.add_argument(
        "-c", "--chunk", type=int, default="1500", help="Chunk size for embedding"
    )
    parser.add_argument(
        "-l", "--overlap", type=int, default="10", help="Chunk overlap for embedding"
    )
    parser.add_argument("-o", "--output", help="Vector DB output folder")
    parser.add_argument(
        "-s", "--faiss-vector-size", type=int, default="768", help="Faiss vector size"
    )
    args = parser.parse_args()

    PERSIST_FOLDER = args.output

    faiss_index = faiss.IndexFlatL2(args.faiss_vector_size)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    os.environ["HF_HOME"] = args.model
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    embed_model = HuggingFaceBgeEmbeddings(model_name=args.model)
    service_context = ServiceContext.from_defaults(
        chunk_size=args.chunk,
        chunk_overlap=args.overlap,
        embed_model=embed_model,
        llm=None,
    )

    documents = SimpleDirectoryReader(args.folder, recursive=True).load_data()

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        service_context=service_context,
        show_progress=True,
    )
    index.set_index_id(PRODUCT_INDEX)
    index.storage_context.persist(persist_dir=PERSIST_FOLDER)

    metadata = {}
    metadata["execution-time"] = time.time() - start_time
    metadata["llm"] = "None"
    metadata["embedding-model"] = args.model
    metadata["index_id"] = PRODUCT_INDEX
    metadata["vector-db"] = "faiss"
    metadata["chunk"] = args.chunk
    metadata["overlap"] = args.overlap
    metadata["total-embedded-files"] = len(documents)

    with open(os.path.join(PERSIST_FOLDER, "metadata.json"), "w") as file:
        file.write(json.dumps(metadata))
