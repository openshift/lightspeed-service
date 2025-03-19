"""Utility script for querying RAG database for set of questions."""

import argparse
import os
from collections import defaultdict

from llama_index.core import Settings, load_index_from_storage
from llama_index.core.llms.utils import resolve_llm
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from pandas import DataFrame

QNA_QUERIES = [
    "what is kubernetes?",
    "what is openshift virtualization?",
    "What is the purpose of the imagePullPolicy in Red Hat OpenShift Container Platform?",
    # "What is the purpose of the APIRequestCount object?",
    # "What is the purpose of the ClusterVersion object in OpenShift updates?",
    "How does Red Hat OpenShift Pipelines automate deployments?",
    "what is a limitrange?",
    "What is the purpose of the Vertical Pod Autoscaler Operator in Openshift?",
    "Is there a doc on updating clusters?",
    # "Can you tell me how to install OpenShift in FIPS mode?",
    "How do I find my clusterID?",
    "do you recommend using DeploymentConfig?",
    "give me sample deployment yaml that uses MongoDB image",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Utility script for querying RAG database"
    )
    parser.add_argument(
        "-p",
        "--db-path",
        required=False,
        help="path to the vector db",
        default="./vector_db/ocp_product_docs/4.15",
    )
    parser.add_argument(
        "-x",
        "--product-index",
        required=False,
        help="product index",
        default="ocp-product-docs-4_15",
    )
    parser.add_argument(
        "-m",
        "--model-path",
        required=False,
        help="path to the embedding model",
        default="./embeddings_model",
    )
    parser.add_argument(
        "-q",
        "--queries",
        nargs="+",
        default=QNA_QUERIES,
        help="queries to run",
    )
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        help="similarity_top_k",
        default=10,
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        help="Directory, where the retrieved data will stored as csv file",
        default=None,
    )
    args = parser.parse_args()

    os.environ["TRANSFORMERS_CACHE"] = args.model_path
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    embed_model = HuggingFaceEmbedding(
        model_name=args.model_path,
        # trust_remote_code=True,
    )

    Settings.llm = resolve_llm(None)
    Settings.embed_model = embed_model

    storage_context = StorageContext.from_defaults(
        vector_store=FaissVectorStore.from_persist_dir(args.db_path),
        persist_dir=args.db_path,
    )
    vector_index = load_index_from_storage(
        storage_context=storage_context,
        index_id=args.product_index,
    )
    result = defaultdict(list)

    for query in args.queries:
        print(f"Getting chunks for the query: {query}")
        retriever = vector_index.as_retriever(similarity_top_k=args.top_k)
        nodes = retriever.retrieve(query)

        for node in nodes:
            result["query"].append(query)
            result["score"].append(node.score)
            result["title"].append(node.metadata.get("doc_title", None))
            result["docs_url"].append(node.metadata.get("doc_url", None))
            result["doc_text"].append(node.get_text())

    result_df = DataFrame.from_dict(result)

    # Save result
    result_dir = os.path.join(
        (args.output_dir or os.path.dirname(__file__)), "eval_result"
    )
    os.makedirs(result_dir, exist_ok=True)
    result_df.to_csv(os.path.join(result_dir, "retrieved_chunks.csv"), index=False)
