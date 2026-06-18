"""Ingest markdown files from a local directory into Qdrant.

Uses FastEmbed with the same model as mcp-server-qdrant (all-MiniLM-L6-v2)
to ensure embedding dimensions match at query time.

Usage:
    python ingest.py --docs-path ./demo-docs --collection byok-demo
"""

import argparse
import hashlib
import sys
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
VECTOR_NAME = "fast-all-minilm-l6-v2"


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks by character count."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap
    return chunks


def generate_id(text: str) -> str:
    """Generate a deterministic hex ID from text content."""
    return hashlib.md5(text.encode()).hexdigest()[:16]


def load_documents(docs_path: Path) -> list[dict]:
    """Load all markdown and text files from a directory tree."""
    docs = []
    extensions = {".md", ".txt", ".text", ".rst"}
    for file_path in sorted(docs_path.rglob("*")):
        if file_path.suffix.lower() in extensions and file_path.is_file():
            content = file_path.read_text(encoding="utf-8", errors="replace")
            if content.strip():
                relative = file_path.relative_to(docs_path)
                docs.append({
                    "path": str(relative),
                    "content": content,
                    "title": extract_title(content, relative),
                })
    return docs


def extract_title(content: str, path: Path) -> str:
    """Extract title from first markdown heading or use filename."""
    for line in content.split("\n"):
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return path.stem.replace("-", " ").replace("_", " ").title()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into Qdrant for BYOK MCP POC")
    parser.add_argument("--docs-path", type=Path, required=True, help="Path to document directory")
    parser.add_argument("--collection", default="byok-demo", help="Qdrant collection name")
    parser.add_argument("--qdrant-url", default="http://localhost:6333", help="Qdrant server URL")
    args = parser.parse_args()

    if not args.docs_path.is_dir():
        print(f"Error: {args.docs_path} is not a directory", file=sys.stderr)
        sys.exit(1)

    print(f"Loading documents from {args.docs_path}...")
    docs = load_documents(args.docs_path)
    if not docs:
        print("No documents found.", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(docs)} documents")

    all_chunks = []
    for doc in docs:
        chunks = chunk_text(doc["content"])
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "metadata": {
                    "source": doc["path"],
                    "title": doc["title"],
                    "chunk_index": i,
                },
            })
    print(f"Created {len(all_chunks)} chunks")

    print(f"Generating embeddings with {EMBEDDING_MODEL}...")
    try:
        from fastembed import TextEmbedding
    except ImportError:
        print("Error: fastembed not installed. Run: pip install fastembed", file=sys.stderr)
        sys.exit(1)

    model = TextEmbedding(model_name=EMBEDDING_MODEL)
    texts = [c["text"] for c in all_chunks]
    embeddings = list(model.embed(texts))
    print(f"Generated {len(embeddings)} embeddings (dim={len(embeddings[0])})")

    print(f"Connecting to Qdrant at {args.qdrant_url}...")
    client = QdrantClient(url=args.qdrant_url)

    if client.collection_exists(args.collection):
        print(f"Dropping existing collection '{args.collection}'...")
        client.delete_collection(args.collection)

    client.create_collection(
        collection_name=args.collection,
        vectors_config={
            VECTOR_NAME: VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        },
    )

    points = [
        PointStruct(
            id=idx,
            vector={VECTOR_NAME: embedding.tolist()},
            payload={
                "document": all_chunks[idx]["text"],
                "metadata": all_chunks[idx]["metadata"],
            },
        )
        for idx, embedding in enumerate(embeddings)
    ]

    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(collection_name=args.collection, points=batch)
        print(f"  Uploaded {min(i + batch_size, len(points))}/{len(points)} points")

    info = client.get_collection(args.collection)
    print(f"\nDone! Collection '{args.collection}' has {info.points_count} points")
    print(f"Qdrant URL: {args.qdrant_url}")
    print(f"Collection: {args.collection}")
    print(f"Embedding model: {EMBEDDING_MODEL}")


if __name__ == "__main__":
    main()
