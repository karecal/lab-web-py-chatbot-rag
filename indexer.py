"""
Indexer — reads docs/*.txt, chunks them, creates embeddings via LM Studio,
and stores everything in ChromaDB with metadata.
"""

import os
import sys
import glob
import time
import tiktoken
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
API_KEY = os.getenv("LM_STUDIO_API_KEY", "lm-studio")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5-GGUF")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
DOCS_PATH = os.getenv("DOCS_PATH", "./docs")

CHUNK_SIZE = 500   # max tokens per chunk
CHUNK_OVERLAP = 50  # token overlap between consecutive chunks
COLLECTION_NAME = "technova_docs"

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


def load_documents(docs_path: str) -> list[dict]:
    """Return list of {filename, content} dicts for every .txt in docs_path."""
    pattern = os.path.join(docs_path, "*.txt")
    files = glob.glob(pattern)
    if not files:
        print(f"[ERROR] No .txt files found in {docs_path}")
        sys.exit(1)

    documents = []
    for filepath in sorted(files):
        with open(filepath, encoding="utf-8") as f:
            content = f.read().strip()
        if content:
            documents.append({"filename": os.path.basename(filepath), "content": content})
    return documents


def chunk_text(text: str, max_tokens: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping token-based chunks."""
    # cl100k_base is the tiktoken encoding used by most modern models
    try:
        enc = tiktoken.encoding_for_model("text-embedding-ada-002")
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")

    tokens = enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(enc.decode(chunk_tokens))
        if end == len(tokens):
            break
        start = end - overlap
    return chunks


def create_embedding(text: str) -> list[float]:
    """Create an embedding vector for the given text via LM Studio."""
    response = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
    return response.data[0].embedding


def index_documents():
    print("=" * 60)
    print("TechNova RAG Indexer")
    print("=" * 60)

    # Load documents
    print(f"\n[1/4] Loading documents from '{DOCS_PATH}'...")
    documents = load_documents(DOCS_PATH)
    print(f"      Found {len(documents)} documents: {[d['filename'] for d in documents]}")

    # Chunk documents
    print("\n[2/4] Chunking documents...")
    all_chunks = []
    for doc in documents:
        chunks = chunk_text(doc["content"])
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "filename": doc["filename"],
                "chunk_id": i,
                "doc_chunk_id": f"{doc['filename']}::chunk_{i}",
            })
        print(f"      {doc['filename']}: {len(chunks)} chunks")

    total_chunks = len(all_chunks)
    print(f"      Total: {total_chunks} chunks across {len(documents)} documents")

    # Estimate tokens
    try:
        enc = tiktoken.encoding_for_model("text-embedding-ada-002")
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    total_tokens = sum(len(enc.encode(c["text"])) for c in all_chunks)
    # text-embedding-3-small pricing is ~$0.02/1M tokens; local = $0
    estimated_cost_openai = (total_tokens / 1_000_000) * 0.02
    print(f"      Total tokens: {total_tokens:,}")
    print(f"      Estimated cost if using OpenAI API: ${estimated_cost_openai:.4f}  (local LM Studio = $0.00)")

    # Connect to ChromaDB
    print(f"\n[3/4] Connecting to ChromaDB at '{CHROMA_DB_PATH}'...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # Delete existing collection to allow full re-index
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
        print(f"      Deleted existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Create embeddings and store
    print(f"\n[4/4] Creating embeddings and storing in ChromaDB...")
    print(f"      Using model: {EMBEDDING_MODEL}")
    print(f"      This may take a while for large document sets...\n")

    batch_size = 10
    for i in range(0, total_chunks, batch_size):
        batch = all_chunks[i : i + batch_size]
        texts = [c["text"] for c in batch]
        ids = [c["doc_chunk_id"] for c in batch]
        metadatas = [{"filename": c["filename"], "chunk_id": c["chunk_id"]} for c in batch]

        embeddings = []
        for text in texts:
            embedding = create_embedding(text)
            embeddings.append(embedding)
            time.sleep(0.05)  # gentle rate limiting for local server

        collection.add(documents=texts, embeddings=embeddings, ids=ids, metadatas=metadatas)

        processed = min(i + batch_size, total_chunks)
        print(f"      [{processed}/{total_chunks}] chunks indexed...", end="\r")

    print(f"\n\n{'=' * 60}")
    print("Indexing complete!")
    print(f"  Documents : {len(documents)}")
    print(f"  Chunks    : {total_chunks}")
    print(f"  Tokens    : {total_tokens:,}")
    print(f"  Collection: {COLLECTION_NAME}")
    print(f"  DB path   : {CHROMA_DB_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    index_documents()
