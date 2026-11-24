"""
list_sources.py
----------------
Shared vectorstore connection + a function to list every unique document
currently stored in ChromaDB. Can be run standalone (prints a report) or
imported by main.py (just uses the vectorstore/list_sources function).
"""

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

CHROMA_DIR = "chroma_db"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)


def list_sources():
    """Return {filename: chunk_count} for every document in the DB."""
    all_data = vectorstore.get()
    metadatas = all_data["metadatas"]

    sources = {}
    for meta in metadatas:
        source = meta.get("source", "unknown")
        sources[source] = sources.get(source, 0) + 1
    return sources


if __name__ == "__main__":
    sources = list_sources()
    total_chunks = sum(sources.values())
    print(f"Total chunks: {total_chunks}")
    print(f"Unique documents: {len(sources)}\n")
    for name, count in sorted(sources.items()):
        print(f"- {name}  ({count} chunks)")
