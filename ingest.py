"""
ingest.py
---------
Purpose: Read PDF study notes, break into chunks, convert chunks into
embeddings (vectors), store them in a local ChromaDB so we can later
search them semantically.

Exposes ingest_single_file() for use by FastAPI's /api/upload endpoint,
and can also be run standalone to bulk-ingest everything in data/:
    python ingest.py
"""

import re
from collections import Counter
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from openai_clients import EMBEDDING_MODEL

# ---- Config ----
DATA_DIR = Path("data")
CHROMA_DIR = "chroma_db"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def remove_repeated_boilerplate(pages):
    """
    Auto-detect lines that repeat across most pages of THIS document
    (e.g. book title in footer, page numbers) and strip them out.
    Generic - works for any PDF, not hardcoded to one book's title.
    """
    if len(pages) < 3:
        return pages  # too few pages to reliably detect repetition

    line_counts = Counter()
    for page in pages:
        lines = {line.strip() for line in page.page_content.split("\n") if line.strip()}
        line_counts.update(lines)

    # A line appearing on >50% of pages is almost certainly boilerplate
    threshold = len(pages) * 0.5
    boilerplate_lines = {line for line, count in line_counts.items() if count > threshold}

    for page in pages:
        cleaned_lines = [
            line for line in page.page_content.split("\n")
            if line.strip() not in boilerplate_lines
        ]
        page.page_content = "\n".join(cleaned_lines)
        # Also strip standalone page-number-only lines
        page.page_content = re.sub(r"(?m)^\s*\d+\s*$", "", page.page_content).strip()

    return pages


def load_single_pdf(file_path: str):
    """Load one PDF, clean boilerplate, tag metadata with filename."""
    file_path = Path(file_path)
    loader = PyPDFLoader(str(file_path))
    docs = loader.load()  # one Document object per page

    for doc in docs:
        doc.metadata["source"] = file_path.name

    docs = remove_repeated_boilerplate(docs)
    return docs


def load_documents():
    """Load every PDF in data/ folder (bulk mode for standalone script)."""
    all_docs = []
    pdf_files = list(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {DATA_DIR}/")

    for pdf_path in pdf_files:
        docs = load_single_pdf(str(pdf_path))
        all_docs.extend(docs)
        print(f"Loaded {len(docs)} pages from {pdf_path.name}")

    return all_docs


def split_documents(docs):
    """Break page-level documents into smaller overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)

    # Drop near-empty chunks (leftover noise after cleaning, e.g. < 30 chars)
    chunks = [c for c in chunks if len(c.page_content.strip()) >= 30]

    print(f"Split into {len(chunks)} chunks "
          f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


def embed_and_store(chunks):
    """Generate embeddings for each chunk and persist in ChromaDB."""
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    print(f"Stored {len(chunks)} chunks in ChromaDB at '{CHROMA_DIR}/'")
    return vectorstore


def ingest_single_file(file_path: str) -> int:
    """
    Public function for API use: ingest ONE file (already saved to disk),
    returns number of chunks created. Used by /api/upload.
    """
    docs = load_single_pdf(file_path)
    chunks = split_documents(docs)
    embed_and_store(chunks)
    return len(chunks)


def main():
    docs = load_documents()
    chunks = split_documents(docs)
    embed_and_store(chunks)
    print("\nIngestion complete. You can now run rag_chain.py to query your notes.")


if __name__ == "__main__":
    main()
