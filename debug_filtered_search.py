"""
debug_filtered_search.py
-------------------------
Shows how to restrict retrieval to ONE specific document using metadata
filtering, instead of plain semantic search across everything.
"""

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

# Without filter - searches ALL documents (what we did before)
print("--- WITHOUT filter (searches everything) ---")
results = vectorstore.similarity_search("Kotlin concepts", k=3)
for doc in results:
    print(doc.metadata.get("source"))

# With filter - restrict to ONLY Day+3+.pdf
print("\n--- WITH filter (source = Day+3+.pdf only) ---")
results_filtered = vectorstore.similarity_search(
    "Kotlin concepts",
    k=3,
    filter={"source": "Day+3+.pdf"},
)
for doc in results_filtered:
    print(doc.metadata.get("source"), "-", doc.page_content[:80])
