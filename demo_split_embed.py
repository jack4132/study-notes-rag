"""
demo_split_embed.py
--------------------
Tiny standalone demo to SEE how splitting + embedding work.
No PDF, no database - just plain text, so it's easy to inspect.

Run:
    python demo_split_embed.py
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# ---- Step 1: Some sample text (pretend this is one page of notes) ----
sample_text = """
React Native lets you build mobile apps using JavaScript and React.
It renders real native UI components, not a webview.

Incremental adoption means you don't rewrite your whole app at once.
Instead, you introduce React Native screen by screen, inside your
existing native iOS or Android app.

This approach reduces risk because you can test one screen,
learn from it, then migrate the next screen later.
"""

print("=" * 60)
print("ORIGINAL TEXT")
print("=" * 60)
print(sample_text)
print(f"\nLength: {len(sample_text)} characters\n")


# ---- Step 2: SPLITTING ----
# This just cuts the text into smaller pieces. No AI involved here -
# pure string logic. It tries to cut on paragraph breaks first, then
# sentences, then words - whatever keeps chunk_size under the limit.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=120,   # small on purpose, so you SEE multiple chunks
    chunk_overlap=20, # each chunk repeats last 20 chars of previous one
)

chunks = splitter.split_text(sample_text)

print("=" * 60)
print(f"SPLITTING RESULT: {len(chunks)} chunks")
print("=" * 60)
for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i+1} ({len(chunk)} chars) ---")
    print(chunk)


# ---- Step 3: EMBEDDING ----
# This calls OpenAI's embedding model. Each chunk of TEXT gets turned
# into a LIST OF NUMBERS (a vector). Similar meaning = similar numbers.
print("\n" + "=" * 60)
print("EMBEDDING RESULT")
print("=" * 60)

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
vectors = embeddings_model.embed_documents(chunks)

for i, vector in enumerate(vectors):
    print(f"\nChunk {i+1} -> vector of {len(vector)} numbers")
    print(f"First 8 numbers: {vector[:8]}")

# ---- Step 4: PROVE similarity actually works ----
# Embed a question, compare it to each chunk's vector using cosine
# similarity (a math formula: 1.0 = identical meaning, 0 = unrelated).
import numpy as np

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

question = "Why is incremental adoption less risky?"
question_vector = embeddings_model.embed_query(question)

print("\n" + "=" * 60)
print(f"QUESTION: {question}")
print("=" * 60)

scores = [(cosine_similarity(question_vector, v), chunks[i])
          for i, v in enumerate(vectors)]
scores.sort(reverse=True, key=lambda x: x[0])

for score, chunk in scores:
    print(f"\nSimilarity: {score:.4f}")
    print(chunk)
