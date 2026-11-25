"""
rag_chain.py
------------
Purpose: Core RAG logic.
Takes a user question -> finds relevant chunks in ChromaDB -> builds a
prompt that forces the LLM to answer ONLY from those chunks, with citations
-> sends to OpenAI -> returns answer.

Run directly to test from terminal:
    python rag_chain.py
"""

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from typing import Optional
from openai_clients import get_http_client, get_async_http_client, EMBEDDING_MODEL, CHAT_MODEL, get_api_key

CHROMA_DIR = "chroma_db"
TOP_K = 3  # how many chunks to retrieve per question

# ---- 1. Load existing vector store (built by ingest.py) ----
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=get_api_key(), http_client=get_http_client())
vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})


def get_retriever(source: Optional[str] = None):
    """
    Returns a retriever scoped to ONE document if `source` (filename) is
    given, otherwise searches across all documents. This is how the
    frontend's document picker scopes Q&A to a specific uploaded file.
    """
    search_kwargs = {"k": TOP_K}
    if source:
        search_kwargs["filter"] = {"source": source}
    return vectorstore.as_retriever(search_kwargs=search_kwargs)

# ---- 2. Prompt template that forces grounded answers + citations ----
PROMPT_TEMPLATE = """You are a study assistant. Answer using ONLY the
context below. Do not use outside knowledge.

For every factual claim, cite the source in this exact format:
[source: {{filename}}, page {{page}}]

There are two types of questions:

1. FACTUAL questions (the context directly states the answer):
   Answer normally with citations.

2. JUDGMENT/INTERPRETIVE questions (the context has relevant facts, but
   answering requires inference, opinion, or a conclusion not explicitly
   stated):
   - First state the relevant FACTS from the context, with citations.
   - Then clearly separate your interpretation using this exact label:
     "Judgment (based on above facts, not explicitly stated in notes):"
     followed by your reasoning.
   - End with: "Final decision is up to you."

If the context has NO relevant facts at all for the question, respond
exactly with: "Not found in notes." Do not add this phrase if you already
provided an answer above.

Context:
{context}

Question: {question}

Answer (with citations):"""

prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

# ---- 3. LLM ----
llm = ChatOpenAI(
    model=CHAT_MODEL,
    temperature=0,
    api_key=get_api_key(),
    http_client=get_http_client(),
    http_async_client=get_async_http_client(),
)


def format_docs(docs):
    """Turn retrieved chunks into a single context string, each chunk
    tagged with its source + page so the LLM can cite correctly."""
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        formatted.append(f"[source: {source}, page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


# ---- 4. The chain (LCEL - LangChain Expression Language) ----
# Reads right to left conceptually: question -> retriever -> format ->
# prompt -> llm -> parse to plain string.
def build_chain(source: Optional[str] = None):
    """Builds the LCEL chain using a retriever scoped to `source` if given."""
    scoped_retriever = get_retriever(source)
    return (
        {"context": scoped_retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


def ask(question: str, source: Optional[str] = None) -> str:
    """Public function: ask a question, get a grounded, cited answer.
    Pass `source` (a filename) to scope retrieval to just that document."""
    chain = build_chain(source)
    return chain.invoke(question)


async def ask_stream(question: str, source: Optional[str] = None):
    """
    Async generator version - yields answer tokens one at a time as the
    LLM generates them, instead of waiting for the full answer.
    Used by the streaming API endpoint (SSE). Pass `source` to scope
    retrieval to just that document.
    """
    chain = build_chain(source)
    async for chunk in chain.astream(question):
        yield chunk


if __name__ == "__main__":
    print("Ask questions about your study notes. Type 'exit' to quit.\n")
    while True:
        q = input("Q: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        if not q:
            continue
        answer = ask(q)
        print(f"\nA: {answer}\n")
