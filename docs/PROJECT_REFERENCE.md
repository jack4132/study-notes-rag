# Project Reference — Study Notes Assistant with Citations

Master reference doc. Explains tech stack, architecture, every file, every
function, models used, and how data flows end-to-end.

---

## 1. Tech Stack Summary

| Layer | Technology | Why |
|---|---|---|
| Backend framework | FastAPI (Python) | Async-native, fast, auto-docs, handles SSE streaming well |
| Server runner | Uvicorn | ASGI server that actually runs the FastAPI app |
| Orchestration | LangChain (LCEL) | Chains retrieval + prompt + LLM together declaratively |
| LLM (generation) | OpenAI `gpt-4o-mini` | Cheap, fast, good enough quality for citation-grounded Q&A |
| Embeddings | OpenAI `text-embedding-3-small` | Converts text chunks into vectors for semantic search |
| Vector database | ChromaDB | Local, persistent, stores chunks + their embeddings + metadata |
| PDF parsing | `pypdf` (via `PyPDFLoader`) | Extracts text + page numbers from PDF files |
| Text splitting | `RecursiveCharacterTextSplitter` | Breaks long text into smaller overlapping chunks |
| Validation | Pydantic | Defines/validates API request & response shapes |
| Observability (planned) | LangSmith | Traces every chain run, token usage, latency |
| Frontend (planned) | React (Web) + Kotlin/Jetpack Compose (Android) | Two clients, same backend API |

---

## 2. High-Level Architecture

```
┌─────────────┐     ┌─────────────┐
│  Web (React) │     │Android (Kotlin)│     <- Phase 4, not built yet
└──────┬──────┘     └──────┬──────┘
       │                   │
       └─────────┬─────────┘
                  │  HTTP (JSON / SSE)
                  ▼
         ┌─────────────────┐
         │   FastAPI        │   main.py
         │   (REST API)     │
         └────────┬─────────┘
                   │
       ┌───────────┴────────────┐
       ▼                        ▼
┌─────────────┐         ┌──────────────┐
│ rag_chain.py │         │ ingest.py     │
│ (answer Qs)  │         │ (add new docs)│
└──────┬──────┘         └───────┬──────┘
       │                        │
       └───────────┬────────────┘
                    ▼
           ┌─────────────────┐
           │    ChromaDB      │   chroma_db/ (persisted on disk)
           │ (vectors + text  │
           │  + metadata)     │
           └─────────────────┘
                    ▲
                    │ embeddings
           ┌─────────────────┐
           │  OpenAI API      │
           │ (embeddings +    │
           │  chat completion)│
           └─────────────────┘
```

---

## 3. File-by-file Reference

### `ingest.py` — turns PDFs into searchable vectors

| Function | What it does |
|---|---|
| `clean_page_text(text)` | Strips repeated page footer/header boilerplate (e.g. book title repeated on every page) using regex, before chunking — prevents noise chunks from polluting search results |
| `load_documents()` | Loads every PDF in `data/` folder via `PyPDFLoader`, one `Document` object per page, sets `metadata["source"]` to filename, cleans text |
| `split_documents(docs)` | Uses `RecursiveCharacterTextSplitter` (chunk_size=500, overlap=50) to break pages into smaller chunks. Drops chunks under 30 chars (leftover noise) |
| `embed_and_store(chunks)` | Calls OpenAI's `text-embedding-3-small` to embed every chunk, stores in ChromaDB at `chroma_db/` (persisted to disk) |
| `main()` | Runs all 3 steps in order. Entry point when run as `python ingest.py` |

**Key concept:** this file does the one-time (per-document) heavy lifting —
splitting + embedding. Run manually right now; later, FastAPI's
`/api/upload` endpoint will call this logic automatically when a user
uploads a file.

---

### `rag_chain.py` — core Q&A logic

| Function/object | What it does |
|---|---|
| `vectorstore` | Connects to the EXISTING ChromaDB (doesn't create new, just reads what `ingest.py` built) |
| `retriever` | Wraps vectorstore into a retriever object, configured to return top 3 (`TOP_K`) most similar chunks per query |
| `PROMPT_TEMPLATE` | The instructions sent to the LLM every time — forces: (1) answer only from context, (2) cite sources in `[source: file, page X]` format, (3) distinguish Factual vs Judgment-type questions, (4) say "Not found in notes" if nothing relevant exists |
| `llm` | `ChatOpenAI(model="gpt-4o-mini", temperature=0)` — temperature=0 means deterministic/non-random output, good for factual Q&A |
| `format_docs(docs)` | Converts retrieved `Document` objects into one text blob, tagging each chunk with its source+page so the LLM can literally copy that into its citation |
| `rag_chain` | The actual LCEL chain: retriever+format → prompt → llm → output parser, built with `|` pipe operators |
| `ask(question)` | Public sync function — runs the full chain, returns complete answer string (blocking, waits for full response) |
| `ask_stream(question)` | Public async generator — same chain but streams tokens one at a time via `.astream()`, used for SSE endpoint |

**Key concept:** this file never touches the file system or does
ingestion — it only READS what's already in ChromaDB and answers
questions about it.

---

### `list_sources.py` — document inventory (metadata-only, no LLM)

| Function | What it does |
|---|---|
| `list_sources()` | Pulls ALL stored metadata directly from ChromaDB (`vectorstore.get()`), counts chunks per unique `source` filename. Returns a plain dict `{filename: chunk_count}` |
| `if __name__ == "__main__"` block | When run directly (`python list_sources.py`), prints a human-readable report |

**Why this is separate from `rag_chain.py`:** listing/counting documents
is NOT a semantic search problem (we proved this through testing — "how
many PDFs" type questions fail when routed through the LLM/retriever,
since there's no single chunk of text that "is" the answer to a counting
question). This file does a direct database query instead — always
accurate, no LLM involved, instant.

---

### `main.py` — FastAPI backend, exposes everything over HTTP

| Endpoint | Method | What it does |
|---|---|---|
| `/` | GET | Health check, confirms server is running |
| `/api/chat` | POST | Takes `{"message": "..."}`, calls `rag_chain.ask()`, returns `{"answer": "..."}` — full response, waits for completion |
| `/api/chat/stream` | POST | Same question input, but returns SSE stream — calls `rag_chain.ask_stream()`, wraps each token into `data: ...\n\n` format, sends `data: [DONE]` at the end |
| `/api/documents` | GET | Calls `list_sources()`, returns document list + chunk counts as JSON — powers future "document picker" UI |

**Middleware:** `CORSMiddleware` added so browser-based frontends (React
dev server) aren't blocked by cross-origin restrictions.

**Pydantic models:** `ChatRequest` (expects `message: str`), `ChatResponse`
(returns `answer: str`) — define and validate the shape of data going
in/out of `/api/chat`.

---

### Debug/demo scripts (not part of production app, used for learning/testing)

| File | Purpose |
|---|---|
| `demo_split_embed.py` | Standalone teaching demo — shows splitting + embedding + manual cosine similarity math on tiny sample text, no DB involved |
| `debug_retrieve.py` | Quick script to test what chunks get retrieved for a given query, without going through the full chain/LLM |
| `check_count.py` | Prints total chunk count in ChromaDB |
| `debug_filtered_search.py` | Demonstrates metadata filtering (`filter={"source": "..."}`) to restrict search to one specific document |

---

## 4. Models Used — Exact Roles

| Model | Provider | Used in | Role |
|---|---|---|---|
| `text-embedding-3-small` | OpenAI | `ingest.py`, `rag_chain.py` | Converts text (chunks AND user questions) into vectors for semantic similarity search. NOT used for generating answers. |
| `gpt-4o-mini` | OpenAI | `rag_chain.py` | Reads retrieved context + question, generates the final written answer with citations. The only model that actually "writes" text. |

No model does the chunking/splitting — that's pure algorithmic string
processing (`RecursiveCharacterTextSplitter`), zero AI involved, zero API
cost, zero network call.

---

## 5. Data Flow — Two Scenarios

### Scenario A: Ingesting a new document
```
PDF file → PyPDFLoader (extract text per page) → clean_page_text()
(remove boilerplate) → RecursiveCharacterTextSplitter (break into
chunks) → OpenAI text-embedding-3-small (chunk text → vector) →
ChromaDB (store vector + text + metadata together)
```

### Scenario B: Answering a question
```
User question → OpenAI text-embedding-3-small (question text → vector)
→ ChromaDB similarity search (compare question vector against ALL
stored chunk vectors, cosine similarity, return top 3 closest) →
format_docs() (tag chunks with source/page) → PROMPT_TEMPLATE (inject
context + question) → gpt-4o-mini (generate answer, streamed or
complete) → StrOutputParser (extract plain text) → returned to
whatever called it (CLI, or FastAPI endpoint)
```

---

## 6. Related Docs
- `FASTAPI_GUIDE.md` — FastAPI fundamentals explained step by step
- `STREAMING_DEEP_DIVE.md` — exactly how `.astream()` works internally,
  SSE vs webhooks explained
- `PLAN.md` — full phased build plan, what's done vs pending
