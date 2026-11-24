# Study Notes Assistant with Citations — Build Plan

**Goal:** RAG app that answers questions from your PDF study notes, always citing
source (file + page), with streaming chat UI. **Two frontends: Web (React)
and Android (Kotlin native), both built together, both talking to the same
FastAPI backend.**

**Stack:**
- Backend: FastAPI (Python)
- Frontend (Web): React (JS), Vite
- Frontend (Android): Kotlin, Jetpack Compose, Retrofit/OkHttp, MVVM
- Orchestration: LangChain
- LLM: OpenAI `gpt-4o-mini`
- Embeddings: OpenAI `text-embedding-3-small`
- Vector DB: ChromaDB (local, persistent)
- Observability: LangSmith

**Platform behavior (confirmed earlier, still applies):**
| Platform | Upload behavior | Persistence |
|---|---|---|
| Web | Plain one-time upload → analyze → Q&A | Ephemeral, session-based |
| Android | Upload multiple PDFs over time | Persistent personal knowledge base |

---

## Phase 0 — Setup
- [ ] Create Python virtual environment
- [ ] Install core deps: `langchain`, `langchain-openai`, `langchain-chroma`,
      `chromadb`, `openai`, `python-dotenv`, `pypdf`, `fastapi`, `uvicorn`
- [ ] Create `.env` with `OPENAI_API_KEY`
- [ ] Add `.gitignore` (`.env`, `__pycache__/`, `venv/`, `chroma_db/`)
- [ ] Gather sample PDF study notes (1-2 files) to test with
- [ ] (Optional) Create OpenAI project + set usage limit in billing settings

Folder structure target:
```
openAI/
├── .env
├── .gitignore
├── data/                 # your PDF notes go here
├── ingest.py             # Phase 1
├── rag_chain.py          # Phase 2
├── main.py               # Phase 3 - FastAPI app
├── frontend/             # Phase 4 - React app
└── chroma_db/            # auto-created, persisted vectors
```

---

## Phase 1 — Ingestion Pipeline (`ingest.py`)
**Concept learned:** chunking, embeddings, vector storage.

- [ ] Load PDF(s) from `data/` folder using `PyPDFLoader`
- [ ] Split text into chunks: `RecursiveCharacterTextSplitter`
      (chunk_size=500, chunk_overlap=50 — tune later)
- [ ] Attach metadata to every chunk: `{source: filename, page: page_num}`
      (critical — needed for citations in Phase 2)
- [ ] Generate embeddings with `text-embedding-3-small`
- [ ] Store chunks + embeddings in local ChromaDB (`chroma_db/` dir, persistent)
- [ ] Run script once manually, verify chunk count + no errors

**Test:** open Python shell, query Chroma directly, confirm top results
make sense before moving on.

---

## Phase 2 — Retrieval Chain (`rag_chain.py`)
**Concept learned:** retriever, prompt templates, context injection, LCEL chains.

- [ ] Load existing ChromaDB (don't re-ingest)
- [ ] Create retriever: `vectorstore.as_retriever(search_kwargs={"k": 3})`
- [ ] Write citation-enforcing prompt template, e.g.:
      "Answer ONLY using the context below. Cite source as [filename, page X]
      after each claim. If answer not in context, say 'Not found in notes.'"
- [ ] Build chain with LCEL: retriever → format context → prompt → LLM → parser
- [ ] Write `ask(query: str) -> str` function wrapping the chain
- [ ] Test with 5-10 manual questions in terminal, check citations appear
      and are accurate (no hallucinated page numbers)

**Test:** run function directly from CLI/script before touching API layer.

---

## Phase 3 — FastAPI Backend (`main.py`)
**Concept learned:** REST API, async, SSE streaming.

- [ ] Create FastAPI app, enable CORS (for React dev server origin)
- [ ] Build `POST /api/chat` endpoint accepting `{ "message": "..." }`
- [ ] First version: non-streaming, return full JSON answer — confirm
      request/response cycle works via `curl` or Postman
- [ ] Convert to streaming: use `StreamingResponse` with
      `media_type="text/event-stream"`, yield LLM tokens as they arrive
      (LangChain `.astream()` on the chain)
- [ ] Format each streamed chunk as SSE: `data: {token}\n\n`
- [ ] Test streaming with `curl -N http://localhost:8000/api/chat`

---

## Phase 4 — Dual Frontend: Web (React) + Android (Kotlin)
**Concept learned:** building two clients against one API, platform-specific
networking/streaming, shared backend contract.

**4A — Web (React)**
- [ ] Scaffold React app (Vite)
- [ ] Build chat UI: message list + input box
- [ ] `fetch()` call to `/api/chat`, read SSE stream via `ReadableStream`,
      append tokens to chat bubble live
- [ ] Simple file upload UI → `/api/upload` (plain one-time upload, no
      persistent library, matches web's ephemeral behavior)
- [ ] Display citations under each answer

**4B — Android (Kotlin)**
- [ ] Scaffold Android project (Kotlin + Jetpack Compose)
- [ ] Add `INTERNET` permission in `AndroidManifest.xml`
- [ ] Retrofit + OkHttp setup for API calls to FastAPI backend
- [ ] Chat UI in Compose (MVVM: ViewModel holds state, Composable observes)
- [ ] Call `/api/chat`, consume SSE stream via OkHttp response body reading
- [ ] PDF upload flow: `ActivityResultContracts` file picker → multipart
      upload to `/api/upload` (persistent library behavior — ties to user_id)
- [ ] Display citations in chat bubble UI
- [ ] On emulator, use `10.0.2.2` instead of `localhost` to reach backend

**Shared backend note:** both clients hit same `/api/chat` and `/api/upload`
endpoints from Phase 3 — no backend duplication, just two sets of UI code.

---

## Phase 5 — Citations Hardening + Observability (applies to both clients)
**Concept learned:** prompt engineering for grounding, tracing/debugging AI apps.

- [ ] Stress-test prompt: ask questions NOT in your notes, confirm it says
      "Not found" instead of hallucinating — verify via both Web and Android
- [ ] Ask questions where answer spans multiple chunks, confirm multiple
      citations shown correctly on both UIs
- [ ] Sign up / use LangSmith, set `LANGCHAIN_TRACING_V2=true` +
      `LANGCHAIN_API_KEY` in `.env`
- [ ] Run chat queries from both Web and Android, view traces in LangSmith
      dashboard — confirm both platforms' requests show up correctly
- [ ] Inspect: retrieved chunks per query, token usage, latency per step

---

## Future Phase — Meiteilon (Manipuri) Support
- [ ] Decide script: Meitei Mayek vs Bengali-based transliteration
- [ ] Test gpt-4o-mini's native fluency in Meiteilon (quick manual test first)
- [ ] Likely approach: keep embeddings/retrieval in English (reliable),
      add instruction in final prompt to respond in Meiteilon (generation
      step only) — avoids weak embedding quality for low-resource language
- [ ] If source notes are uploaded in Meitei Mayek script directly, test
      embedding quality on that script specifically before committing
- [ ] Only start this after core English RAG app fully works end-to-end

---

## Phase 6 — Session/User Identity & Data Isolation
**Concept learned:** session vs user identity, metadata filtering, data isolation.

(Builds on Phase 4's dual-client split — this phase makes the Web
ephemeral-session and Android persistent-library behaviors actually real
on the backend.)

- [ ] Same FastAPI backend serves both (no need for separate backends)
- [ ] Add `session_id` (web, random/temp, no login) vs `user_id` (Android,
      tied to real account/login) as metadata on every ingested chunk
- [ ] Update retriever calls to filter by `session_id` or `user_id` so
      users/sessions never see each other's documents
- [ ] Web: auto-expire/delete session data after N hours (no long-term storage)
- [ ] Android: needs real auth (login/signup) + separate user-accounts DB
      (e.g. SQLite/Postgres) — ChromaDB only stores vectors, not identities
- [ ] Add document management: list uploaded docs per user, delete a doc
      (removes its chunks from Chroma using stored metadata)

---

## Phase 7 — Camera/Image OCR Input (Android app only)
**Concept learned:** vision models, OCR pipeline, image-to-text grounding.

**Scope note:** camera capture is Android-only. Web app stays plain file
upload (no camera feature).

- [ ] Add image capture flow in Android app (camera → photo file)
- [ ] Send photo to backend new endpoint `POST /api/upload-image`
- [ ] Extract text from image using vision-capable model (`gpt-4o`, not
      mini — mini has no vision support) OR OCR engine (Tesseract/Cloud Vision)
- [ ] Feed extracted text into existing `split_documents()` +
      `embed_and_store()` — no changes needed downstream, reuses Phase 1 code
- [ ] Adjust citation format for photos — no "page number", use filename/
      timestamp instead (e.g. `[source: photo_2024_11_21.jpg]`)
- [ ] Consider image quality guidance in UI (lighting/angle tips) since
      garbage photo in = garbage text extracted = garbage answers
- [ ] Note cost difference: vision API calls cost more per use than
      plain text embedding — factor into usage limits if scaling up

---

## Stretch goals (after core works)
- [ ] Multi-file upload UI (drag & drop PDFs into app, trigger ingestion)
- [ ] Highlight exact matching sentence in source doc, not just filename
- [ ] Conversation memory (follow-up questions referencing earlier answers)
- [ ] Swap ChromaDB → pgvector or Pinecone (cloud-scale version)
- [ ] Add auth so it's a proper multi-user app

---

## Key concepts checklist (what you're learning, mapped to phases)
| Concept | Phase |
|---|---|
| Chunking strategy | 1 |
| Embeddings | 1 |
| Vector similarity search | 1, 2 |
| Metadata-driven citations | 1, 2 |
| Prompt engineering / grounding | 2, 5 |
| LCEL chains | 2 |
| REST API design | 3 |
| Async Python | 3 |
| SSE streaming | 3, 4 |
| Frontend stream consumption | 4 |
| LLM observability/tracing | 5 |
