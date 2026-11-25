"""
main.py
-------
FastAPI backend wrapping our existing RAG logic (rag_chain.py) and
ingestion logic (ingest.py) so frontends (Web, Android) can call it
over HTTP.

Run:
    uvicorn main:app --reload --port 8000

Test:
    http://localhost:8000/docs   (interactive API docs)
"""

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pathlib import Path
from typing import Optional
import shutil

from rag_chain import ask, ask_stream
from list_sources import list_sources
from ingest import ingest_single_file

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Study Notes Assistant API")

# Allow frontend (React dev server, Android emulator) to call this API.
# In production, replace "*" with your actual frontend URL(s).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- Request/response shapes (Pydantic models) ----
class ChatRequest(BaseModel):
    message: str
    source: Optional[str] = None  # optional filename to scope retrieval to


class ChatResponse(BaseModel):
    answer: str


# ---- Endpoints ----

@app.get("/")
def root():
    return {"status": "ok", "service": "Study Notes Assistant API"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Takes a user question, runs it through the RAG chain (retrieval +
    citation-enforcing prompt + LLM), returns the answer.
    Non-streaming version - full answer returned in one response.
    """
    answer = ask(request.message, source=request.source)
    return ChatResponse(answer=answer)


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Same as /api/chat but streams the answer token-by-token using
    Server-Sent Events (SSE), so the frontend can show a live "typing"
    effect instead of waiting for the full answer.
    """
    async def event_generator():
        async for token in ask_stream(request.message, source=request.source):
            # SSE wire format: each message starts with "data: " and
            # ends with two newlines. We escape real newlines in the
            # token so one SSE message isn't accidentally split.
            safe_token = token.replace("\n", "\\n")
            yield f"data: {safe_token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/documents")
def list_documents():
    """
    Direct metadata query - lists every unique document currently in
    ChromaDB, with chunk counts. NOT routed through the LLM/RAG chain,
    since "how many/which documents" is a database question, not a
    content question (see our earlier debugging notes on this).
    """
    sources = list_sources()
    return {
        "total_chunks": sum(sources.values()),
        "documents": [
            {"filename": name, "chunk_count": count}
            for name, count in sources.items()
        ],
    }


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Accepts a PDF file upload, saves it to data/, runs ingestion
    (split + embed + store in ChromaDB), returns how many chunks were
    created. This is what powers "upload a document" in the frontend.
    """
    if not file.filename.lower().endswith(".pdf"):
        return {"error": "Only PDF files are supported right now."}

    save_path = DATA_DIR / file.filename
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    chunk_count = ingest_single_file(str(save_path))

    return {
        "filename": file.filename,
        "chunks_created": chunk_count,
        "status": "ingested",
    }
