# FastAPI Basics — Explained Step by Step

This doc teaches FastAPI fundamentals using OUR actual project as the
example. Read this alongside `main.py`.

---

## 1. What is FastAPI, in one sentence

FastAPI is a Python framework for building web APIs — you write plain
Python functions, decorate them with routes (URLs), and it automatically
turns them into a working web server that accepts HTTP requests and
returns responses (usually JSON).

---

## 2. The absolute minimum FastAPI app

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello world"}
```

Breaking this down:
- `FastAPI()` — creates the "app" object. Everything (routes, middleware,
  config) attaches to this one object.
- `@app.get("/")` — a **decorator**. Means: "when someone sends an HTTP
  GET request to the URL path `/`, run the function below."
- The function just returns a Python dict — FastAPI automatically
  converts it to JSON for you. No manual serialization needed.

To run it:
```bash
uvicorn main:app --reload
```
- `main` = filename (`main.py`)
- `app` = the variable name of your FastAPI instance
- `--reload` = auto-restart server when you save code changes (dev only,
  remove in production)

---

## 3. HTTP methods you'll actually use

| Method | Decorator | Used for |
|---|---|---|
| GET | `@app.get(...)` | Fetching data (e.g. list documents) — no body, just reads |
| POST | `@app.post(...)` | Sending data to create/process something (e.g. chat message, file upload) |

We'll mainly use `POST` (sending chat messages, uploading files) and `GET`
(listing documents).

---

## 4. Request bodies — how data comes IN

When a frontend sends data (like a chat message), FastAPI needs to know
the expected **shape** of that data. This is done with **Pydantic models**
— plain Python classes that define fields + types.

```python
from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat")
def chat(request: ChatRequest):
    # request.message is now a validated string, guaranteed to exist
    return {"answer": f"You asked: {request.message}"}
```

What happens automatically here:
- FastAPI reads incoming JSON body: `{"message": "hello"}`
- Validates it matches `ChatRequest` shape (has a `message` field, it's a string)
- If frontend sends malformed data (missing field, wrong type), FastAPI
  auto-returns a clean 422 error — you don't write that validation
  yourself
- Inside the function, `request.message` is just a normal Python string,
  fully typed or your editor/IDE autocompletes

**This is why we don't just use plain dicts** — Pydantic models give you
free validation + auto-documentation (see point 8).

---

## 5. Async functions — why FastAPI uses `async def`

```python
@app.post("/api/chat")
async def chat(request: ChatRequest):
    answer = await some_slow_operation()
    return {"answer": answer}
```

- `async def` instead of `def` — tells Python this function can "pause"
  while waiting on slow things (like calling OpenAI's API, which takes
  time over the network) and let the server handle OTHER requests during
  that wait, instead of freezing up
- `await` — used when calling another async function, means "pause here
  until this finishes, but don't block everything else"
- **Why this matters for us:** every OpenAI API call (embedding, chat
  completion) takes real network time (hundreds of ms to seconds). If our
  server wasn't async, one slow request would freeze the WHOLE server for
  every other user. Async lets multiple users chat simultaneously without
  blocking each other.

You can mix `def` and `async def` in the same app — FastAPI handles both,
but `async def` is preferred for I/O-heavy work (API calls, file reads).

---

## 6. Path parameters and query parameters (for later reference)

```python
@app.get("/api/documents/{doc_id}")
def get_document(doc_id: str):
    # doc_id comes from the URL itself, e.g. /api/documents/day3
    ...

@app.get("/api/documents")
def list_documents(limit: int = 10):
    # limit comes from ?limit=5 in the URL, defaults to 10 if not given
    ...
```

- `{doc_id}` in the route path = **path parameter**, part of the URL
  structure itself
- Function arguments NOT in the path = **query parameters**, come from
  `?key=value` in the URL

We may use these later for things like `/api/documents/{filename}` to
delete a specific document.

---

## 7. Streaming responses (SSE) — needed for Phase 3B

```python
from fastapi.responses import StreamingResponse

@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    def generate():
        for chunk in some_generator():
            yield f"data: {chunk}\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream")
```

- Instead of returning one complete JSON response, we return a
  **generator** (a function using `yield` instead of `return`) — it
  produces small pieces of output one at a time
- `media_type="text/event-stream"` — tells the browser/client "this is
  SSE, a stream of events, not a normal JSON response"
- Format `data: {content}\n\n` is the required SSE wire format — each
  message must start with `data: ` and end with two newlines
- Frontend reads this stream piece by piece, appending text as it arrives
  — this is what gives the "typing" effect

We'll build this AFTER the simple non-streaming version works first.

---

## 8. Automatic interactive docs (free feature)

Once your server is running, FastAPI auto-generates a testable UI at:
```
http://localhost:8000/docs
```
This shows every endpoint, lets you fill in example requests and
actually test them in the browser — no need for curl/Postman always.
This works because of the Pydantic models (point 4) — FastAPI uses them
to know what fields/types to show in this UI.

---

## 9. CORS — why we need it

Your frontend (React on `localhost:5173` or Android emulator) runs on a
DIFFERENT origin than your backend (`localhost:8000`). Browsers block
cross-origin requests by default for security. We fix this with:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in production, list exact frontend URLs instead
    allow_methods=["*"],
    allow_headers=["*"],
)
```

- **Middleware** = code that runs on EVERY request/response, before/after
  your actual route functions — used for cross-cutting concerns like
  CORS, logging, auth checks
- Without this, your React app's `fetch()` calls would get silently
  blocked by the browser with a CORS error

---

## 10. Our project's planned endpoint structure

```
main.py
├── POST /api/chat          -> runs rag_chain.ask(), returns answer
├── POST /api/chat/stream   -> same but streams tokens via SSE
├── POST /api/upload        -> accepts PDF file, runs ingest logic
├── GET  /api/documents     -> lists all ingested docs (metadata query,
│                              NOT an LLM call - like list_sources.py)
└── DELETE /api/documents/{filename} -> removes a doc's chunks (later)
```

Each endpoint is just a thin wrapper that calls functions we ALREADY
wrote in `ingest.py` and `rag_chain.py` — FastAPI doesn't replace that
logic, it just exposes it over HTTP so frontends can reach it.

---

## 11. Running the server

```bash
pip install fastapi uvicorn python-multipart
uvicorn main:app --reload --port 8000
```
Then test with:
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the steps to create an XCFramework?"}'
```
Or just open `http://localhost:8000/docs` in a browser and test there.
