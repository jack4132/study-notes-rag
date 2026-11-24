# Deep Dive: How Streaming (`.astream()`) Actually Works

This doc answers: "how is each word generated and sent over the API? Does
it use webhooks?" — with full technical detail, layer by layer.

---

## 1. How does the MODEL generate words in the first place?

Before anything about APIs/streaming — understand how an LLM produces
text at all:

- GPT-style models are **autoregressive** — they generate ONE token
  (roughly a word or word-piece) at a time, where each new token is
  predicted based on ALL previous tokens (the question + everything
  generated so far)
- Process: `predict next token → append it to the sequence → predict
  next token again (now with one more token of context) → repeat` until
  a stop condition (end-of-sequence token, max length, etc)
- This is happening on OpenAI's servers, on GPU hardware. Each token
  takes some small amount of compute time to produce (this is literally
  WHY long answers take longer — more tokens = more generation steps)
- **Key point:** the model is NEVER "thinking of the whole answer" and
  then typing it out. It génuinely does not know the full sentence in
  advance — it's predicting the immediate next piece, over and over.
  This is why streaming is even POSSIBLE — the tokens really do become
  available one at a time, not all at once internally.

---

## 2. How does OpenAI expose this over their API?

When you call OpenAI's chat completion endpoint with `stream=True`:

- Instead of OpenAI waiting for generation to fully finish, then sending
  you one complete HTTP response — they keep the HTTP connection OPEN
  and send you each token (or small group of tokens) as soon as it's
  generated, as a sequence of small chunks
- Technically, this uses **HTTP chunked transfer encoding** combined with
  **Server-Sent Events (SSE) format** — this is OpenAI's own API
  contract, not something LangChain or we invented
- Each chunk looks roughly like this over the wire:
  ```
  data: {"choices":[{"delta":{"content":"The"}}]}

  data: {"choices":[{"delta":{"content":" first"}}]}

  data: {"choices":[{"delta":{"content":" step"}}]}

  ...

  data: [DONE]
  ```
- Your client (in our case, LangChain's Python SDK) reads this stream
  continuously, parses each `data: ...` line as it arrives, and extracts
  just the new text fragment (`"delta"` content)

**This is a single, long-lived HTTP request-response cycle** — NOT
multiple separate calls, NOT webhooks (see section 5 for why webhooks
are different).

---

## 3. How does LangChain turn this into `.astream()`?

- `langchain-openai`'s `ChatOpenAI` class, under the hood, calls OpenAI's
  official Python SDK with `stream=True`
- OpenAI's SDK itself exposes this as a Python iterator/async iterator —
  something you can loop over with `for chunk in response:` (sync) or
  `async for chunk in response:` (async)
- LangChain wraps each raw chunk into its own `AIMessageChunk` object
  (adds metadata, keeps consistent interface across different LLM
  providers — this is WHY swapping to Ollama later doesn't require
  rewriting your whole app, LangChain normalizes the interface)
- `.astream()` is part of LangChain's **Runnable Protocol** — a shared
  interface EVERY LangChain component implements (prompts, retrievers,
  LLMs, parsers, and full chains built with `|`). Any Runnable
  automatically supports: `.invoke()` (sync, wait for full result),
  `.batch()` (multiple inputs at once), `.stream()` (sync streaming),
  `.astream()` (async streaming) — you get all 4 for free once your
  chain is built with LCEL (`|` syntax)

---

## 4. How does streaming flow through YOUR WHOLE CHAIN (not just the LLM)?

Recall our chain:
```python
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

When you call `rag_chain.astream(question)`:

1. **Retriever + format_docs** — these run to completion FIRST,
   synchronously-ish (not token-by-token, they don't produce "tokens" at
   all, they do one database lookup + string formatting = fast, instant,
   no streaming nature to them)
2. **Prompt** — fills the template with that context, instant, one-shot
3. **LLM (ChatOpenAI)** — THIS is where real streaming begins. It starts
   emitting `AIMessageChunk` pieces as OpenAI sends them
4. **StrOutputParser()** — as each chunk arrives from the LLM, this
   immediately extracts just the plain text string from it and passes it
   along — doesn't wait for more chunks, processes each one as it comes
5. Your `ask_stream()` function's `async for chunk in rag_chain.astream(question): yield chunk`
   receives each of these plain text pieces in real time, and immediately
   re-yields them further up to FastAPI

So the "streaming" part of the whole pipeline is ONLY the LLM generation
step — everything before it (retrieval, prompt building) happens once,
upfront, non-streamed, because those parts aren't inherently gradual
processes.

---

## 5. Does this use webhooks? NO — here's the actual difference

This is a common confusion, worth being precise about:

| | **SSE / Streaming (what we're using)** | **Webhooks** |
|---|---|---|
| Connection | ONE long-lived HTTP request stays open the whole time | MANY separate, independent HTTP requests |
| Who initiates each piece | Server pushes data continuously over the SAME open connection | Server calls YOUR separate endpoint URL, a fresh new HTTP POST, each time something happens |
| Direction | One-way, server → client, within one request/response cycle | Server → client, but as discrete, disconnected calls to a URL you registered in advance |
| Use case | Real-time progressive data within a single operation (like this chat answer generating) | Event notifications about something async elsewhere (like "payment completed", "file finished processing in background") — happens later, possibly way later, unrelated to any specific open connection |
| Example | ChatGPT-style live typing effect | Stripe calling your server when a customer's payment succeeds, Github calling your server when someone pushes code |

**In short:** streaming = "stay on the phone call, I'll keep talking until
I'm done." Webhook = "hang up now, I'll call you back later on a
different line when something happens."

Our `/api/chat/stream` endpoint is 100% streaming (SSE), zero webhook
involved anywhere in this flow.

---

## 6. Full mechanism chain, summarized top to bottom

```
GPT model generates tokens one-by-one (autoregressive, inherent to how
LLMs work)
        ↓
OpenAI's API exposes this via stream=True, using HTTP chunked transfer +
SSE wire format (data: ...\n\n per chunk)
        ↓
OpenAI's official Python SDK parses this into a Python async iterator
        ↓
LangChain's ChatOpenAI wraps each piece into AIMessageChunk objects,
exposed via .astream() (part of the shared Runnable Protocol all
LangChain components implement)
        ↓
Our LCEL chain (retriever | format_docs | prompt | llm | StrOutputParser)
runs non-streaming parts first (retrieval, prompt fill), then streams
only the LLM + parser portion
        ↓
Our ask_stream() function re-yields each piece of plain text
        ↓
FastAPI's StreamingResponse wraps each piece into SSE format again
(data: token\n\n) and sends over OUR OWN API's HTTP connection to
whatever client called /api/chat/stream
        ↓
Frontend (React/Android, Phase 4) reads this stream, appends each piece
to the chat bubble on screen as it arrives
```

Two separate SSE streams actually exist in this whole system:
1. OpenAI → our backend (internal, invisible to the end user)
2. Our backend → the frontend client (the one we built, what the user's
   app actually consumes)

---

## Open items for further research (optional, if you want to go deeper later)
- Look at OpenAI's raw API response structure directly (bypass LangChain)
  using `curl` with `stream: true` manually, to see the raw
  `data: {...}` JSON chunks yourself
- Read about "KV cache" — a performance technique LLMs use internally to
  avoid recomputing the whole sequence from scratch for every new token
  (makes autoregressive generation fast enough to be usable)
- Compare SSE vs WebSockets — WebSockets allow TWO-way communication
  (client can also send data mid-stream), SSE is one-way only
  (server → client). We're using SSE since chat doesn't need the client
  to interrupt mid-generation (yet — that'd be a WebSocket use case)
