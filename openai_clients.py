"""
openai_clients.py
------------------
Shared OpenAI client configuration.

Render (and some other cloud hosts) have broken/incomplete IPv6 outbound
networking. httpx (used internally by the OpenAI SDK) tries IPv6 first
by default, which hangs/fails on these hosts and surfaces as a generic
"APIConnectionError: Connection error" - even though the API key and
everything else is correct.

Fix: force IPv4-only connections via a custom httpx client, shared by
both embeddings and chat model instances.
"""

import httpx

# local_address="0.0.0.0" forces the underlying socket to bind using
# IPv4, skipping the IPv6 attempt entirely.
_http_client = httpx.Client(
    transport=httpx.HTTPTransport(local_address="0.0.0.0")
)
_async_http_client = httpx.AsyncClient(
    transport=httpx.AsyncHTTPTransport(local_address="0.0.0.0")
)

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"


def get_http_client():
    return _http_client


def get_async_http_client():
    return _async_http_client
