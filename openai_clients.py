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

import os

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
