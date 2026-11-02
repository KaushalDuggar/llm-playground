# LLM Playground

One app for learning LLM concepts, one lab at a time. This first step is the
client layer and a chat UI: pick a provider and one of its free models, then chat.

## Run

```bash
export GROQ_API_KEY=... NVIDIA_API_KEY=... GOOGLE_API_KEY=... OPENROUTER_API_KEY=...
# or: cp .env.example .env and fill it in
uv run uvicorn app.server:app --reload
```

Open http://localhost:8000. A provider without a key shows as unavailable.
Ollama shows up when `ollama serve` is running.

## Layout

```
core/
  providers.yaml   providers, key env var names, curated free models
  client.py        load_providers, list_models, get_chat_model, chat -> LLMResult
app/
  server.py        FastAPI: GET /api/providers, POST /api/chat
  index.html       the chat UI
```

`core` knows nothing about the UI. Later labs call `client.chat()`, or
`client.get_chat_model()` when they need the LangChain model itself (LangGraph).

No retries and no fallbacks: a call goes to exactly the model you picked, and
failures show the provider's real error in the chat.
