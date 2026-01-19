# pixie-questioner

Pixie Questioner is a lightweight chat API for guided intake Q&A. The UI sends a
question, a `session_id`, and optional `history`, and receives a typed response
as JSON or streamed via SSE for realtime rendering.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env.dev
uvicorn apps.api.main:app --reload
```

Open `http://localhost:8000/docs` for the OpenAPI UI.

## API Contract

### POST /chat

- Query: `stream=false|true`
- Request JSON:
  - `question` (str)
  - `session_id` (str)
  - `history` (list[str], default empty)
  - `questionnaire` (str, optional)
  - `summary` (str, optional)
  - `current_question` (str, optional)
  - `summarize` (bool, default false)
- Response (non-stream): `{"type": "answer|summary", "text": "...", "conversation_id": "..."}`
- Response (stream): `text/event-stream` with `data:` lines and a final `[DONE]`
The `conversation_id` echoes the request `session_id`.

### POST /ask

Alias for `/chat` to keep backward compatibility.

### GET /health

Returns `{"status": "ok"}`.

## Streaming example

```bash
curl -N "http://localhost:8000/chat?stream=true" \
  -H "Content-Type: application/json" \
  -d '{"question":"Hello","session_id":"abc","history":[]}'
```

## Configuration

Set environment variables in `.env.dev`:

- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_KEY`
- `ADVANCED_AZURE_OPENAI_API_VERSION`
- `ADVANCED_AZURE_OPENAI_MODEL`
- `ADVANCED_AZURE_OPENAI_TEMPERATURE`
- `SYSTEM_PROMPT_PATH`
- `SUMMARIZER_PROMPT_PATH` (optional)
- `COMPLETION_TOKEN` (optional)

`system_prompt` is loaded from the file pointed to by `SYSTEM_PROMPT_PATH` (store this file outside the repo).
`summarizer_prompt` is loaded from `SUMMARIZER_PROMPT_PATH` when provided.
`COMPLETION_TOKEN` is stripped from assistant outputs if present, but summary generation is triggered
by the request payload (`summarize=true`) rather than a model-emitted token.

The API expects an LLM wrapper at `src/pixie_questioner/azure_openai_llm.py`
that exposes a `model` with `ainvoke` and `astream` methods.

## Docker (dev)

```bash
docker compose -f docker-compose.dev.yml up --build
```

## Tests

```bash
pytest
```
