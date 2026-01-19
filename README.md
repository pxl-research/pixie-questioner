# pixie-questioner

Pixie Questioner is a lightweight chat API for guided intake Q&A. The UI sends a
`current_answer`, an opaque `state` blob, and a `session_id`, and receives a typed
JSON response with the next question or a final summary.

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

- Request JSON:
  - `session_id` (str)
  - `current_answer` (str)
  - `state` (object, default empty)
- Response: `{"type": "answer|summary", "text": "...", "conversation_id": "...", "state": {...}}`.
The `conversation_id` echoes the request `session_id`.
Store the returned `state` and send it back on the next turn.
On the first turn, send an empty object for `state`.
Treat `state` as opaque; its contents are service-specific.

### GET /health

Returns `{"status": "ok"}`.

## Configuration

Set environment variables in `.env.dev`:

- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_KEY`
- `ADVANCED_AZURE_OPENAI_API_VERSION`
- `ADVANCED_AZURE_OPENAI_MODEL`
- `ADVANCED_AZURE_OPENAI_TEMPERATURE`
- `SYSTEM_PROMPT_PATH`
- `SUMMARIZER_PROMPT_PATH` (optional)
- `QUESTIONNAIRE_NOTULIST_PROMPT_PATH`
- `QUESTIONNAIRE_TEMPLATE_PATH`
- `COMPLETION_TOKEN` (optional)

`system_prompt` is loaded from the file pointed to by `SYSTEM_PROMPT_PATH` (store this file outside the repo).
`summarizer_prompt` is loaded from `SUMMARIZER_PROMPT_PATH` when provided.
`questionnaire_notulist_prompt` is loaded from `QUESTIONNAIRE_NOTULIST_PROMPT_PATH`.
`questionnaire_template` is loaded from `QUESTIONNAIRE_TEMPLATE_PATH` and used to seed the first turn.
`COMPLETION_TOKEN` is only used to fill/strip the question-asker prompt placeholder; summary generation
is triggered by the orchestrator logic, not by a model-emitted token.
`QUESTIONNAIRE_NOTULIST_PROMPT_PATH` must be set for the API to start.

The API expects an LLM wrapper at `src/pixie_questioner/azure_openai_llm.py`
that exposes a `model` with an `ainvoke` method.

## Docker (dev)

```bash
docker compose -f docker-compose.dev.yml up --build
```

## Tests

```bash
pytest
```
