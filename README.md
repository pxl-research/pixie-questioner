# pixie-questioner

Pixie Questioner is a lightweight chat API for guided intake Q&A. The UI sends a
`current_answer`, an opaque `state` blob, and a `session_id`, and receives a typed
JSON response with the next question or a final summary.

## Quickstart

```bash
cp .env.example .env.dev
docker compose -f docker-compose.dev.yml up --build
```

Open `http://localhost:8000/docs` for the OpenAPI UI.

## API Contract

### POST /chat

- Request JSON:
  - `session_id` (str)
  - `current_answer` (str)
  - `state` (object, default empty)
- Response (next question): `{"type": "answer", "text": "...", "conversation_id": "...", "state": {...}}`.
- Response (summary): `{"type": "summary", "text": "...", "summary": "...", "conversation_id": "...", "state": {...}}`.
The `conversation_id` echoes the request `session_id`.
Store the returned `state` and send it back on the next turn.
On the first turn, send an empty object for `state`.
Treat `state` as opaque; its contents are service-specific.
When `type` is `summary`, `text` is the handoff message and `summary` contains the actual summary.
When `type` is `answer`, the optional field `quick_replies` may be present with up to 6 `{label, value}` options.

### GET /health

Returns `{"status": "ok"}`.

## Audit Event Contract

When `AUDIT_API_URL` is configured, this service posts events to `POST /events`
on pixie-audit using a strict outbound payload contract:

- `conversation_id` (str)
- `profile` (str)
- `messages` (non-empty list of `{role, content}` with roles limited to `user`, `assistant`, `summary`)
- `metadata` (optional object with only `location`)

No additional top-level fields are sent to pixie-audit.
The source-of-truth contract lives in `pixie-contracts/contracts/audit/v1/openapi.yaml`.

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
- `AUDIT_API_URL` (optional)
- `AUDIT_API_KEY` (optional)
- `AUDIT_PROFILE` (optional)
- `AUDIT_TIMEOUT_SECONDS` (optional, default 5)

`system_prompt` is loaded from the file pointed to by `SYSTEM_PROMPT_PATH` (store this file outside the repo).
`summarizer_prompt` is loaded from `SUMMARIZER_PROMPT_PATH` when provided.
`questionnaire_notulist_prompt` is loaded from `QUESTIONNAIRE_NOTULIST_PROMPT_PATH`.
`questionnaire_template` is loaded from `QUESTIONNAIRE_TEMPLATE_PATH` and used to seed the first turn.
`COMPLETION_TOKEN` is only used to fill/strip the question-asker prompt placeholder; summary generation
is triggered by the orchestrator logic, not by a model-emitted token.
`QUESTIONNAIRE_NOTULIST_PROMPT_PATH` must be set for the API to start.
If `AUDIT_API_URL` is set, the service posts interaction events to `/events` using `AUDIT_PROFILE`
as the required profile value and `AUDIT_API_KEY` for the `X-API-Key` header when provided.

The API expects an LLM wrapper at `src/pixie_questioner/azure_openai_llm.py`
that exposes a `model` with an `ainvoke` method.

## Docker (dev)

```bash
docker compose -f docker-compose.dev.yml up --build
```

## Tests

```bash
docker compose -f docker-compose.dev.yml run --rm api pytest
```

## Format

```bash
docker compose -f docker-compose.dev.yml run --rm api black .
```

## Lint

```bash
docker compose -f docker-compose.dev.yml run --rm api ruff check .
```
