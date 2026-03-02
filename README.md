# pixie-questioner

Pixie Questioner is a lightweight chat API for LLM-driven wine advising. The UI
sends `current_answer`, an opaque `state` object, and a `session_id`, and receives
either one next question or catalog-backed recommendations.

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
- Response (next question): `{"type": "question", "text": "...", "conversation_id": "...", "state": {...}}`.
- Response (recommendations): `{"type": "recommendations", "text": "...", "recommendations": [...], "conversation_id": "...", "state": {...}}`.
The `conversation_id` echoes the request `session_id`.
Store the returned `state` and send it back on the next turn.
On the first turn, send an empty object for `state`.
Treat `state` as opaque; its contents are service-specific.
Optional: set `state.debug_mode=true` to request compact debug traces in `state.debug.trace`
when backend debug mode is enabled.
When `type` is `recommendations`, each recommendation item contains:
- `name`
- `price_eur`
- `country`
- `region`

Behavior guarantees:
- One question per turn while intake is active.
- Intake stops after at most 6 questions.
- Recommendations are selected from the configured wine catalog only.

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
- `DOMAIN_PROFILE_PATH`
- `WINE_CATALOG_PATH` (optional; fallback: `profiles/cretskens/cretskens_wijnen_reduced.xlsx`)
- `WINE_ADVISOR_DEBUG_ENABLED` (optional; default `true`, set `false` to disable debug traces server-side)
- `AUDIT_API_URL` (optional)
- `AUDIT_API_KEY` (optional)
- `AUDIT_PROFILE` (optional)
- `AUDIT_TIMEOUT_SECONDS` (optional, default 5)

`WINE_CATALOG_PATH` points to the XLSX catalog with required columns:
- `barcode`
- `naam`
- `prijs`
- `land`
- `regio`
- `kleur`
- `druiven_beschrijving`
- `geur_beschrijving`
- `smaak_beschrijving`
- `foodpair_beschrijving`

`DOMAIN_PROFILE_PATH` still points to a YAML domain profile that defines:
- prompt file paths (question asker, notulist, template, summarizer, relevance classifier)
- localized runtime copy (handoff, feedback, off-topic, goodbye)
- quick replies and normalization tokens
- stop detection phrases/context tokens
- behavior knobs (`completion_token`, off-topic threshold)

Prompt paths inside the profile can be absolute or relative to the profile file directory.
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
