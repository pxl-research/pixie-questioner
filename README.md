# pixie-questioner

Pixie Questioner is a lightweight chat API with two isolated cases:
- `brooklyn` (jeans intake and handoff summary flow)
- `cretskens` (LLM-driven wine advising with catalog recommendations)

The UI sends `current_answer`, an opaque `state` object, and a `session_id`, and
receives exactly one next message for the selected case.

## Quickstart

```bash
cp .env.example .env.dev
docker compose -f docker-compose.dev.yml up --build
```

Set real Azure values in `.env.dev` before calling the LLM-backed routes.

Open `http://localhost:8000/docs` for the OpenAPI UI.

## API Contract

### POST /cases/{case_id}/chat

- `case_id` supports `brooklyn` and `cretskens`.
- Request JSON:
  - `session_id` (str)
  - `current_answer` (str)
  - `state` (object, default empty)
- Response shape depends on case:
  - `brooklyn`: `answer` or `summary`
  - `cretskens`: `question` or `recommendations`
The `conversation_id` echoes the request `session_id`.
Store the returned `state` and send it back on the next turn.
On the first turn, send an empty object for `state`.
Treat `state` as opaque; its contents are service-specific.
When `type` is `recommendations`, each recommendation item contains:
- `name`
- `price_eur`
- `country`
- `region`

Behavior guarantees:
- One question per turn while intake is active.
- Intake stops after at most 6 questions.
- Recommendations are selected from the configured wine catalog only.
- State is signed and scoped per case; state from one case is rejected by the other.

Response headers:
- `X-Pixie-Case`: resolved case id (`brooklyn` or `cretskens`)
- `X-Pixie-Chat-Contract`: contract version marker
- `X-Pixie-Case-Contract`: exact case contract version

Canonical chat contracts live in `pixie-contracts`:
- `pixie-contracts/contracts/chat/brooklyn/v2/openapi.yaml`
- `pixie-contracts/contracts/chat/cretskens/v1/openapi.yaml`

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
- `ADVANCED_AZURE_OPENAI_TIMEOUT_SECONDS` (optional, default `15`)
- `ADVANCED_AZURE_OPENAI_MAX_TOKENS` (optional, default `450`)
- `BROOKLYN_DOMAIN_PROFILE_PATH` (optional, default `profiles/brooklyn/domain_profile.yml`)
- `CRETSKENS_PROFILE_PATH` (optional, default `profiles/cretskens/advisor_profile.yml`)
- `CRETSKENS_WINE_CATALOG_PATH` (optional, default `profiles/cretskens/cretskens_wijnen_reduced.xlsx`)
- `WINE_ADVISOR_CANDIDATE_LIMIT` (optional, default `40`)
- `WINE_ADVISOR_MAX_INPUT_CHARS` (optional, default `600`)
- `WINE_ADVISOR_MAX_STATE_BYTES` (optional, default `12000`)
- `WINE_ADVISOR_MAX_WORKING_CONTEXT_CHARS` (optional, default `2200`)
- `WINE_ADVISOR_MAX_HISTORY_ITEMS` (optional, default `8`)
- `WINE_ADVISOR_MAX_HISTORY_ITEM_CHARS` (optional, default `240`)
- `AUDIT_API_URL` (optional)
- `AUDIT_API_KEY` (optional)
- `AUDIT_PROFILE` (optional)
- `AUDIT_TIMEOUT_SECONDS` (optional, default 5)
- `STATE_SIGNING_KEY` (recommended; if omitted, signed state breaks across restarts)
- `STATE_SESSION_TTL_SECONDS` (optional, default `1800`)

`CRETSKENS_WINE_CATALOG_PATH` points to the XLSX catalog with required columns:
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

`BROOKLYN_DOMAIN_PROFILE_PATH` points to a YAML domain profile that defines:
- prompt file paths (question asker, notulist, template, summarizer, relevance classifier)
- localized runtime copy (handoff, feedback, off-topic, goodbye)
- quick replies and normalization tokens
- stop detection phrases/context tokens
- behavior knobs (`completion_token`, off-topic threshold)

`CRETSKENS_PROFILE_PATH` points to a YAML advisor profile that defines:
- prompt file paths for the turn controller and ranker
- the recommendation intro copy used in API responses

Prompt paths inside the profile can be absolute or relative to the profile file directory.
If `AUDIT_API_URL` is set, the service posts interaction events to `/events` using `AUDIT_PROFILE`
as the required profile value and `AUDIT_API_KEY` for the `X-API-Key` header when provided.
If `STATE_SIGNING_KEY` is omitted, the backend generates an ephemeral key at startup and
client `state` values become invalid after a restart.

## Docker (dev)

```bash
docker compose -f docker-compose.dev.yml up --build
```

## Tests

```bash
RUN_LIVE_LLM_TESTS=0 docker compose -f docker-compose.dev.yml run --rm api pytest
```

This is the recommended local default. Use
`docker compose -f docker-compose.dev.yml run --rm api pytest`
only when you explicitly want the environment default.

## Live Smoke Test

```bash
RUN_LIVE_LLM_TESTS=1 docker compose -f docker-compose.dev.yml run --rm api pytest tests/test_live_llm_smoke.py
```

## Type Check

```bash
docker compose -f docker-compose.dev.yml run --rm api uv tool run ty check
```

## Security Lint

```bash
docker compose -f docker-compose.dev.yml run --rm api ruff check . --select S
docker compose -f docker-compose.dev.yml run --rm api uv tool run bandit -r src
```

## Format

```bash
docker compose -f docker-compose.dev.yml run --rm api black .
```

## Lint

```bash
docker compose -f docker-compose.dev.yml run --rm api ruff check .
```
