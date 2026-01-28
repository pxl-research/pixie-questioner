# AGENTS.md

## Project

Pixie Questioner is a guided intake Q&A API for denim/jeans preferences. The client sends `current_answer`, an opaque `state`, and `session_id`; the API returns the next question or a final summary.

- **Stack**: Python 3.12+, FastAPI, Pydantic v2, LangChain (Azure OpenAI)

## Documentation

Use [Context7](https://context7.com) MCP to fetch up-to-date documentation for libraries used in this project (FastAPI, Pydantic, LangChain, etc.).

## Run & Validate (always via Docker Compose)

```bash
# Run API
docker compose -f docker-compose.dev.yml up --build

# Run tests
docker compose -f docker-compose.dev.yml run --rm api pytest

# Format
docker compose -f docker-compose.dev.yml run --rm api black .

# Lint
docker compose -f docker-compose.dev.yml run --rm api ruff check .
```

Run the API, tests, formatting, and linting via Docker Compose—not directly on the host.

## Dependencies

- Use uv only.
- Never use pip or requirements.txt.
- Do not regenerate uv.lock unless explicitly asked.

```bash
# Sync dependencies
docker compose -f docker-compose.dev.yml run --rm api uv sync
```

## Code Style

- Formatter: Black
- Linter: Ruff
- Prefer explicit typing.

Good:

```python
def parse_state(state: dict[str, Any]) -> ParsedState:
    ...
```

Bad:

```python
def parse_state(state):
    ...
```

## Code Philosophy

- **Readability first**: Write code that people with less technical background can still understand.
- **Use existing libraries**: Leverage functionality from already-added libraries instead of adding new ones with overlapping features.
- **Keep it compact**: Avoid unnecessary small helper functions; keep code concise but readable.
- **Minimize new files**: Do not add new files unless it improves readability or is logically required.

## Tests

Focus on regression tests, not mock scaffolding.

Good:

- ChatService flow tests
- state transition tests
- stop-condition tests

Bad:

- deep mock hierarchies
- fake infrastructure layers
- tests that only assert mock calls

## Invariants

- API schemas backward compatible.
- `state` structure stable.
- Questionnaire fields exactly match spec and template.
- Prompts: no stylistic rewrites.
- Behavior:
  - one question per turn
  - only missing fields
  - stop when complete or user stops
  - consistent language (NL → EN → FR)

## Forbidden unless asked

- dependency upgrades / uv.lock changes
- Docker / infra changes
- API or state schema changes
- questionnaire changes
- large refactors
- tooling changes

## Repo map

```text
.
├── .env.example
├── .github
│   └── workflows
│       └── docker-publish.yml
├── AGENTS.md
├── apps
│   └── api
│       └── main.py
├── docker-compose.dev.yml
├── infra
│   └── docker
│       └── Dockerfile.dev
├── prompts
│   ├── conversation_summarizer.txt
│   ├── question_asker.txt
│   ├── questionnaire_notulist.txt
│   └── questionnaire_template.txt
├── pyproject.toml
├── QUESTIONER_SPEC.md
├── README.md
├── src
│   └── pixie_questioner
│       ├── __init__.py
│       ├── abstractclasses/
│       ├── audit_client.py
│       ├── azure_openai_llm.py
│       ├── chat_service.py
│       ├── config.py
│       ├── notulist.py
│       ├── schemas.py
│       └── summarizer.py
├── tests
│   └── test_chat_service.py
└── uv.lock
```

## Change Rules

- Keep diffs small.
- Add regression tests for bug fixes.
- If unsure: ask first.

## Committing Changes

After changes are approved and tested, use [.agents/commit.md](.agents/commit.md) for commit instructions. This agent follows Conventional Commits, groups changes logically, and splits commits when appropriate.

Invoke: *"Use `.agents/commit.md` to commit the approved changes."*
