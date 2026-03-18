# AGENTS.md

## Agent Role

Do not optimize, refactor, or generalize unless explicitly asked.

---

## Project

Pixie Questioner is a guided intake Q&A API with two isolated cases:

- `brooklyn`: jeans intake and handoff summary flow
- `cretskens`: LLM-driven wine advising with catalog recommendations

The client sends `current_answer`, an opaque `state`, and `session_id`;
the API returns **exactly one** next message for the selected case.

Stack: Python 3.12+, FastAPI, Pydantic v2, LangChain (Azure OpenAI)

---

## MCP Tools

- **Context7**: fetch library docs (FastAPI, Pydantic v2, LangChain). Prefer over training-time knowledge.
- **GitHub**: read-only repo context (issues, PRs, diffs). Do not write unless explicitly asked.

If an MCP server is unavailable, stop and ask the user.

---

## Execution & Validation

All commands run via Docker Compose. Never run directly on the host.

```bash
docker compose -f docker-compose.dev.yml up --build                          # Run API
RUN_LIVE_LLM_TESTS=0 docker compose -f docker-compose.dev.yml run --rm api pytest  # Tests (live LLM tests disabled)
docker compose -f docker-compose.dev.yml run --rm api ruff check .           # Lint
docker compose -f docker-compose.dev.yml run --rm api black --check .        # Format
docker compose -f docker-compose.dev.yml run --rm api uv tool run ty check   # Type check
```

Live LLM smoke test (only when LLM behavior changes):
```bash
RUN_LIVE_LLM_TESTS=1 docker compose -f docker-compose.dev.yml run --rm api pytest tests/test_live_llm_smoke.py
```

If blocked by credentials or external service, report the exact blocker.

---

## Dependencies

- **uv** only. Never `pip` or `requirements.txt`.
- Do not regenerate `uv.lock` unless explicitly asked.

---

## Invariants (Must Never Break)

- API schemas remain backward compatible
- `state` structure is stable; mutations are explicit
- Questionnaire fields exactly match spec and template
- Question order is deterministic
- Do not infer missing answers
- Prompts: **no stylistic rewrites**
- Wine advisor: action + recommendation selection must be LLM-driven; no keyword/token heuristics
- Cretskens intake routing: LLM-based classification only; no keyword/substring heuristics
- Conversation: exactly one question per turn, only ask for missing fields, stop when complete or user stops, language consistency (NL → EN → FR)

---

## Layer Boundaries

`apps/api/*` is transport-only: request/response wiring, DI, middleware.
No business logic, state mutation, or prompt logic in `apps/api/*`.

---

## Forbidden Unless Explicitly Asked

- dependency upgrades or `uv.lock` changes
- Docker or infrastructure changes
- API or state schema changes
- questionnaire or prompt changes
- large refactors

---

## Clarification Policy

If ambiguous: ask up to 3 concrete questions.
Do not infer API, schema, or behavior changes.

---

## Definition of Done

- Validation passed via Docker Compose
- Commits on a non-main branch
- No temporary `docs/plans/` files committed (except `.gitkeep`)
- Push includes pre-push maintenance review (see `.agents/push.md`)

---

## References

- `.agents/repo-map.md` — file structure and request flow
- `.agents/testing.md` — validation strategy by change type
- `.agents/code-style.md` — style and placement guidance
- `.agents/commit.md` — commit mechanics and branch workflow
- `.agents/push.md` — push and merge workflow
- `.agents/planning.md` — when and how to plan