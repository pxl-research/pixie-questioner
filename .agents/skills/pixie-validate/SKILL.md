---
name: pixie-validate
description: Run the correct pixie-questioner validation commands through Docker Compose and report exact blockers.
argument-hint: "[logic|api|llm|full]"
---

# Pixie Validate

Run validation only via Docker Compose for this repository.

## Command Set

```bash
RUN_LIVE_LLM_TESTS=0 docker compose -f docker-compose.dev.yml run --rm api pytest
docker compose -f docker-compose.dev.yml run --rm api ruff check .
docker compose -f docker-compose.dev.yml run --rm api black --check .
docker compose -f docker-compose.dev.yml run --rm api uv tool run ty check
```

Live smoke test (only for LLM behavior changes):

```bash
RUN_LIVE_LLM_TESTS=1 docker compose -f docker-compose.dev.yml run --rm api pytest tests/test_live_llm_smoke.py
```

## Mode Selection

- `logic`: run `pytest`, `ruff`, `black --check`, `ty check`
- `api`: run `pytest`, `ruff`, `black --check`, `ty check`, then health check against running API
- `llm`: run base checks, then optionally live smoke test if prompts/LLM behavior changed
- `full` or empty argument: run all base checks; include live smoke test only if explicitly required

## Reporting Rules

- Report each command with pass/fail and exact blocker details.
- If blocked by credentials or external services, stop and report the exact blocker.
- Do not claim success without command output evidence from this run.
