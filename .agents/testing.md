# .agents/testing.md

Validation guidance for `pixie-questioner`.
Standard commands are in `AGENTS.md` § Execution & Validation; this file adds strategy.

---

## Strategy

1. **Smallest check first** — run a single test file or just lint if that covers the change
2. **Health check for API surface changes** — `curl http://localhost:8000/health` must return `{"status":"ok"}`
3. **Targeted over full suite** — full suite only when invariants may be impacted
4. **Live LLM tests locally are opt-in** — run them only for LLM behavior changes; never block local progress on missing credentials

CI note:
- `.github/workflows/ci.yml` enforces a `/health` startup check and runs the live-enabled suite for PRs and pushes to `main`.

---

## Minimum Validation by Change Type

| Change type              | pytest | lint+format+type | health check | live LLM |
|--------------------------|--------|-------------------|--------------|----------|
| Pure Python logic        | ✓      | ✓                 |              |          |
| API wiring / endpoint    | ✓      | ✓                 | ✓            |          |
| LLM behavior             | ✓      | ✓                 |              | optional |

---

## Test Conventions

- No `conftest.py` — fixtures are inline per test file
- Env setup via `monkeypatch.setenv()`; file fixtures via `tmp_path`
- Tests create domain profiles + prompt files in `tmp_path` and point config there
- Sync execution using `TestClient` (from `fastapi.testclient`), not async fixtures
- Wine catalog tests build `.xlsx` in-memory

---

## Failure Handling

Fix → re-run the same check → do not continue until resolved.
If blocked by credentials or external service, report the exact blocker and stop.
