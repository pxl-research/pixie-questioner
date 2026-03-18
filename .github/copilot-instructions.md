# Copilot Instructions

- Use Docker Compose only. Never run commands on the host.
- Use `uv` for dependencies. Do not use pip or requirements.txt.
- Do not introduce heuristic or keyword-based logic for LLM behavior.
- Keep API layer (`apps/api/*`) strictly transport-only.
- Respect invariants: state structure, deterministic ordering, schema compatibility.
- Do not modify prompts, questionnaires, or schemas unless explicitly requested.
- Prefer minimal, focused changes. Do not refactor unless asked.
- Always validate via Docker before committing (tests, lint, format, type check).
- Do not commit from main branch.
- Ask for clarification if behavior or schema is ambiguous.

See AGENTS.md for full project rules.