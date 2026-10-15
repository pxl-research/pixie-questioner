---
name: pixie-guardrails
description: Apply hard project guardrails before implementing changes in pixie-questioner.
---

# Pixie Guardrails

Use this checklist before and during implementation.

## Never Break

- API schemas stay backward compatible.
- `state` structure stays stable; every mutation must be explicit.
- Questionnaire fields must match spec/templates exactly.
- Question order must remain deterministic.
- Never infer missing answers.
- Prompts: no stylistic rewrites.
- Wine advisor action/recommendation selection stays LLM-driven (no token heuristics).
- Cretskens intake routing stays LLM-driven (no keyword/substring routing).
- Conversation rules: exactly one question per turn, only missing fields, stop when complete or user stops, and preserve language consistency.

## Layer Boundary

- `apps/api/*` is transport-only.
- No business logic, prompt logic, or state mutation inside `apps/api/*`.

## Forbidden Without Explicit Request

- Dependency upgrades or `uv.lock` changes
- Docker/infrastructure changes
- API or state schema changes
- Questionnaire or prompt changes
- Large refactors

## Ambiguity Policy

- Ask up to 3 concrete questions when ambiguous.
- Do not infer behavior/schema/API changes.
