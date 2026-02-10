# AGENTS.md (pixie_questioner)

## Scope

This file applies to the core questionnaire domain.

## Domain Ownership

This layer defines:

- conversation flow
- stop conditions
- state transitions
- questionnaire semantics

Changes here are **high impact**.

## Extra Invariants

- Question order is deterministic
- State mutations must be explicit
- Do not infer missing answers
- Do not skip questions unless stop-condition applies

## Prompt Safety

- Prompt content is contractual
- Never rewrite tone, structure, or wording
- Only substitute variables

## Change Discipline

- Prefer extending logic over rewriting
- Any change requires at least one regression test
- If behavior is unclear: STOP and ask
