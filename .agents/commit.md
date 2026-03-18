# .agents/commit.md

Operational guide for commits in `pixie-questioner`.
Principles and forbidden-change rules live in `AGENTS.md`; this file covers mechanics only.

---

## Branch Guardrail

Before any commit, verify the branch:

```bash
git branch --show-current
```

If on `main`/`master`, create a task branch first:

```bash
git switch -c <type>/<short-description>
```

Branch prefixes: `feat/`, `fix/`, `refactor/` (only if requested), `chore/`.

---

## Pre-Commit Checklist

1. Branch is not `main`/`master`
2. Staged changes match the task (`git diff --cached`)
3. No forbidden changes, temp files, or `docs/plans/` files staged
4. Relevant validation passed (see `.agents/testing.md`)

---

## Commit Messages

Format: `<type>: <imperative description>`

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`

Examples:
- `fix: preserve cretskens state mutation order`
- `test: cover empty current_answer handling`

Add a body only when it materially improves review clarity.

---

## Splitting vs. Single Commit

Split when changes affect different logical areas or one part could be reverted independently.
Keep together when splitting would break behavior or make history harder to follow.

---

## Iteration Pattern

For non-trivial work: implement small change → validate via Docker → commit → repeat.
Avoid accumulating large uncommitted diffs.
