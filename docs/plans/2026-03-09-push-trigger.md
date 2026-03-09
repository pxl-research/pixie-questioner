# Push Trigger Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Document a stable repo-local push workflow that performs pre-push maintenance before pushing the current branch to `origin`.

**Architecture:** Store the trigger and its pre-push maintenance behavior as explicit workflow rules in `AGENTS.md`. Do not add shell aliases, slash commands, or tool-specific config because those would be less portable than repo documentation.

**Tech Stack:** Markdown, git workflow documentation

---

### Task 1: Document the Push Trigger

**Files:**
- Modify: `AGENTS.md`
- Modify: `docs/plans/2026-03-09-push-trigger-design.md`
- Modify: `docs/plans/2026-03-09-push-trigger.md`

**Step 1: Write the failing test**

Configuration-only change. No automated test file is added.

**Step 2: Run test to verify it fails**

Not applicable for this documentation workflow change.

**Step 3: Write minimal implementation**

Update the `Push Workflow` section in `AGENTS.md` so `Push deze branch` triggers a pre-push maintenance pass that checks relevant documentation and `pixie-contracts`, applies any clear updates automatically, validates them, commits them on the current branch, and only then pushes.

**Step 4: Run test to verify it passes**

Run: `docker compose -f docker-compose.dev.yml run --rm api python - <<'PY' ...`
Expected: the script confirms the required pre-push maintenance text is present in `AGENTS.md`.

**Step 5: Commit**

```bash
git add AGENTS.md docs/plans/2026-03-09-push-trigger-design.md docs/plans/2026-03-09-push-trigger.md
git commit -m "docs(workflow): add pre-push maintenance"
```
