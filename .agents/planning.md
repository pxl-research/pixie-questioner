# .agents/planning.md

When to plan and how to structure it. Skip planning for small, obvious changes.

---

## When to Plan

Plan when: multi-file change, ambiguous behavior, invariants at risk, or LLM logic affected.
Skip when: single-file low-risk change or obvious fix.

---

## Plan Template

```text
Objective:   what needs to be achieved
Scope:       files/modules that will be touched
Approach:    ordered steps
Validation:  how correctness will be verified
Risks:       anything uncertain or requiring confirmation
```

Keep each section to 1–3 lines.

---

## Rules

- Focus on *what* and *where*, not detailed code
- Call out assumptions and affected invariants explicitly
- If assumptions or invariant risk exist → ask for confirmation before implementing
- Never commit planning documents under `docs/plans/`