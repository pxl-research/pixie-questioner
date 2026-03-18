# .agents/code-style.md

Practical code-style guidance for `pixie-questioner`.
Invariants, layer boundaries, and forbidden changes live in `AGENTS.md`.

---

## Guiding Principles

- Consistency with existing code over introducing new patterns
- Simple, readable control flow — avoid deep nesting and hidden side effects
- Change only what the task requires; no speculative improvements

---

## Where Code Belongs

See `.agents/repo-map.md` for the full map. Quick rule:
- HTTP wiring → `apps/api/*`
- Business / questionnaire logic → `src/pixie_questioner/chat/`
- Wine advisor logic → `src/pixie_questioner/wine_advisor/`
- LLM calls & prompts → respective layer; never replace with heuristics

---

## Naming & Typing

- Explicit type annotations everywhere
- Domain-specific names over generic ones
- Consistent with existing conventions in the repo

---

## When in Doubt

If a change could impact state, question order, or LLM behavior — stop and ask.
