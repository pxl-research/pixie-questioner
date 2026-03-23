# Aroma/Taste Classifier Analysis & Redesign Proposal

**Date:** 2026-03-23
**Branch:** fix/cretskens-origin-fallback
**Status:** Analysis complete — no code changes made

---

## 1. Problem Statement

When a user answers **"fris en fruitig"** to the aroma/taste preference question
(field 8 in the Cretskens intake questionnaire), the system produces an error.
Investigation reveals this is a symptom of a deeper architectural mismatch: the
`aroma_taste_classifier` subsystem was designed for a data model that no longer
exists.

---

## 2. Current Data Model

The wine catalog (`profiles/cretskens/cretskens_wijnen_reduced.xlsx`) contains
**free-text description columns**:

| Column               | Type      | Example                                            |
|----------------------|-----------|----------------------------------------------------|
| `geur_beschrijving`  | Free text | "Citrus, frisse agrum, witte bloesems"             |
| `smaak_beschrijving` | Free text | "Wit steenfruiten met elegante, zachte zuren, sappig" |

These are loaded into `WineCatalogRecord.aroma_description` and
`WineCatalogRecord.taste_description` by `catalog_loader.py`.

There are **no boolean columns** for aroma or taste categories anywhere in the
data. The catalog has exactly 10 columns: `barcode`, `naam`, `prijs`, `land`,
`regio`, `kleur`, `druiven_beschrijving`, `geur_beschrijving`,
`smaak_beschrijving`, `foodpair_beschrijving`.

---

## 3. Current Aroma/Taste Classifier Architecture

### 3.1 The Boolean Flag System

`categorized_handoff.py` defines 21 hardcoded boolean keys in
`AROMA_TASTE_KEY_ORDER`:

```
geur_citrus, geur_groen_vegetaal, geur_wit_fruit, geur_tropisch_fruit,
geur_steenfruit, geur_rood_fruit, geur_donker_fruit, geur_floraal,
geur_kruidig, geur_mineraal_aards, geur_zoet_rijping, geur_hout_geroosterd,
smaak_zoet, smaak_droog, smaak_fris_zuur, smaak_fruitig, smaak_kruidig,
smaak_bruisend, smaak_houtgerijpt, smaak_mineralig, smaak_romig
```

### 3.2 The Classifier Prompt

`profiles/cretskens/aroma_taste_classifier.txt` instructs the LLM to:

1. Take user input (e.g. "fris en fruitig")
2. Return a JSON object with boolean values using **only** the allowed keys
3. Set `true` when the user explicitly prefers it, `false` when explicitly
   rejected, omit when not mentioned

Example expected output for "fris en fruitig":
```json
{"smaak_fris_zuur": true, "smaak_fruitig": true}
```

### 3.3 The Processing Pipeline

The pipeline has multiple layers of parsing, normalization, and fallback:

```
User input: "fris en fruitig"
  │
  ▼
questionnaire_notulist (LLM) → writes to field 8 as free text
  │
  ▼
normalize_questionnaire() [on every intake turn]
  ├── extract field 8 value
  ├── parse_aroma_taste_flags() — tries JSON parse, then text regex
  │   ├── parse_json_object() — json.loads, ast.literal_eval
  │   ├── _extract_flags_from_mapping() — normalize keys, extract booleans
  │   └── _extract_flags_from_text() — regex for "key:true/false" patterns
  ├── if no flags found → _classify_aroma_taste_flags() (LLM call)
  │   ├── first attempt with allowed keys
  │   ├── parse result → if empty, retry with strict prompt
  │   └── second attempt with stricter instructions
  └── serialize back to field 8 as compact JSON
  │
  ▼
build_handoff_categories_v1() [at handoff]
  ├── _classify_handoff_core_categories() → foodpair, kleur, land, regio, prijs
  ├── parse field 8 again → extract aroma_taste boolean flags
  ├── merge flags into categories dict
  └── serialize to JSON string
  │
  ▼
cretskens_service._build_working_context()
  └── includes handoff_categories_v1 as text:
      "Handoff categories:\n{\"smaak_fris_zuur\":true,\"smaak_fruitig\":true}"
  │
  ▼
Wine advisor receives working_context as text
  ├── rank_catalog_records() — tokenized text overlap scoring
  ├── turn_controller — LLM sees working_context as text
  └── ranker — LLM sees working_context + full catalog descriptions
```

### 3.4 Supporting Infrastructure

The processing pipeline includes significant supporting code:

- **Key normalization**: `_NORMALIZED_AROMA_KEYS` dict mapping normalized
  forms back to canonical keys
- **Suffix matching**: `_AROMA_SUFFIX_MAP` for matching partial keys
  (e.g. "kruidig" → both "geur_kruidig" and "smaak_kruidig")
- **Boolean extraction**: `_extract_bool_value()` with string-to-bool coercion
- **Candidate key resolution**: `_candidate_aroma_keys()` with fallback chain
- **Nested dict handling**: `_extract_flags_from_mapping()` unwraps nested dicts
- **Text regex fallback**: `_extract_flags_from_text()` for unparseable LLM output
- **Serialization**: `serialize_aroma_taste_flags()` with ordered key output

This is ~150 lines of parsing and normalization code for a subsystem that,
as shown below, provides no downstream value.

---

## 4. Why the Boolean Approach Cannot Work

### 4.1 The Flags Don't Map to Any Data

The 21 boolean keys in `AROMA_TASTE_KEY_ORDER` do not correspond to any columns
in the Excel catalog. The old data model presumably had individual boolean columns
like `geur_citrus`, `smaak_zoet`, etc., enabling deterministic filtering:

```
Old model (hypothetical):
  geur_citrus=true AND smaak_fruitig=true → filter matching wines

Current model:
  geur_beschrijving="Citrus, frisse agrum, witte bloesems" (free text)
  smaak_beschrijving="Wit steenfruiten met elegante, zachte zuren, sappig"
```

There is no code path that matches boolean flags against the free-text
descriptions in the catalog. The flags are orphaned.

### 4.2 The Flags Are Destructive

When the user says "fris en fruitig", the notulist correctly stores this as free
text in field 8. However, `normalize_questionnaire()` then **overwrites** this
free text with:

```json
{"smaak_fris_zuur":true,"smaak_fruitig":true}
```

This destroys the natural language that would be useful downstream:

- **Token-based retrieval** (`retrieval.py`) tokenizes the working context and
  matches against catalog text. The tokens "fris", "fruitig" would partially match
  descriptions like "frisse agrum" and "Rijpe bosvruchten". The tokens
  "smaak_fris_zuur" and "smaak_fruitig" match nothing in the catalog.

- **LLM-based ranking** (`llm_ranker.py`) sends the working context alongside
  full catalog descriptions. An LLM can match "fris en fruitig" to descriptions
  like "zachte zuren, sappig" far better than it can interpret
  `{"smaak_fris_zuur":true}`.

### 4.3 The Classification Adds Latency and Fragility

The pipeline makes **up to 4 extra LLM calls** per conversation for aroma/taste:

1. `normalize_questionnaire()` → `_classify_aroma_taste_flags()` (1st attempt)
2. `_classify_aroma_taste_flags()` retry (if 1st fails)
3. `build_handoff_categories_v1()` → `_classify_aroma_taste_flags()` (1st attempt)
4. `build_handoff_categories_v1()` retry (if 1st fails)

Each call adds latency, cost, and a failure point. The LLM may return
non-standard JSON, unexpected keys, or malformed output — triggering the error
the user observed.

### 4.4 The Notulist Already Handles This Correctly

The `questionnaire_notulist.txt` prompt (lines 112-120) already instructs the LLM
to store field 8 as a **short free-text summary** using language close to the
aroma/taste groups. This is the correct approach for the current data model — the
notulist output is directly useful for downstream text matching and LLM ranking.

Relevant notulist instructions:
```
13. For field 8 (desired aroma/taste profile):
    - keep a short free-text summary of what the customer wants.
    - keep explicit negations when present (e.g. "niet zoet", "geen houttoetsen").
    - if the customer corrects themselves, keep the latest statement.
14. For field 8, prefer wording that stays close to these aroma/taste groups:
    - geur: citrus, groen/vegetaal, wit fruit, tropisch fruit, steenfruit, ...
    - smaak: zoet, droog, fris/zuur, fruitig, kruidig, bruisend, ...
```

---

## 5. How the Error Manifests

When a user answers "fris en fruitig":

1. The notulist writes something like `fris en fruitig` to field 8
2. `normalize_questionnaire()` detects this isn't JSON or key:value format
3. It calls `_classify_aroma_taste_flags()` which invokes the LLM
4. The LLM may return various formats that the parsing chain cannot handle,
   or the LLM call itself may fail
5. Even when successful, the boolean flags replace the original text
6. Downstream, the boolean flags provide no matching value

The specific runtime error likely occurs when:
- The LLM returns output that doesn't parse as JSON (both attempts)
- The retry chain exhausts, returning `{}`, and then the field 8 value becomes
  empty or malformed
- Or: the questionnaire normalization disrupts the intake flow state, causing
  a subsequent turn to fail

---

## 6. Impact Assessment

### Code Affected

| File | Lines | What it does |
|------|-------|-------------|
| `chat/categorized_handoff.py` | 93-257 | `AROMA_TASTE_KEY_ORDER`, all parsing/serialization, classification methods |
| `chat/categorized_handoff.py` | 337-434 | `CategorizedHandoffBuilder.normalize_questionnaire()`, `build_handoff_categories_v1()`, `_classify_aroma_taste_flags()` |
| `profiles/cretskens/aroma_taste_classifier.txt` | 1-19 | Entire prompt template |
| `profiles/cretskens/domain_profile.yml` | 37 | `aroma_taste_classifier_path` reference |
| `src/pixie_questioner/config.py` | 67, 131, 372, 387, 431, 449, 457, 472 | Config field and loading |
| `src/pixie_questioner/app_runtime.py` | 124-128 | LLM invoker construction |
| `src/pixie_questioner/chat_service.py` | 35, 65 | Service field, builder injection |
| `profiles/cretskens/questionnaire_notulist.txt` | 116-120 | Notulist field-8 group hints (keep) |

### What Already Works Correctly Without the Classifier

The following downstream components work with **free text** and do NOT consume
boolean flags:

1. **`retrieval.py` → `rank_catalog_records()`**: Tokenizes the working context
   and matches against `aroma_description` and `taste_description` free text in
   the catalog. Free-text preferences like "fris en fruitig" produce useful token
   overlap.

2. **`llm_ranker.py` → `LLMWineRanker`**: Sends full catalog records (including
   `aroma_description` and `taste_description`) to the LLM alongside the working
   context. The LLM can match natural language preferences against natural language
   descriptions.

3. **`turn_controller_prompt.txt`**: Reads the working context as text. Natural
   language preferences are more interpretable than boolean flag JSON.

---

## 7. Redesign Proposal

### 7.1 Core Change: Remove the Boolean Classification Layer

**Remove the entire aroma_taste_classifier subsystem** and let field 8 flow
through as free text. The notulist already formats field 8 appropriately (short
free-text summary, close to the aroma/taste vocabulary).

### 7.2 Specific Changes

#### A. `CategorizedHandoffBuilder.normalize_questionnaire()` — Simplify

**Current behavior:** Extracts field 8, converts to boolean flags, serializes
back.

**New behavior:** No-op for field 8 (the notulist output is already suitable).
Either remove the `normalize_questionnaire()` method or make it skip field 8
normalization.

#### B. `CategorizedHandoffBuilder.build_handoff_categories_v1()` — Remove aroma flags

**Current behavior:** Merges boolean aroma/taste flags into the handoff
categories JSON.

**New behavior:** Keep the handoff categories for foodpair, kleur, land, regio,
prijs (these still map to actual catalog columns). For aroma/taste, include
the free-text field 8 value as-is under a descriptive key (e.g.
`"geur_smaak_voorkeur": "fris en fruitig"`), or omit it from the categories
entirely since it's already in the questionnaire text which is also included in
the working context.

#### C. Remove the aroma_taste_classifier LLM invoker

- Remove `aroma_taste_classifier.txt` prompt (or leave unused)
- Remove `aroma_taste_classifier_path` from `domain_profile.yml`
- Remove `aroma_taste_classifier` field from `ChatService`
- Remove LLM invoker construction in `app_runtime.py`
- Remove `aroma_taste_classifier` field from `CategorizedHandoffBuilder`

#### D. Remove parsing infrastructure

Remove from `categorized_handoff.py`:
- `AROMA_TASTE_KEY_ORDER` constant
- `_NORMALIZED_AROMA_KEYS` dict
- `_AROMA_SUFFIX_MAP` dict
- `parse_aroma_taste_flags()`
- `_extract_bool_value()`
- `_candidate_aroma_keys()`
- `_extract_flags_from_mapping()`
- `_extract_flags_from_text()`
- `serialize_aroma_taste_flags()`

#### E. Keep the notulist field-8 instructions

The notulist prompt (lines 112-120) already guides the LLM to store a concise
free-text summary using vocabulary aligned with the aroma/taste groups. **This
should remain unchanged.** It produces exactly the right format for downstream
text matching and LLM-based ranking.

#### F. Optionally include free-text preference in handoff categories

If downstream consumers benefit from seeing the aroma/taste preference separately
from the full questionnaire, add a simple string key to the handoff categories:

```json
{
  "foodpair_roodvlees": true,
  "kleur": "rood",
  "land": "Frankrijk",
  "prijs": 25,
  "geur_smaak_voorkeur": "fris en fruitig"
}
```

This is a plain passthrough of field 8 — no LLM call, no boolean conversion.

### 7.3 What This Changes in the Flow

```
BEFORE:
User: "fris en fruitig"
  → notulist: "fris en fruitig" in field 8
  → normalize_questionnaire: {"smaak_fris_zuur":true,"smaak_fruitig":true}
  → handoff_categories_v1: {"smaak_fris_zuur":true,"smaak_fruitig":true,...}
  → working_context: "Handoff categories:\n{\"smaak_fris_zuur\":true,...}"
  → retrieval: tokens ["smaak_fris_zuur","smaak_fruitig"] → poor match
  → ranker LLM: sees boolean flags → suboptimal matching

AFTER:
User: "fris en fruitig"
  → notulist: "fris en fruitig" in field 8
  → (no normalization)
  → handoff_categories_v1: {"geur_smaak_voorkeur":"fris en fruitig",...}
  → working_context: "...geur_smaak_voorkeur: fris en fruitig..."
  → retrieval: tokens ["fris","fruitig"] → matches "frisse agrum", "bosvruchten"
  → ranker LLM: sees "fris en fruitig" → natural matching against descriptions
```

### 7.4 Migration Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| `handoff_categories_v1` schema change | Medium | The schema already contains mixed types (bool, str, int). Adding a string key for aroma/taste is backward-compatible. Omitting the aroma flags entirely is also safe since no downstream code queries individual boolean aroma keys. |
| Existing state in flight | Low | Active sessions with boolean flags in field 8 will still work — the questionnaire text is also included in working_context. |
| Test changes | Medium | Tests that mock the aroma_taste_classifier or assert boolean flag output will need updating. ~15 test references in `test_chat_service.py`. |
| Notulist output quality | Low | The notulist is already instructed to produce the right format. No prompt change needed. |

---

## 8. Tests to Add/Update

### New regression test
- User answers "fris en fruitig" for field 8 → verify field 8 retains free text
  (not converted to boolean JSON)
- Verify handoff_categories_v1 does NOT contain `smaak_fris_zuur` or similar
  boolean keys
- Verify working_context passed to wine advisor contains the free-text preference

### Update existing tests
- All tests in `test_chat_service.py` referencing `_make_aroma_taste_classifier`
  or asserting boolean aroma flags in handoff output
- Config tests asserting `aroma_taste_classifier_prompt` is loaded

---

## 9. Recommendation

**Proceed with the redesign.** The boolean aroma/taste classification layer:

1. Is based on a data model that no longer exists
2. Actively degrades recommendation quality by destroying useful free text
3. Adds unnecessary LLM calls (latency + cost)
4. Introduces fragile parsing that causes runtime errors
5. Is fully redundant — the notulist + downstream LLM already handle this correctly

The change is a net removal of ~200 lines of code, 1 prompt template, and up to
4 LLM calls per conversation, with improved recommendation quality as a bonus.

---

## 10. Files Summary (Proposed Changes)

| Action | File | Detail |
|--------|------|--------|
| **Remove** | `profiles/cretskens/aroma_taste_classifier.txt` | Entire file |
| **Edit** | `profiles/cretskens/domain_profile.yml` | Remove `aroma_taste_classifier_path` |
| **Edit** | `src/pixie_questioner/chat/categorized_handoff.py` | Remove AROMA_TASTE_KEY_ORDER, all aroma parsing/serialization, simplify normalize_questionnaire and build_handoff_categories_v1 |
| **Edit** | `src/pixie_questioner/chat_service.py` | Remove `aroma_taste_classifier` field |
| **Edit** | `src/pixie_questioner/app_runtime.py` | Remove aroma_taste_classifier invoker construction |
| **Edit** | `src/pixie_questioner/config.py` | Remove `aroma_taste_classifier_prompt` field and loading |
| **Keep** | `profiles/cretskens/questionnaire_notulist.txt` | No changes — field 8 instructions are already correct |
| **Keep** | `profiles/cretskens/question_asker.txt` | No changes |
| **Update** | `tests/test_chat_service.py` | Update ~15 test references |
| **Update** | `tests/test_config.py` | Update config test assertions |
| **Add** | `tests/` | Regression test for free-text field 8 passthrough |
