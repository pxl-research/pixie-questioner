## Context

De huidige implementatie werkt functioneel, maar heeft hoge cognitieve complexiteit. Kleine veranderingen raken snel meerdere paden tegelijk (state, taal, policy, LLM-calls), waardoor review en debugging zwaar worden.

Doel: hetzelfde gedrag behouden, maar de code zo structureren dat elke verantwoordelijkheid apart testbaar en leesbaar is.

## Goals / Non-Goals

**Goals**

- Gedragspariteit met huidige flow.
- Lagere complexiteit per bestand/functie.
- Duidelijke scheiding tussen:
  - state parsing/serialisatie
  - business flow
  - infrastructuur/LLM calls
  - policy/localization lookup
- Betere testisolatie per domeinregel.

**Non-Goals**

- Geen functionele feature-uitbreidingen.
- Geen wijziging van response- of state-contract.
- Geen aanpassing van promptcopy of domeinpolicy-inhoud.

## Clean Code Principles (Binding)

- Eerst expliciete leesbaarheid, dan pas extra abstrahering.
- Functies met één duidelijke verantwoordelijkheid; vermijd “god helpers”.
- Geen nieuwe classes zonder eigen state of levenscyclus.
- Geen import-cycli; afhankelijkheden moeten richting `ChatService` lopen, niet omgekeerd.
- “Rule of 2”: abstraheer pas als logica op minstens twee plaatsen echt nodig is.
- Geen pass-through wrappers die alleen parameters doorgeven zonder gedrag.
- Namen moeten domeintaal spreken (`phase`, `feedback`, `off_topic`) en geen generieke technische labels.

## Target Architecture

### 1) Facade laag

`ChatService` blijft publiek entrypoint met `ask(...)`, maar doet enkel:

- input/state decode aanroepen
- relevante flowstap kiezen
- externe services (`notulist`, `summarizer`, LLM question-asker) triggeren
- output/state encode teruggeven

### 2) Core flow laag

`flow.py` bevat de domeintransities en return-intenties (zonder transportdetail):

- `terminated`
- `awaiting_feedback`
- `completed`
- `intake` on-topic/off-topic
- questionnaire complete vs niet complete

### 3) State codec laag

`state_codec.py` beheert:

- parse van ruwe `dict[str, str]`
- type/waarde-validatie
- defaults en normalisaties
- serialize naar API-state

### 4) Language laag

`language.py` beheert:

- candidate detectie
- supported language selectie
- onzekerheidsheuristieken
- preferred language beslissing

### 5) Messaging/policy laag

`messages.py` beheert:

- localized text fallback (`lang` -> prefix -> `en` -> first non-empty)
- quick reply fallback
- handoff/feedback/goodbye/off-topic builder
- technical term replacements

### 6) Stop/relevance utilities

- `stop_detection.py`: tokeniseren + phrase matching
- `relevance.py`: LLM classifier + ON_TOPIC/OFF_TOPIC parsing

## Dependency Direction Rules

- `chat_service.py` mag alle chat-submodules importeren.
- `types.py` en `policy.py` importeren geen andere chat-submodules.
- `state_codec.py` mag enkel `types.py` gebruiken.
- `messages.py` mag enkel `policy.py` gebruiken.
- `language.py` en `stop_detection.py` blijven zelfstandig (geen project-lokale imports nodig).
- `relevance.py` mag zelfstandig blijven; geen import van `ChatService`.
- `flow.py` (indien toegevoegd) mag enkel `types.py` gebruiken en geen I/O doen.

## Data Flow Per Turn

1. `ChatService.ask()` decodeert state via `state_codec`.
2. `flow` beslist in welke hoofdroute de beurt valt.
3. Indien nodig worden utilities aangeroepen:
   - `stop_detection`
   - `relevance`
   - `language`
   - `messages`
4. Bij intake update:
   - notulist update questionnaire
   - summarizer update summary
   - vraaggenerator produceert volgende vraag
5. `state_codec` serialiseert final state.

## Migration Strategy (Low Risk)

### Phase 1: Mechanical extraction

- Verplaats dataclasses/protocols en pure helpers naar nieuwe modules.
- Laat `ChatService` dezelfde method-signatures houden.
- Geen behavior changes.

### Phase 2: Flow isolation

- Evalueer eerst of `ask()` na eerdere extracties nog te complex is.
- Verplaats enkel dan conditionele flowstukken naar `flow.py` helpers.
- Houd LLM-calls in facade; enkel beslislogica eruit.

### Phase 3: Cleanup and hardening

- Verwijder dode code en duplicaties.
- Voeg gerichte regressietests toe voor eerder buggevoelige paden.

## Risks / Mitigations

- **Risk:** subtiele state-regressie.
  - **Mitigation:** snapshot-achtige regressietests op state-velden per fase.
- **Risk:** taalkeuze verandert onverwacht.
  - **Mitigation:** bestaande language tests behouden, extra edge-case tests toevoegen.
- **Risk:** fallback-copy wijzigt stilzwijgend.
  - **Mitigation:** tests op exact gekozen message set per language/fallbackpad.

## Acceptance Criteria

- Alle bestaande tests blijven groen zonder contractwijzigingen.
- `ChatService` bevat enkel orchestratie (geen grote utilityblokken meer).
- Elke nieuwe module heeft minstens één directe unit test.
- Geen verschil in API payloads voor bestaande scenario’s.
- Geen module die enkel pass-through code bevat zonder eigen gedrag.
- `flow.py` bestaat alleen als die effectief complexiteit verlaagt.
