## Why

`src/pixie_questioner/chat_service.py` combineert momenteel meerdere verantwoordelijkheden in één klasse:

- state-machine en state-normalisatie
- taaldetectie en taalkeuze-heuristieken
- domeinpolicy/localized copy selectie
- LLM-orchestratie (question-asker, notulist, summarizer, relevance)

Dat maakt regressies waarschijnlijker, testdekking moeilijker te interpreteren en wijzigingen risicovol. We willen dit opsplitsen met behoud van exact hetzelfde API/state-gedrag.

## What Changes

- Splits `chat_service.py` op in kleinere interne modules per verantwoordelijkheid.
- Maak `ChatService` een dunne orchestrator/facade die losse componenten aanroept.
- Behoud publieke contracten:
  - `POST /chat` request/response ongewijzigd
  - `state` structuur ongewijzigd
  - bestaande conversational rules ongewijzigd
- Houd migratie incrementeel: gedrag-pariteit moet per stap aantoonbaar zijn via tests.

## Simplicity Guardrails

- Introduceer alleen abstrahering die direct leesbaarheid verhoogt; geen “future-proof” lagen.
- `flow.py` is optioneel en wordt enkel toegevoegd als `ask()` na eerdere extracties nog moeilijk leesbaar blijft.
- Gebruik bij voorkeur pure functies boven extra classes wanneer er geen eigen state nodig is.
- Vermijd generic frameworks/patterns (registries, base classes, factories) voor dit domein.
- Houd modulegrenzen pragmatisch: als een extractie enkel doorstuurt zonder eigen logica, dan terug inline.

## Proposed Modules

- `src/pixie_questioner/chat/types.py`
  - `ChatState`, `ChatResult`, fase- en feedback-constanten
- `src/pixie_questioner/chat/policy.py`
  - `ChatDomainPolicy`, `ChatQuickReply`
- `src/pixie_questioner/chat/state_codec.py`
  - parse/serialize + state-validatie + defaults
- `src/pixie_questioner/chat/language.py`
  - taaldetectie en preferred-language selectie
- `src/pixie_questioner/chat/messages.py`
  - localized message/selectie, quick replies, term replacements
- `src/pixie_questioner/chat/stop_detection.py`
  - stop phrase matching logic
- `src/pixie_questioner/chat/relevance.py`
  - `LLMAnswerRelevanceClassifier`
- `src/pixie_questioner/chat/flow.py` (optioneel)
  - pure flow/transitiebeslissingen voor intake/off-topic/feedback/completed

## Impact

- Primair geraakt:
  - `src/pixie_questioner/chat_service.py`
  - nieuwe map `src/pixie_questioner/chat/`
- Testimpact:
  - `tests/test_chat_service.py`
  - eventueel beperkte updates in `tests/test_api.py` en `tests/test_live_llm_smoke.py` bij importwijzigingen

## Out Of Scope

- Geen API-schemawijzigingen
- Geen promptinhoud-wijzigingen
- Geen nieuwe dependencies
- Geen wijziging aan `domain_profile.yml` contract
