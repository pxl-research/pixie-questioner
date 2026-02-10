## Why

De huidige flow rond samenvatting en feedback is instabiel: samenvattingen verschijnen soms niet, verschijnen te vroeg, of de sessie sluit voordat feedback mogelijk is. Dit veroorzaakt regressies in de UI en maakt verdere uitbreidingen risicovol. We moeten de flow contract-first en state-machine-gedreven maken.

## What Changes

- Definieer een expliciete conversatie-flow met duidelijke fasen: intake, summary-presented, awaiting-feedback, completed.
- Garandeer dat de samenvatting exact eenmaal wordt gegenereerd en consistent wordt teruggegeven tijdens de feedbackfase.
- Garandeer dat de feedbackvraag altijd wordt gesteld na de samenvatting, maar dat antwoorden niet verplicht zijn om de flow te begrijpen.
- Beperk feedbackinterpretatie tot ja/nee-semantiek voor eenvoudige verwerking in state.
- Maak API-responsevelden en state-overgangen expliciet zodat de frontend zonder heuristieken kan renderen.
- Verwijder tijdelijke/adhoc gedragspaden; geen backward-compat fallback nodig omdat dit nog niet gedeployed is.

## Capabilities

### New Capabilities
- `post-summary-feedback-flow`: Contractuele state-machine en API-flow voor samenvatting gevolgd door optionele ja/nee feedback.

### Modified Capabilities
- *(none)*

## Impact

- Affected code:
  - `src/pixie_questioner/chat_service.py`
  - `src/pixie_questioner/schemas.py`
  - `apps/api/main.py`
- Affected tests:
  - `tests/test_chat_service.py`
  - `tests/test_api.py`
  - `tests/test_live_llm_smoke.py`
- API impact:
  - Response contract voor post-summary fase wordt expliciet en leidend voor frontend-rendering.
- Dependencies/infrastructure:
  - geen nieuwe dependencies
  - geen infra- of dockerwijzigingen
