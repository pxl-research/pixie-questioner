## ADDED Requirements

### Requirement: Explicit post-summary phase state
The system MUST expose an explicit post-summary phase in `state` so clients can render and control flow without heuristics.

#### Scenario: Enter feedback phase after questionnaire completion
- **WHEN** the questionnaire becomes complete or the user requests stop during intake
- **THEN** the system MUST set state phase to `awaiting_feedback`
- **AND THEN** the system MUST include a normalized feedback status of `pending`

#### Scenario: Enter completed phase after valid feedback
- **WHEN** the system is in `awaiting_feedback` and receives a valid yes/no feedback answer
- **THEN** the system MUST set state phase to `completed`
- **AND THEN** the system MUST set feedback status to `yes` or `no`

### Requirement: Summary-first response contract
On the first response after intake completion, the system MUST return summary content and feedback prompt as separate payload fields in fixed order semantics.

#### Scenario: First post-intake response includes all render blocks
- **WHEN** intake transitions to post-summary feedback
- **THEN** the response MUST contain `text` with the handoff message
- **AND THEN** the response MUST contain `summary` with the generated final summary
- **AND THEN** the response MUST contain `follow_up_question` with the yes/no feedback question

#### Scenario: Flow remains open while feedback is pending
- **WHEN** the system returns the first post-intake response with pending feedback
- **THEN** the response type MUST keep the conversation open for further user input

### Requirement: Summary immutability during feedback phase
The final summary MUST be generated once at transition to post-summary feedback and MUST remain unchanged through all pending feedback turns.

#### Scenario: Repeated feedback turns keep the same summary
- **WHEN** feedback is pending and the user sends additional messages
- **THEN** the response MUST keep returning the same summary value
- **AND THEN** the system MUST NOT regenerate or rewrite the final summary content

### Requirement: Yes/no-only feedback interpretation
The system MUST only persist feedback as normalized `yes` or `no` values.

#### Scenario: Recognized affirmative response
- **WHEN** feedback phase is active and user input is interpreted as affirmative
- **THEN** the system MUST persist feedback status as `yes`

#### Scenario: Recognized negative response
- **WHEN** feedback phase is active and user input is interpreted as negative
- **THEN** the system MUST persist feedback status as `no`

#### Scenario: Unrecognized response keeps feedback pending
- **WHEN** feedback phase is active and user input is not recognized as yes or no
- **THEN** the system MUST keep feedback status as `pending`
- **AND THEN** the system MUST ask the same yes/no feedback question again

### Requirement: Feedback question is always asked
The system MUST always ask for reuse feedback after presenting the final summary, while not requiring a response to consider the summary delivered.

#### Scenario: Summary is delivered regardless of feedback completion
- **WHEN** intake completes
- **THEN** the system MUST present the summary even before any feedback answer is provided

#### Scenario: Feedback remains optional for user session continuity
- **WHEN** a user does not provide a valid feedback answer
- **THEN** the system MUST keep the session in `awaiting_feedback` without dropping or hiding the summary

### Requirement: Language consistency for handoff and feedback prompt
The handoff message and feedback question MUST use the same selected conversation language for the post-summary response.

#### Scenario: Dutch post-summary response
- **WHEN** the selected post-summary language is Dutch
- **THEN** `text` and `follow_up_question` MUST both be Dutch

#### Scenario: Non-Dutch post-summary response
- **WHEN** the selected post-summary language is English, French, or another strongly-detected language
- **THEN** `text` and `follow_up_question` MUST both be in that same language
