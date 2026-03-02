from pixie_questioner.chat.flow import (
    ensure_feedback_context,
    mark_feedback_completed,
    to_categorized_handoff_completed_state,
    to_feedback_state,
    to_off_topic_redirect_state,
    to_terminated_state,
)
from pixie_questioner.chat.types import (
    PHASE_AWAITING_FEEDBACK,
    PHASE_COMPLETED,
    PHASE_INTAKE,
    PHASE_TERMINATED,
    ChatState,
)


def test_to_terminated_state_resets_feedback_data() -> None:
    initial = ChatState(
        questionnaire="q",
        summary="s",
        last_question="l",
        phase=PHASE_INTAKE,
        feedback_status="yes",
        final_summary="final",
        post_summary_language="nl",
        boundary_violation_count=2,
    )

    terminated = to_terminated_state(initial, language="fr")

    assert terminated.phase == PHASE_TERMINATED
    assert terminated.feedback_status == "pending"
    assert terminated.final_summary == ""
    assert terminated.post_summary_language == "fr"
    assert terminated.boundary_violation_count == 2


def test_to_off_topic_redirect_state_keeps_counter_and_resets_feedback() -> None:
    initial = ChatState(
        questionnaire="q",
        summary="s",
        last_question="l",
        phase=PHASE_TERMINATED,
        feedback_status="yes",
        final_summary="final",
        post_summary_language="fr",
        boundary_violation_count=3,
    )

    redirected = to_off_topic_redirect_state(initial)

    assert redirected.phase == PHASE_INTAKE
    assert redirected.feedback_status == "pending"
    assert redirected.final_summary == ""
    assert redirected.post_summary_language == ""
    assert redirected.boundary_violation_count == 3


def test_to_feedback_state_stores_summary_and_language() -> None:
    initial = ChatState(
        questionnaire="old",
        summary="old summary",
        last_question="last question",
        boundary_violation_count=1,
    )

    feedback = to_feedback_state(
        state=initial,
        updated_questionnaire="new q",
        updated_summary="new summary",
        language="en",
    )

    assert feedback.phase == PHASE_AWAITING_FEEDBACK
    assert feedback.questionnaire == "new q"
    assert feedback.summary == "new summary"
    assert feedback.final_summary == "new summary"
    assert feedback.post_summary_language == "en"
    assert feedback.boundary_violation_count == 1


def test_feedback_helpers_complete_and_fill_missing_context() -> None:
    initial = ChatState(
        summary="stored summary",
        phase=PHASE_AWAITING_FEEDBACK,
        feedback_status="pending",
        final_summary="",
        post_summary_language="",
    )

    enriched, summary_text, language = ensure_feedback_context(
        initial,
        detect_language=lambda _: "de",
    )
    completed = mark_feedback_completed(enriched, "yes")

    assert summary_text == "stored summary"
    assert language == "de"
    assert enriched.final_summary == "stored summary"
    assert enriched.post_summary_language == "de"
    assert completed.phase == PHASE_COMPLETED
    assert completed.feedback_status == "yes"


def test_to_categorized_handoff_completed_state_sets_phase_and_payload() -> None:
    initial = ChatState(
        questionnaire="q",
        summary="s",
        last_question="last question",
        phase=PHASE_INTAKE,
        feedback_status="yes",
        final_summary="old-final",
        post_summary_language="fr",
        boundary_violation_count=4,
    )

    completed = to_categorized_handoff_completed_state(
        state=initial,
        updated_questionnaire="new q",
        updated_summary="new summary",
        handoff_categories_v1='{"course_type":"hoofdgerecht"}',
    )

    assert completed.phase == PHASE_COMPLETED
    assert completed.questionnaire == "new q"
    assert completed.summary == "new summary"
    assert completed.feedback_status == "pending"
    assert completed.final_summary == ""
    assert completed.post_summary_language == ""
    assert completed.handoff_categories_v1 == '{"course_type":"hoofdgerecht"}'
    assert completed.boundary_violation_count == 0
