"""Tests for controller/retrieval_controller.py."""
from controller.retrieval_controller import RetrievalController
from controller.stream_simulator import TranscriptChunk


def _chunk(t: float, text: str, final: bool = False) -> TranscriptChunk:
    return TranscriptChunk(timestamp_s=t, text=text, is_final=final)


class TestWaitDecision:
    def test_incomplete_thought_waits(self):
        ctrl = RetrievalController()
        decision = ctrl.on_chunk(_chunk(0.0, "I need to plan a"))
        assert decision.action == "wait"

    def test_single_word_waits(self):
        ctrl = RetrievalController()
        decision = ctrl.on_chunk(_chunk(0.0, "Hello"))
        assert decision.action == "wait"


class TestRetrieveDecision:
    def test_entities_trigger_retrieve(self):
        ctrl = RetrievalController()
        decision = ctrl.on_chunk(
            _chunk(0.8, "I need a venue in Pune for 30 people.")
        )
        assert decision.action == "retrieve"

    def test_utterance_end_always_retrieves(self):
        ctrl = RetrievalController()
        ctrl.on_chunk(_chunk(0.0, "Tell me about Model 7 compressor"))
        decision = ctrl.on_chunk(_chunk(2.0, "[Utterance End]", final=True))
        assert decision.action == "retrieve"

    def test_question_with_entities_retrieves(self):
        ctrl = RetrievalController()
        decision = ctrl.on_chunk(
            _chunk(0.5, "What is the capacity of Venue B in Pune?")
        )
        assert decision.action == "retrieve"


class TestSuppressDecision:
    def test_repeat_request_suppressed(self):
        ctrl = RetrievalController()
        decision = ctrl.on_chunk(
            _chunk(0.0, "Please repeat your last answer in two bullets.")
        )
        assert decision.action == "suppress"
        assert "presentation" in decision.reason

    def test_shorten_request_suppressed(self):
        ctrl = RetrievalController()
        decision = ctrl.on_chunk(
            _chunk(0.0, "Make that shorter please")
        )
        assert decision.action == "suppress"

    def test_reformat_suppressed(self):
        ctrl = RetrievalController()
        decision = ctrl.on_chunk(
            _chunk(0.0, "Can you reformat that as bullet points?")
        )
        assert decision.action == "suppress"


class TestReretrieveDecision:
    def test_late_constraint_triggers_reretrieve(self):
        ctrl = RetrievalController()
        ctrl.mark_answered()
        decision = ctrl.on_chunk(
            _chunk(3.0, "Actually, make that 50 people not 30.")
        )
        assert decision.action == "reretrieve"

    def test_correction_triggers_reretrieve(self):
        ctrl = RetrievalController()
        ctrl.mark_answered()
        decision = ctrl.on_chunk(
            _chunk(3.0, "Sorry, the trip was international, not domestic.")
        )
        assert decision.action == "reretrieve"

    def test_refinement_only_after_answer(self):
        """Before an answer is given, 'actually' should not trigger reretrieve."""
        ctrl = RetrievalController()
        # NOT marked as answered
        decision = ctrl.on_chunk(
            _chunk(0.5, "Actually I need a venue in Pune for 50 people.")
        )
        # Should be retrieve, not reretrieve
        assert decision.action != "reretrieve"


class TestBufferAccumulation:
    def test_buffer_accumulates_across_chunks(self):
        ctrl = RetrievalController()
        ctrl.on_chunk(_chunk(0.0, "I need"))
        ctrl.on_chunk(_chunk(0.5, "a venue"))
        assert "I need" in ctrl.accumulated_text
        assert "a venue" in ctrl.accumulated_text

    def test_reset_clears_buffer(self):
        ctrl = RetrievalController()
        ctrl.on_chunk(_chunk(0.0, "some text"))
        ctrl.reset_buffer()
        assert ctrl.accumulated_text == ""

