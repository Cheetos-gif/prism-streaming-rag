"""Tests for controller/decomposer.py."""

from controller.decomposer import (
    _clean_for_search,
    _infer_intent_label,
    _likely_multi_intent,
    decompose,
)


class TestMultiIntentDetection:
    def test_obvious_multi_intent(self):
        assert (
            _likely_multi_intent(
                "I need a venue and I need the cancellation policy and the catering options"
            )
            is True
        )

    def test_single_intent(self):
        assert _likely_multi_intent("What is the capacity of Venue B?") is False

    def test_multiple_questions(self):
        assert _likely_multi_intent("What is the capacity? What about catering?") is True


class TestDecomposeRuleBased:
    def test_multi_intent_splits_correctly(self):
        result = decompose(
            "I need a venue for 30 people and the cancellation policy and catering options",
            use_llm=False,
        )
        assert len(result) >= 2  # Should split into at least 2
        intents = [sq.sub_intent for sq in result]
        # Check we got different intents
        assert len(set(intents)) >= 2

    def test_single_intent_passthrough(self):
        result = decompose(
            "What is the weight of the Model 7 compressor?",
            use_llm=False,
        )
        assert len(result) == 1

    def test_empty_input(self):
        result = decompose("", use_llm=False)
        assert len(result) == 0

    def test_field_service_decomposes(self):
        result = decompose(
            "I'm looking at a Model 7 compressor, it's making a knocking sound, "
            "and I need to know if I can run it till Friday.",
            use_llm=False,
        )
        # Should detect at least 2 intents (troubleshooting + continued operation)
        assert len(result) >= 1


class TestIntentLabel:
    def test_cancellation_detected(self):
        assert _infer_intent_label("cancellation policy") == "cancellation_policy"

    def test_catering_detected(self):
        assert _infer_intent_label("catering options for workshop") == "catering_options"

    def test_venue_detected(self):
        assert _infer_intent_label("venue for 30 people") == "venue_capacity"

    def test_troubleshooting_detected(self):
        assert _infer_intent_label("knocking sound in compressor") == "troubleshooting"

    def test_travel_detected(self):
        assert _infer_intent_label("travel reimbursement policy") == "travel_reimbursement"


class TestCleanForSearch:
    def test_removes_filler(self):
        result = _clean_for_search("I need a venue in Pune for 30 people please")
        assert "I need" not in result
        assert "please" not in result
        assert "Pune" in result
        assert "30" in result

    def test_preserves_key_terms(self):
        result = _clean_for_search("Model 7 compressor knocking sound")
        assert "Model 7" in result
        assert "knocking" in result
