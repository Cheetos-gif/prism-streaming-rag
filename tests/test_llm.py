"""Unit tests for shared/llm.py."""

from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock, patch

import httpx
import pytest

from shared import llm
from shared.llm import ClientError, ServerError


@pytest.fixture(autouse=True)
def reset_llm_state(monkeypatch):
    """Ensure clean LLM client and config state for every test."""
    llm.reset_client()
    monkeypatch.setattr(llm, "DEFAULT_RETRY_DELAY_SECONDS", 0.0)
    yield
    llm.reset_client()


def test_missing_api_key_raises_clear_error(monkeypatch):
    """If GEMINI_API_KEY is not set, raise a clear error message."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY environment variable is not set"):
        llm.get_client()


def test_model_name_resolution(monkeypatch):
    """Test default model, env var override, and explicit argument override."""
    monkeypatch.setenv("MODEL_PROVIDER", "gemini")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    assert llm.get_model_name() == "gemini-2.0-flash"

    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-pro")
    assert llm.get_model_name() == "gemini-2.5-pro"

    assert llm.get_model_name("gemini-custom") == "gemini-custom"


def test_generate_success(monkeypatch, caplog):
    """Test successful plain text generation and logging."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "This is a summary."
    mock_client.models.generate_content.return_value = mock_response

    llm.set_client(mock_client)

    with caplog.at_level(logging.INFO):
        result = llm.generate(
            "Summarize this document...", system_instruction="Be concise", temperature=0.3
        )

    assert result == "This is a summary."
    mock_client.models.generate_content.assert_called_once()
    call_kwargs = mock_client.models.generate_content.call_args.kwargs
    assert call_kwargs["model"] == "gemini-2.0-flash"
    assert call_kwargs["contents"] == "Summarize this document..."
    assert call_kwargs["config"].temperature == 0.3
    assert call_kwargs["config"].system_instruction == "Be concise"
    assert call_kwargs["config"].response_mime_type is None

    # Check logging
    log_messages = [rec.message for rec in caplog.records]
    assert any(
        "prompt_len=" in msg and "response_len=" in msg and "latency_ms=" in msg
        for msg in log_messages
    )


def test_generate_json_success_dict(monkeypatch):
    """Test generate_json returning a parsed dict."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = json.dumps({"intent": "query", "confidence": 0.95})
    mock_client.models.generate_content.return_value = mock_response

    llm.set_client(mock_client)

    result = llm.generate_json("Analyze intent: hello")
    assert isinstance(result, dict)
    assert result == {"intent": "query", "confidence": 0.95}

    call_kwargs = mock_client.models.generate_content.call_args.kwargs
    assert call_kwargs["config"].response_mime_type == "application/json"
    assert call_kwargs["config"].temperature == 0.1


def test_generate_json_success_list_with_markdown_fence(monkeypatch):
    """Test generate_json parsing a list wrapped in markdown ```json ... ``` code fence."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = '```json\n["item1", "item2", "item3"]\n```'
    mock_client.models.generate_content.return_value = mock_response

    llm.set_client(mock_client)

    result = llm.generate_json("List items")
    assert isinstance(result, list)
    assert result == ["item1", "item2", "item3"]


def test_generate_json_non_dict_or_list_raises_value_error(monkeypatch):
    """Test that a scalar JSON response (e.g. 42 or 'hello') raises ValueError."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "42"
    mock_client.models.generate_content.return_value = mock_response

    llm.set_client(mock_client)

    with pytest.raises(ValueError, match="Expected JSON object or array"):
        llm.generate_json("Give me a number")


def test_retry_on_transient_server_error(monkeypatch, caplog):
    """Transient 503 ServerError retries once and succeeds on the second attempt."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Recovered response"

    server_err = ServerError(
        503, {"error": {"code": 503, "message": "Service unavailable", "status": "UNAVAILABLE"}}
    )
    mock_client.models.generate_content.side_effect = [server_err, mock_response]

    llm.set_client(mock_client)

    with caplog.at_level(logging.WARNING):
        result = llm.generate("Hello")

    assert result == "Recovered response"
    assert mock_client.models.generate_content.call_count == 2
    assert any("transient error" in rec.message.lower() for rec in caplog.records)


def test_retry_on_transient_rate_limit(monkeypatch, caplog):
    """Transient 429 ClientError retries once and succeeds."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = '{"status": "ok"}'

    rate_limit_err = ClientError(
        429, {"error": {"code": 429, "message": "Quota exceeded", "status": "RESOURCE_EXHAUSTED"}}
    )
    mock_client.models.generate_content.side_effect = [rate_limit_err, mock_response]

    llm.set_client(mock_client)

    result = llm.generate_json("Status check")
    assert result == {"status": "ok"}
    assert mock_client.models.generate_content.call_count == 2


def test_retry_on_httpx_network_error(monkeypatch):
    """Transient httpx.RequestError retries once and succeeds."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Recovered"

    network_err = httpx.ConnectError("Connection failed")
    mock_client.models.generate_content.side_effect = [network_err, mock_response]

    llm.set_client(mock_client)

    result = llm.generate("Hello network")
    assert result == "Recovered"
    assert mock_client.models.generate_content.call_count == 2


def test_persistent_client_error_not_retried(monkeypatch):
    """Persistent 400 Bad Request error raises immediately without retry."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    mock_client = MagicMock()
    bad_request_err = ClientError(
        400, {"error": {"code": 400, "message": "Bad request", "status": "INVALID_ARGUMENT"}}
    )
    mock_client.models.generate_content.side_effect = bad_request_err

    llm.set_client(mock_client)

    with pytest.raises(ClientError):
        llm.generate("Invalid prompt")

    assert mock_client.models.generate_content.call_count == 1


def test_persistent_transient_error_fails_after_retry(monkeypatch):
    """If a transient error occurs twice, it is raised after the second attempt."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    mock_client = MagicMock()
    server_err = ServerError(
        500, {"error": {"code": 500, "message": "Internal error", "status": "INTERNAL"}}
    )
    mock_client.models.generate_content.side_effect = server_err

    llm.set_client(mock_client)

    with pytest.raises(ServerError):
        llm.generate("Fail twice")

    assert mock_client.models.generate_content.call_count == 2


def test_generate_json_retries_on_malformed_json(monkeypatch):
    """If model returns malformed JSON on attempt 1, retries once."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    mock_client = MagicMock()
    bad_json_response = MagicMock()
    bad_json_response.text = "not a valid json {"
    good_json_response = MagicMock()
    good_json_response.text = '{"fixed": true}'

    mock_client.models.generate_content.side_effect = [bad_json_response, good_json_response]

    llm.set_client(mock_client)

    result = llm.generate_json("Return JSON")
    assert result == {"fixed": True}
    assert mock_client.models.generate_content.call_count == 2


def test_free_local_mode_generate_json(monkeypatch):
    """Test 100% free offline local provider without any API keys."""
    monkeypatch.setenv("MODEL_PROVIDER", "local")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    prompt = 'Decompose this utterance into independent sub-queries:\n\n"I need a venue in Pune and catering"'
    result = llm.generate_json(prompt)
    assert isinstance(result, list)
    assert len(result) >= 1
    assert "sub_intent" in result[0]
    assert "search_query" in result[0]


def test_free_groq_mode_mock_call(monkeypatch):
    """Test free Groq provider using OpenAI-compatible format."""
    monkeypatch.setenv("MODEL_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_mock_key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with patch.object(
        llm, "_generate_openai_compatible", return_value='{"groq_decomposed": true}'
    ) as mock_call:
        res = llm.generate_json("Decompose this prompt")
        assert res == {"groq_decomposed": True}
        mock_call.assert_called_once()
        kwargs = mock_call.call_args.kwargs
        assert kwargs["base_url"] == "https://api.groq.com/openai/v1"
        assert kwargs["api_key"] == "gsk_mock_key"
        assert kwargs["is_json"] is True
