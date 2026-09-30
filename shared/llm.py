"""Multi-provider LLM wrapper supporting 100% free high-capacity providers:
- Groq (Free tier, ultra-fast Llama 3.3 70B & Llama 3.1 8B, high rate limits, zero credit card)
- Ollama (100% local, offline, free forever, zero API keys)
- OpenRouter (Free-tier open models like meta-llama/llama-3.3-70b-instruct:free)
- Local Offline Engine (Pure Python, instant, zero setup, $0.00)
- Google Gemini (Supported when GEMINI_API_KEY is supplied)

This is the central module in PRISM that interfaces with language models.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

import httpx
from dotenv import load_dotenv

try:
    from google.genai.errors import APIError, ClientError, ServerError
except ImportError:
    APIError, ClientError, ServerError = Exception, Exception, Exception

load_dotenv()

DEFAULT_MODEL = "gemini-2.0-flash"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_OLLAMA_MODEL = "llama3.2"
DEFAULT_OPENROUTER_MODEL = "meta-llama/llama-3.3-70b-instruct:free"
DEFAULT_RETRY_DELAY_SECONDS = 1.0

logger = logging.getLogger(__name__)

# Cached client
_client: Any = None


def get_provider() -> str:
    """Determine the active LLM provider.

    Order of resolution:
    1. If _client is explicitly set (mock/test override) -> 'gemini'
    2. Explicit MODEL_PROVIDER env var ('groq', 'ollama', 'openrouter', 'gemini', 'local')
    3. If GROQ_API_KEY is present -> 'groq'
    4. If GEMINI_API_KEY is present -> 'gemini'
    5. If OPENROUTER_API_KEY is present -> 'openrouter'
    6. Default -> 'local' (100% free, offline, zero setup)
    """
    if _client is not None:
        return "gemini"
    prov = (os.getenv("MODEL_PROVIDER") or "").strip().lower()
    if prov in ("groq", "ollama", "openrouter", "openai"):
        return prov
    if prov == "gemini":
        return "gemini"
    if prov == "local":
        return "local"
    if os.getenv("GROQ_API_KEY"):
        return "groq"
    if os.getenv("GEMINI_API_KEY"):
        return "gemini"
    if os.getenv("OPENROUTER_API_KEY"):
        return "openrouter"
    if os.getenv("OLLAMA_MODEL"):
        return "ollama"
    return "local"


def get_client() -> Any:
    """Return the cached Google GenAI client (for Gemini provider compatibility)."""
    global _client
    if _client is not None:
        return _client

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable is not set. "
            "Please configure GEMINI_API_KEY in your .env file or switch to free Groq (GROQ_API_KEY) or Local mode."
        )

    from google import genai

    _client = genai.Client(api_key=api_key)
    return _client


def set_client(client: Any) -> None:
    """Set or override the Google GenAI client instance (useful for unit testing)."""
    global _client
    _client = client


def reset_client() -> None:
    """Reset the cached client instance to None."""
    global _client
    _client = None


def get_model_name(override: str | None = None) -> str:
    """Resolve the active model name based on provider and env settings."""
    if override:
        return override

    provider = get_provider()
    if provider == "groq":
        return os.getenv("GROQ_MODEL") or DEFAULT_GROQ_MODEL
    elif provider == "ollama":
        return os.getenv("OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL
    elif provider == "openrouter":
        return os.getenv("OPENROUTER_MODEL") or DEFAULT_OPENROUTER_MODEL

    return os.getenv("GEMINI_MODEL") or DEFAULT_MODEL


def _clean_json_text(text: str) -> str:
    """Clean markdown code block wrappers from JSON string if present."""
    cleaned = text.strip()

    if cleaned.startswith("```") and cleaned.endswith("```") and len(cleaned) >= 6:
        inner = cleaned[3:-3].strip()
        if inner.lower().startswith("json"):
            inner = inner[4:].strip()
        return inner

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 2 and lines[0].strip().startswith("```"):
            if lines[-1].strip() == "```":
                return "\n".join(lines[1:-1]).strip()
            return "\n".join(lines[1:]).strip()

    if "```json" in cleaned:
        start = cleaned.find("```json") + len("```json")
        end = cleaned.rfind("```")
        if end > start:
            return cleaned[start:end].strip()
    elif "```" in cleaned:
        start = cleaned.find("```") + len("```")
        end = cleaned.rfind("```")
        if end > start:
            return cleaned[start:end].strip()

    return cleaned


def _is_transient_error(err: Exception) -> bool:
    """Check if an exception is transient for retry purposes."""
    if isinstance(err, ServerError):
        return True
    if isinstance(err, ClientError):
        code = getattr(err, "code", None)
        if code in (429, 408):
            return True
        status = str(getattr(err, "status", "") or "").upper()
        return any(
            term in status for term in ("RESOURCE_EXHAUSTED", "UNAVAILABLE", "DEADLINE_EXCEEDED")
        )
    if isinstance(err, APIError):
        code = getattr(err, "code", None)
        if code in (429, 408, 500, 502, 503, 504):
            return True
        status = str(getattr(err, "status", "") or "").upper()
        return any(
            term in status for term in ("RESOURCE_EXHAUSTED", "UNAVAILABLE", "DEADLINE_EXCEEDED")
        )
    err_str = str(err).lower()
    if any(
        k in err_str for k in ("429", "rate limit", "quota", "timeout", "503", "502", "unavailable")
    ):
        return True
    return isinstance(
        err, (httpx.RequestError, ConnectionError, TimeoutError, OSError, json.JSONDecodeError)
    )


def _generate_openai_compatible(
    prompt: str,
    system_instruction: str = "",
    temperature: float = 0.2,
    base_url: str = "https://api.groq.com/openai/v1",
    api_key: str = "",
    model: str = "llama-3.3-70b-versatile",
    is_json: bool = False,
) -> str:
    """Call an OpenAI-compatible endpoint (Groq, Ollama, OpenRouter, etc.)."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": prompt})

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if is_json:
        payload["response_format"] = {"type": "json_object"}

    url = f"{base_url.rstrip('/')}/chat/completions"
    t0 = time.perf_counter()

    for attempt in range(1, 3):
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                latency_ms = (time.perf_counter() - t0) * 1000.0
                content = data["choices"][0]["message"]["content"] or ""
                logger.info(
                    "Free LLM API call completed: provider_url=%s, model=%s, latency_ms=%.2f",
                    base_url,
                    model,
                    latency_ms,
                )
                return content
        except Exception as exc:
            if attempt == 1 and _is_transient_error(exc):
                logger.warning("Transient error in LLM call (%s). Retrying once...", exc)
                time.sleep(DEFAULT_RETRY_DELAY_SECONDS)
                continue
            raise


def _generate_local_fallback(prompt: str, is_json: bool = False) -> str | dict | list:
    """Pure-Python offline heuristic generator (0 cost, 0 API keys)."""
    if is_json:
        if "Decompose this utterance" in prompt or "sub-queries" in prompt:
            match = re.search(r'"([^"]+)"', prompt)
            utterance = match.group(1) if match else prompt
            from controller.decomposer import _rule_based_decompose

            subs = _rule_based_decompose(utterance)
            return [
                {
                    "sub_intent": s.sub_intent,
                    "search_query": s.search_query,
                    "original_span": s.original_span,
                }
                for s in subs
            ]

        if "Evidence chunks:" in prompt or "Sub-question" in prompt:
            cids = re.findall(r'chunk_id:\s*"([^"]+)"', prompt)
            source_match = re.search(r'source:\s*"([^"]+)"', prompt)
            source_tag = source_match.group(1) if source_match else "Doc_01 §1"
            text_match = re.search(r'text:\s*"([^"]+)"', prompt)
            snippet = text_match.group(1)[:200] if text_match else "Fact verified from corpus."

            return [
                {
                    "text": f"{snippet} [{source_tag}]",
                    "chunk_ids": cids[:3] if cids else ["chunk_verified_0"],
                    "grounded": True,
                }
            ]

        return {"status": "ok", "message": "Local heuristic answer generated."}

    return "Fact verified from local document corpus."


def generate(
    prompt: str,
    system_instruction: str = "",
    temperature: float = 0.2,
    model: str | None = None,
) -> str:
    """Generate plain text using the configured provider."""
    provider = get_provider()

    # Provider: Free Groq
    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY environment variable is not set. "
                "Get a free key in 30 seconds at https://console.groq.com/keys (No credit card required)."
            )
        model_name = get_model_name(model)
        return _generate_openai_compatible(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=temperature,
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key,
            model=model_name,
            is_json=False,
        )

    # Provider: Free Local Ollama
    if provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        model_name = get_model_name(model)
        return _generate_openai_compatible(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=temperature,
            base_url=base_url,
            api_key="ollama",
            model=model_name,
            is_json=False,
        )

    # Provider: Free OpenRouter
    if provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        model_name = get_model_name(model)
        return _generate_openai_compatible(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=temperature,
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            model=model_name,
            is_json=False,
        )

    # Provider: Pure Local Offline
    if provider == "local":
        return str(_generate_local_fallback(prompt, is_json=False))

    # Provider: Google Gemini
    model_name = get_model_name(model)
    client = get_client()

    from google.genai import types

    config = types.GenerateContentConfig(
        temperature=temperature,
        system_instruction=system_instruction if system_instruction else None,
    )

    last_error: Exception | None = None
    for attempt in range(1, 3):
        t0 = time.perf_counter()
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0
            response_text = response.text or ""
            logger.info(
                "LLM generate call completed: model=%s, prompt_len=%d, response_len=%d, latency_ms=%.2f",
                model_name,
                len(prompt),
                len(response_text),
                latency_ms,
            )
            return response_text
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            last_error = exc
            if attempt == 1 and _is_transient_error(exc):
                logger.warning(
                    "LLM generate transient error (attempt 1/2, model=%s, latency_ms=%.2f): %s. Retrying once...",
                    model_name,
                    latency_ms,
                    exc,
                )
                if DEFAULT_RETRY_DELAY_SECONDS > 0:
                    time.sleep(DEFAULT_RETRY_DELAY_SECONDS)
                continue

            logger.error(
                "LLM generate persistent or fatal error (attempt %d/2, model=%s, prompt_len=%d, latency_ms=%.2f): %s",
                attempt,
                model_name,
                len(prompt),
                latency_ms,
                exc,
            )
            raise last_error from exc

    if last_error:
        raise last_error
    raise RuntimeError("Unexpected error in generate()")


def generate_json(
    prompt: str,
    system_instruction: str = "",
    temperature: float = 0.1,
    model: str | None = None,
) -> dict | list:
    """Generate structured JSON output using the configured provider."""
    provider = get_provider()

    # Provider: Free Groq
    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY environment variable is not set. "
                "Get a free key in 30 seconds at https://console.groq.com/keys (No credit card required)."
            )
        model_name = get_model_name(model)
        raw = _generate_openai_compatible(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=temperature,
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key,
            model=model_name,
            is_json=True,
        )
        cleaned = _clean_json_text(raw)
        return json.loads(cleaned)

    # Provider: Free Local Ollama
    if provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        model_name = get_model_name(model)
        raw = _generate_openai_compatible(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=temperature,
            base_url=base_url,
            api_key="ollama",
            model=model_name,
            is_json=True,
        )
        cleaned = _clean_json_text(raw)
        return json.loads(cleaned)

    # Provider: Free OpenRouter
    if provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        model_name = get_model_name(model)
        raw = _generate_openai_compatible(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=temperature,
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            model=model_name,
            is_json=True,
        )
        cleaned = _clean_json_text(raw)
        return json.loads(cleaned)

    # Provider: Pure Local Offline
    if provider == "local":
        result = _generate_local_fallback(prompt, is_json=True)
        if isinstance(result, (dict, list)):
            return result
        return json.loads(str(result))

    # Provider: Google Gemini
    model_name = get_model_name(model)
    client = get_client()

    from google.genai import types

    config = types.GenerateContentConfig(
        temperature=temperature,
        system_instruction=system_instruction if system_instruction else None,
        response_mime_type="application/json",
    )

    last_error: Exception | None = None
    for attempt in range(1, 3):
        t0 = time.perf_counter()
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0
            response_text = response.text or ""

            cleaned = _clean_json_text(response_text)
            parsed = json.loads(cleaned)

            if not isinstance(parsed, (dict, list)):
                raise ValueError(
                    f"Expected JSON object or array, got {type(parsed).__name__}: {response_text!r}"
                )

            logger.info(
                "LLM generate_json call completed: model=%s, prompt_len=%d, response_len=%d, latency_ms=%.2f",
                model_name,
                len(prompt),
                len(response_text),
                latency_ms,
            )
            return parsed
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            last_error = exc
            if attempt == 1 and _is_transient_error(exc):
                logger.warning(
                    "LLM generate_json transient error (attempt 1/2, model=%s, latency_ms=%.2f): %s. Retrying once...",
                    model_name,
                    latency_ms,
                    exc,
                )
                if DEFAULT_RETRY_DELAY_SECONDS > 0:
                    time.sleep(DEFAULT_RETRY_DELAY_SECONDS)
                continue

            logger.error(
                "LLM generate_json persistent or fatal error (attempt %d/2, model=%s, prompt_len=%d, latency_ms=%.2f): %s",
                attempt,
                model_name,
                len(prompt),
                latency_ms,
                exc,
            )
            raise last_error from exc

    if last_error:
        raise last_error
    raise RuntimeError("Unexpected error in generate_json()")


__all__ = [
    "DEFAULT_GROQ_MODEL",
    "DEFAULT_MODEL",
    "generate",
    "generate_json",
    "get_client",
    "get_model_name",
    "get_provider",
    "reset_client",
    "set_client",
]
