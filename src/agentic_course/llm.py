"""Minimal, dependency-free chat client for OpenAI-compatible endpoints.

One class covers every hosted or local provider that speaks the
``/chat/completions`` protocol (Ollama, Groq, OpenRouter, OpenAI). A
``MockLLM`` wraps a plain Python function so the whole project runs and is
testable offline, without any model or API key.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Callable, Protocol

Message = dict[str, str]

PROVIDERS: dict[str, dict[str, str]] = {
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "key_env": "",
        "default_model": "llama3.1",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "default_model": "llama-3.3-70b-versatile",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "default_model": "meta-llama/llama-3.3-70b-instruct",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "key_env": "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini",
    },
}


class LLMError(RuntimeError):
    """Raised when a provider cannot be reached or returns an unusable reply."""


class LLM(Protocol):
    name: str

    def complete(self, messages: list[Message], *, temperature: float = 0.2) -> str:
        """Return the assistant's reply text for a list of chat messages."""
        ...


class OpenAICompatLLM:
    """Chat client for any endpoint that implements ``/chat/completions``."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        name: str = "openai-compatible",
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.name = name
        self.timeout = timeout

    def complete(self, messages: list[Message], *, temperature: float = 0.2) -> str:
        body = json.dumps(
            {"model": self.model, "messages": messages, "temperature": temperature}
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise LLMError(f"{self.name} returned HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise LLMError(f"could not reach {self.name} at {self.base_url}: {exc}") from exc
        try:
            return payload["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected reply shape from {self.name}: {payload!r:.300}") from exc


class MockLLM:
    """Offline stand-in: delegates to a deterministic Python function."""

    name = "mock"

    def __init__(self, handler: Callable[[list[Message]], str]) -> None:
        self._handler = handler
        self.calls: list[list[Message]] = []

    def complete(self, messages: list[Message], *, temperature: float = 0.2) -> str:
        self.calls.append(list(messages))
        return self._handler(messages)


def get_llm(
    provider: str | None = None,
    model: str | None = None,
    mock_handler: Callable[[list[Message]], str] | None = None,
) -> LLM:
    """Build a client from arguments or the ``LLM_PROVIDER`` / ``LLM_MODEL`` env vars.

    ``provider`` is one of ``mock``, ``ollama``, ``groq``, ``openrouter``,
    ``openai``. The default is ``mock`` so a fresh clone runs with no setup.
    """
    provider = (provider or os.environ.get("LLM_PROVIDER") or "mock").lower()
    if provider == "mock":
        if mock_handler is None:
            raise LLMError("the mock provider needs a handler function")
        return MockLLM(mock_handler)
    if provider not in PROVIDERS:
        raise LLMError(f"unknown provider '{provider}'; choose from mock, {', '.join(PROVIDERS)}")
    preset = PROVIDERS[provider]
    api_key = os.environ.get(preset["key_env"], "") if preset["key_env"] else ""
    if preset["key_env"] and not api_key:
        raise LLMError(f"set {preset['key_env']} to use the {provider} provider")
    return OpenAICompatLLM(
        base_url=os.environ.get("LLM_BASE_URL") or preset["base_url"],
        model=model or os.environ.get("LLM_MODEL") or preset["default_model"],
        api_key=api_key,
        name=provider,
    )


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply.

    Models often wrap JSON in prose or code fences; this finds the first
    balanced ``{...}`` block and parses it.
    """
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start : index + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(parsed, dict):
                        return parsed
                    break
        start = text.find("{", start + 1)
    raise ValueError("no JSON object found in model reply")
