from __future__ import annotations

import json
import time
from dataclasses import dataclass

import requests
from requests import HTTPError, RequestException

from .config import APIConfig


def _extract_content(content: str | list[dict] | None) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    chunks: list[str] = []
    for item in content:
        if isinstance(item, dict):
            text = item.get("text")
            if isinstance(text, str):
                chunks.append(text)
    return "".join(chunks)


@dataclass
class ChatResult:
    text: str
    reasoning_text: str | None
    raw: dict
    latency_s: float
    usage: dict | None
    finish_reason: str | None
    model: str


def _extract_reasoning(message: dict | None) -> str | None:
    if not isinstance(message, dict):
        return None
    for key in ("reasoning_content", "reasoning"):
        value = message.get(key)
        text = _extract_content(value)
        if text:
            return text
    return None


class QwenChatClient:
    def __init__(self, config: APIConfig):
        self.config = config
        if not self.config.resolved_api_key():
            raise ValueError(
                f"Missing API key. Set `api.api_key` in config or export `{self.config.api_key_env}`."
            )

    @property
    def endpoint(self) -> str:
        return self.config.api_base.rstrip("/") + "/chat/completions"

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.resolved_api_key()}",
        }

    def _payload(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict:
        return {
            "model": model or self.config.model,
            "messages": messages,
            "temperature": self.config.temperature if temperature is None else temperature,
            "max_tokens": self.config.max_tokens if max_tokens is None else max_tokens,
            "stream": self.config.stream,
        }

    def _should_retry_http(self, exc: HTTPError) -> bool:
        response = exc.response
        return response is not None and response.status_code >= 500

    def _post_with_retries(self, payload: dict, stream: bool = False, timeout_override: float | None = None) -> requests.Response:
        attempts = max(1, int(self.config.max_retries) + 1)
        last_exc: Exception | None = None
        effective_timeout = timeout_override if timeout_override is not None else self.config.timeout_s
        for attempt in range(1, attempts + 1):
            try:
                response = requests.post(
                    self.endpoint,
                    headers=self._headers(),
                    json=payload,
                    timeout=effective_timeout,
                    stream=stream,
                )
                response.raise_for_status()
                return response
            except HTTPError as exc:
                last_exc = exc
                if attempt >= attempts or not self._should_retry_http(exc):
                    raise
            except RequestException as exc:
                last_exc = exc
                if attempt >= attempts:
                    raise
            time.sleep(min(8.0, 1.5 * attempt))
        assert last_exc is not None
        raise last_exc

    def complete(
        self,
        messages: list[dict],
        model_override: str | None = None,
        timeout_override: float | None = None,
        temperature_override: float | None = None,
        max_tokens_override: int | None = None,
    ) -> ChatResult:
        model_name = model_override or self.config.model
        payload = self._payload(
            messages,
            model=model_name,
            temperature=temperature_override,
            max_tokens=max_tokens_override,
        )
        start = time.perf_counter()
        if self.config.stream:
            response = self._post_with_retries(payload, stream=True, timeout_override=timeout_override)
            text = self._collect_stream(response)
            latency = time.perf_counter() - start
            return ChatResult(
                text=text,
                reasoning_text=None,
                raw={"stream": True},
                latency_s=latency,
                usage=None,
                finish_reason="stop",
                model=model_name,
            )

        response = self._post_with_retries(payload, stream=False, timeout_override=timeout_override)
        body = response.json()
        latency = time.perf_counter() - start
        choice = body.get("choices", [{}])[0]
        message = choice.get("message", {})
        text = _extract_content(message.get("content"))
        return ChatResult(
            text=text,
            reasoning_text=_extract_reasoning(message),
            raw=body,
            latency_s=latency,
            usage=body.get("usage"),
            finish_reason=choice.get("finish_reason"),
            model=model_name,
        )

    def _collect_stream(self, response: requests.Response) -> str:
        chunks: list[str] = []
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            line = raw_line.strip()
            if line.startswith("data:"):
                line = line[5:].strip()
            if line == "[DONE]":
                break
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            delta = payload.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content")
            if isinstance(content, str):
                chunks.append(content)
            elif isinstance(content, list):
                chunks.append(_extract_content(content))
        return "".join(chunks)
