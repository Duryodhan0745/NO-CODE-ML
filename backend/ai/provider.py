from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from urllib import error, request

class ProviderConfigurationError(RuntimeError):
    pass


class ProviderRequestError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProviderConfig:
    api_key: str
    model: str
    base_url: str = "https://api.openai.com/v1"

    @classmethod
    def from_env(cls) -> "ProviderConfig":
        key = os.getenv("AI_API_KEY", "").strip()
        if not key:
            raise ProviderConfigurationError("AI_API_KEY is not configured; AI requests are unavailable")
        return cls(
            api_key=key,
            model=os.getenv("AI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini",
            base_url=os.getenv("AI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
        )


class OpenAICompatibleProvider:
    """Small provider adapter with no frontend secret exposure or fake fallback responses."""

    def __init__(self, config: ProviderConfig | None = None) -> None:
        self.config = config or ProviderConfig.from_env()

    def complete(self, messages: list[dict], *, json_mode: bool = False, retries: int = 2) -> str:
        body = {"model": self.config.model, "messages": messages}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        req = request.Request(
            f"{self.config.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                with request.urlopen(req, timeout=60) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                break
            except error.HTTPError as exc:
                if exc.code < 500 or attempt == retries:
                    raise ProviderRequestError(f"AI provider returned HTTP {exc.code}") from exc
                last_error = exc
            except (error.URLError, TimeoutError) as exc:
                last_error = exc
                if attempt == retries:
                    raise ProviderRequestError("AI provider request failed") from exc
            time.sleep(0.25 * (2 ** attempt))
        else:
            raise ProviderRequestError("AI provider request failed") from last_error
        try:
            return payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderRequestError("AI provider returned an invalid response") from exc

    def complete_json(self, messages: list[dict], *, retries: int = 2) -> dict:
        raw = self.complete(messages, json_mode=True, retries=retries)
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProviderRequestError("AI provider returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise ProviderRequestError("AI provider JSON response must be an object")
        return value

    def complete_json_stream(self, messages: list[dict], on_text_chunk) -> dict:
        body = {"model": self.config.model, "messages": messages, "stream": True}
        body["response_format"] = {"type": "json_object"}
        req = request.Request(
            f"{self.config.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"******", "Content-Type": "application/json"},
            method="POST",
        )
        raw_response = ""
        key_match = None
        value_cursor = None
        safe_string = ""
        emitted = ""
        try:
            with request.urlopen(req, timeout=60) as response:
                for line in response:
                    if not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        break
                    try:
                        event = json.loads(data)
                        chunk = event["choices"][0]["delta"].get("content") or ""
                    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
                        raise ProviderRequestError("AI provider returned an invalid streaming event") from exc
                    if not isinstance(chunk, str):
                        raise ProviderRequestError("AI provider returned a non-text streaming chunk")
                    raw_response += chunk
                    if value_cursor is None:
                        key_match = re.search(r'"response"\s*:\s*"', raw_response)
                        if key_match:
                            value_cursor = key_match.end()
                    if value_cursor is None or value_cursor == -1:
                        continue
                    while value_cursor < len(raw_response):
                        char = raw_response[value_cursor]
                        if char == '"':
                            value_cursor = -1
                            break
                        if char == "\\":
                            if value_cursor + 1 >= len(raw_response):
                                break
                            escaped = raw_response[value_cursor + 1]
                            unit_length = 2
                            if escaped == "u":
                                if value_cursor + 6 > len(raw_response):
                                    break
                                if not re.fullmatch(r"[0-9a-fA-F]{4}", raw_response[value_cursor + 2:value_cursor + 6]):
                                    raise ProviderRequestError("AI provider returned an invalid response escape")
                                unit_length = 6
                            elif escaped not in '"\\/bfnrt':
                                raise ProviderRequestError("AI provider returned an invalid response escape")
                            unit = raw_response[value_cursor:value_cursor + unit_length]
                        else:
                            if ord(char) < 0x20:
                                raise ProviderRequestError("AI provider returned an invalid response string")
                            unit = char
                        safe_string += unit
                        value_cursor += len(unit)
                        decoded = json.loads(f'"{safe_string}"')
                        if len(decoded) > len(emitted):
                            on_text_chunk(decoded[len(emitted):])
                            emitted = decoded
            payload = json.loads(raw_response)
        except error.HTTPError as exc:
            raise ProviderRequestError(f"AI provider returned HTTP {exc.code}") from exc
        except (error.URLError, TimeoutError) as exc:
            raise ProviderRequestError("AI provider streaming request failed") from exc
        except json.JSONDecodeError as exc:
            raise ProviderRequestError("AI provider returned invalid streamed JSON") from exc
        if not isinstance(payload, dict):
            raise ProviderRequestError("AI provider JSON response must be an object")
        return payload
