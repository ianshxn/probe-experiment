"""Self-contained provider adapters for contrastive generation and verification."""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from email.message import Message
from typing import Any

from .utils import ContrastiveError


@dataclass
class ProviderResponse:
    text: str
    request_id: str | None
    requested_model: str
    resolved_model: str | None
    route: str
    usage: dict[str, Any]
    request_payload: dict[str, Any]
    response_payload: dict[str, Any]


class ProviderError(ContrastiveError):
    """Provider failure with optional wire payloads for the audit trail."""

    def __init__(
        self,
        message: str,
        *,
        request_payload: dict[str, Any] | None = None,
        response_payload: dict[str, Any] | None = None,
        response_headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.request_payload = request_payload
        self.response_payload = response_payload
        self.response_headers = response_headers or {}


class ProviderTransportError(ProviderError):
    """Transport retries were exhausted without a completed model response."""


class MalformedProviderResponse(ProviderError):
    """The provider completed a request but did not return usable model text."""


def _wire_schema(schema: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in schema.items()
        if key not in {"$schema", "$id", "title"}
    }


def _header_dict(headers: Message | Any) -> dict[str, str]:
    if headers is None:
        return {}
    keep = {
        "date",
        "retry-after",
        "x-request-id",
        "x-generation-id",
        "x-gemini-service-tier",
        "x-goog-request-id",
    }
    return {
        key.lower(): value
        for key, value in headers.items()
        if key.lower() in keep
    }


def _request_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
    attempts: int,
) -> tuple[dict[str, Any], dict[str, str]]:
    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    last_error: BaseException | None = None
    for attempt in range(attempts):
        request = urllib.request.Request(
            url,
            data=encoded,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8")
                try:
                    parsed = json.loads(body)
                except json.JSONDecodeError as exc:
                    raise MalformedProviderResponse(
                        f"Provider returned invalid JSON from {url}: {exc}",
                        request_payload=payload,
                        response_headers=_header_dict(response.headers),
                    ) from exc
                if not isinstance(parsed, dict):
                    raise MalformedProviderResponse(
                        f"Provider returned a non-object JSON response from {url}",
                        request_payload=payload,
                        response_headers=_header_dict(response.headers),
                    )
                return parsed, _header_dict(response.headers)
        except MalformedProviderResponse:
            raise
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            headers_out = _header_dict(exc.headers)
            last_error = ProviderTransportError(
                f"HTTP {exc.code} from provider: {body[:1200]}",
                request_payload=payload,
                response_headers=headers_out,
            )
            retryable = exc.code in {408, 429, 500, 502, 503, 504}
            if not retryable or attempt + 1 >= attempts:
                raise last_error from exc
            retry_after = exc.headers.get("Retry-After")
            try:
                delay = min(30.0, max(0.0, float(retry_after)))
            except (TypeError, ValueError):
                delay = min(30.0, 2**attempt + random.random())
            time.sleep(delay)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt + 1 >= attempts:
                break
            time.sleep(min(30.0, 2**attempt + random.random()))
    raise ProviderTransportError(
        f"Provider request failed after {attempts} attempts: {last_error}",
        request_payload=payload,
    )


class BaseProvider:
    def __init__(self, name: str, profile: dict[str, Any]) -> None:
        self.name = name
        self.config = dict(profile)
        self.model = str(profile["model"])

    def check_credentials(self) -> None:
        raise NotImplementedError

    def generate(
        self,
        system: str,
        user: str,
        *,
        temperature: float,
        top_p: float,
        max_output_tokens: int,
        seed: int,
        response_schema: dict[str, Any],
    ) -> ProviderResponse:
        raise NotImplementedError


class OpenAICompatibleProvider(BaseProvider):
    """Adapter for OpenAI-compatible, OpenRouter, and NVIDIA NIM routes."""

    def _api_key(self) -> str:
        env_name = str(self.config.get("api_key_env", "OPENAI_API_KEY"))
        value = os.environ.get(env_name)
        if not value:
            raise ProviderError(f"Missing provider credential in ${env_name}")
        return value

    def check_credentials(self) -> None:
        self._api_key()

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._api_key()}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.config.get("provider") == "openrouter":
            headers["X-OpenRouter-Title"] = "lang-EA-pilot contrastive runner"
        return headers

    def _structured_output_payload(
        self,
        response_schema: dict[str, Any],
    ) -> dict[str, Any]:
        mode = str(self.config.get("structured_output_mode", "json_object"))
        schema = _wire_schema(response_schema)
        if mode == "guided_json":
            return {"guided_json": schema}
        if mode == "json_schema":
            return {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        # The name is provider protocol, not semantic prompt
                        # prose. Keep it language-neutral for Japanese runs.
                        "name": "0",
                        "strict": True,
                        "schema": schema,
                    },
                }
            }
        if mode == "json_object":
            return {"response_format": {"type": "json_object"}}
        if mode == "prompt_only":
            return {}
        raise ContrastiveError(
            "OpenAI-compatible structured_output_mode must be guided_json, "
            "json_schema, json_object, or prompt_only"
        )

    @staticmethod
    def _message_text(response: dict[str, Any]) -> str:
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise MalformedProviderResponse(
                "Malformed OpenAI-compatible response",
                response_payload=response,
            ) from exc
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                str(part["text"])
                for part in content
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            )
        raise MalformedProviderResponse(
            "OpenAI-compatible response contained unsupported message content",
            response_payload=response,
        )

    def generate(
        self,
        system: str,
        user: str,
        *,
        temperature: float,
        top_p: float,
        max_output_tokens: int,
        seed: int,
        response_schema: dict[str, Any],
    ) -> ProviderResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_output_tokens,
            "seed": seed,
            "stream": False,
            **self._structured_output_payload(response_schema),
        }
        # Reasoning models otherwise spend the whole completion budget thinking
        # and return content=None with finish_reason=length, which the parser
        # can only treat as a malformed response.
        reasoning_effort = self.config.get("reasoning_effort")
        if reasoning_effort:
            payload["reasoning"] = {
                "effort": str(reasoning_effort),
                "exclude": True,
            }
        response, response_headers = _request_json(
            str(self.config["base_url"]),
            payload,
            self._headers(),
            float(self.config.get("timeout_seconds", 180)),
            int(self.config.get("api_attempts", 1)),
        )
        try:
            text = self._message_text(response).strip()
        except MalformedProviderResponse as exc:
            exc.request_payload = payload
            exc.response_headers = response_headers
            raise
        if not text:
            raise MalformedProviderResponse(
                "OpenAI-compatible endpoint returned empty message content",
                request_payload=payload,
                response_payload=response,
                response_headers=response_headers,
            )
        usage = response.get("usage") or {}
        completion_details = usage.get("completion_tokens_details") or {}
        normalized_usage = {
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "thinking_tokens": completion_details.get("reasoning_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "cost_usd": usage.get("cost"),
            "raw": usage,
        }
        return ProviderResponse(
            text=text,
            request_id=(
                response.get("id")
                or response_headers.get("x-generation-id")
                or response_headers.get("x-request-id")
            ),
            requested_model=self.model,
            resolved_model=response.get("model"),
            route=str(self.config.get("route", "openai-compatible")),
            usage=normalized_usage,
            request_payload=payload,
            response_payload=response,
        )


class GoogleProvider(BaseProvider):
    _token_lock = threading.Lock()
    _cached_token: tuple[str, float] | None = None

    @staticmethod
    def _gcloud_executable() -> str:
        # On Windows the bare "gcloud" on PATH is a POSIX shell script that
        # CreateProcess cannot execute; shutil.which resolves PATHEXT and
        # returns gcloud.CMD instead.
        return shutil.which("gcloud") or "gcloud"

    def _gcloud(self, *args: str) -> str:
        try:
            result = subprocess.run(
                [self._gcloud_executable(), *args],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ProviderError(f"Could not invoke gcloud: {exc}") from exc
        if result.returncode:
            raise ProviderError(
                f"gcloud {' '.join(args)} failed: {result.stderr.strip()}"
            )
        value = result.stdout.strip()
        if not value:
            raise ProviderError(f"gcloud {' '.join(args)} returned no value")
        return value

    def _access_token(self) -> str:
        with self._token_lock:
            cached = self.__class__._cached_token
            if cached and cached[1] > time.time() + 60:
                return cached[0]
            token = self._gcloud("auth", "print-access-token")
            self.__class__._cached_token = (token, time.time() + 3000)
            return token

    def check_credentials(self) -> None:
        credential_env = str(
            self.config.get("credential_env", "GOOGLE_APPLICATION_CREDENTIALS")
        )
        credential_path = os.environ.get(credential_env)
        if not credential_path or not os.path.isfile(credential_path):
            raise ProviderError(
                f"Google application credential file is unavailable via ${credential_env}"
            )
        self._access_token()

    def _endpoint_and_headers(self) -> tuple[str, dict[str, str], str]:
        project = str(self.config["project"])
        location = str(self.config.get("location", "global"))
        base = str(
            self.config.get(
                "base_url",
                "https://aiplatform.googleapis.com",
            )
        )
        url = (
            f"{base.rstrip('/')}/v1/projects/{project}/locations/{location}/"
            f"publishers/google/models/{self.model}:generateContent"
        )
        return (
            url,
            {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._access_token()}",
                "x-goog-user-project": project,
            },
            str(self.config.get("route", f"google:vertex:{location}")),
        )

    def generate(
        self,
        system: str,
        user: str,
        *,
        temperature: float,
        top_p: float,
        max_output_tokens: int,
        seed: int,
        response_schema: dict[str, Any],
    ) -> ProviderResponse:
        generation_config: dict[str, Any] = {
            "temperature": temperature,
            "topP": top_p,
            "maxOutputTokens": max_output_tokens,
            "seed": seed,
            "responseMimeType": "application/json",
            "responseJsonSchema": _wire_schema(response_schema),
        }
        thinking_level = self.config.get("thinking_level")
        if thinking_level:
            generation_config["thinkingConfig"] = {
                "thinkingLevel": str(thinking_level).upper()
            }
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": generation_config,
        }
        url, headers, route = self._endpoint_and_headers()
        response, response_headers = _request_json(
            url,
            payload,
            headers,
            float(self.config.get("timeout_seconds", 180)),
            int(self.config.get("api_attempts", 1)),
        )
        try:
            parts = response["candidates"][0]["content"]["parts"]
            text = "".join(
                part.get("text", "")
                for part in parts
                if isinstance(part, dict)
            ).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise MalformedProviderResponse(
                "Malformed Google response",
                request_payload=payload,
                response_payload=response,
                response_headers=response_headers,
            ) from exc
        if not text:
            raise MalformedProviderResponse(
                "Google returned no text content",
                request_payload=payload,
                response_payload=response,
                response_headers=response_headers,
            )
        usage = response.get("usageMetadata") or {}
        normalized_usage = {
            "input_tokens": usage.get("promptTokenCount"),
            "output_tokens": usage.get("candidatesTokenCount"),
            "thinking_tokens": usage.get("thoughtsTokenCount"),
            "total_tokens": usage.get("totalTokenCount"),
            "raw": usage,
        }
        return ProviderResponse(
            text=text,
            request_id=(
                response.get("responseId")
                or response_headers.get("x-goog-request-id")
            ),
            requested_model=self.model,
            resolved_model=response.get("modelVersion") or self.model,
            route=route,
            usage=normalized_usage,
            request_payload=payload,
            response_payload=response,
        )


class MockProvider(BaseProvider):
    """Deterministic, contrastive-schema-aware offline provider."""

    def __init__(self, name: str, profile: dict[str, Any]) -> None:
        super().__init__(name, profile)
        self.calls: list[dict[str, Any]] = []
        self._calls_lock = threading.Lock()

    def check_credentials(self) -> None:
        return None

    @staticmethod
    def _generation_payload(schema: dict[str, Any]) -> dict[str, Any]:
        properties = schema.get("properties") or {}
        values_name = (
            "slot_values" if "slot_values" in properties else "各項目の内容"
        )
        key_name = "answer_key" if "answer_key" in properties else "解答記号"
        slot_schema = properties[values_name]
        slot_names = list(slot_schema.get("required") or [])
        japanese = values_name != "slot_values"
        values: dict[str, str] = {}
        for index, slot in enumerate(slot_names):
            if japanese:
                values[str(slot)] = (
                    "確認済みの案内に沿った内容です。"
                    "必要な情報を日本語で具体的に説明しています。"
                    if index == 0
                    else f"選択肢の内容{index + 1}です。"
                )
            else:
                values[str(slot)] = (
                    "Follow the confirmed notice and the supplied details."
                    if index == 0
                    else f"Use the documented choice number {index + 1}."
                )
        answer_schema = properties[key_name]
        answer = (
            str((answer_schema.get("enum") or [""])[0])
            if "enum" in answer_schema
            else ""
        )
        return {values_name: values, key_name: answer}

    def generate(
        self,
        system: str,
        user: str,
        *,
        temperature: float,
        top_p: float,
        max_output_tokens: int,
        seed: int,
        response_schema: dict[str, Any],
    ) -> ProviderResponse:
        schema_id = str(response_schema.get("$id", ""))
        if "verification_extraction" in schema_id:
            free_response = (
                "free-response" in user
                or "選択肢のない自由回答形式" in user
            )
            canonical = {
                "is_answerable_from_content": True,
                "independently_selected_option": None if free_response else 1,
                "leaked_answer_signal": False,
                "matches_authoring_requirements": True,
                "register_appropriate": True,
                "distractors_plausible": True,
                "distractor_notes": "No deterministic defect observed.",
                "register_notes": "Register is consistent.",
            }
            if ".ja." in schema_id:
                names = {
                    "is_answerable_from_content": "内容のみで回答可能",
                    "independently_selected_option": "独立に選んだ選択肢",
                    "leaked_answer_signal": "答えの漏えい",
                    "matches_authoring_requirements": "作成要件に一致",
                    "register_appropriate": "文体が適切",
                    "distractors_plausible": "誤答選択肢が妥当",
                    "distractor_notes": "誤答選択肢の所見",
                    "register_notes": "文体の所見",
                }
                value = {
                    names[key]: item for key, item in canonical.items()
                }
            else:
                value = canonical
        elif "verification_comparison" in schema_id:
            canonical = {
                "pass": True,
                "option_match": True,
                "answerable": True,
                "no_leak": True,
                "matches_authoring_requirements": True,
                "register_appropriate": True,
                "distractors_plausible": True,
                "issues": [],
            }
            if ".ja." in schema_id:
                names = {
                    "pass": "合格",
                    "option_match": "選択肢一致",
                    "answerable": "回答可能",
                    "no_leak": "漏えいなし",
                    "matches_authoring_requirements": "作成要件に一致",
                    "register_appropriate": "文体が適切",
                    "distractors_plausible": "誤答選択肢が妥当",
                    "issues": "問題点",
                }
                value = {
                    names[key]: item for key, item in canonical.items()
                }
            else:
                value = canonical
        else:
            configured = self.config.get("text")
            if isinstance(configured, str):
                text = configured
                value = None
            else:
                value = self._generation_payload(response_schema)
        if "text" not in locals():
            text = json.dumps(value, ensure_ascii=False)
        payload = {
            "system": system,
            "user": user,
            "seed": seed,
            "response_schema": response_schema,
        }
        with self._calls_lock:
            self.calls.append(dict(payload))
        response = {"text": text, "model": self.model}
        return ProviderResponse(
            text=text,
            request_id=f"mock-{seed}",
            requested_model=self.model,
            resolved_model=self.model,
            route="mock",
            usage={},
            request_payload=payload,
            response_payload=response,
        )


def make_provider(name: str, profile: dict[str, Any]) -> BaseProvider:
    provider = str(profile.get("provider"))
    if provider in {"openai_compatible", "openrouter", "nim"}:
        return OpenAICompatibleProvider(name, profile)
    if provider == "google":
        return GoogleProvider(name, profile)
    if provider == "mock":
        return MockProvider(name, profile)
    raise ContrastiveError(f"Unsupported provider type {provider!r} for {name}")
