"""Pinned, structured-output OpenRouter transport used by both model roles."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .io import MVPError, load_yaml, sha256_object
from .schema import validate_against_schema


# XGrammar cannot enforce these JSON Schema keywords. Some hosted backends reject
# the entire request when they are present rather than merely ignoring them. They
# remain in the authoritative local schemas and are enforced after generation.
XGRAMMAR_UNSUPPORTED_KEYWORDS = frozenset({"uniqueItems"})


class OpenRouterResponseError(MVPError):
    """A successful HTTP response whose structured content could not be parsed."""

    def __init__(
        self,
        message: str,
        *,
        raw_response: dict[str, Any],
        provenance: dict[str, Any],
    ) -> None:
        super().__init__(message)
        self.raw_response = raw_response
        self.provenance = provenance


def load_model_profile(path: Path, *, expected_role: str) -> dict[str, Any]:
    profile = load_yaml(path)
    validate_against_schema(
        profile, "model_profile.schema.json", label=f"model profile {path}"
    )
    if profile["role"] != expected_role:
        raise MVPError(
            f"Profile {path} has role={profile['role']!r}; expected {expected_role!r}"
        )
    placeholders = [
        value
        for value in profile.values()
        if isinstance(value, str) and value.startswith("<") and value.endswith(">")
    ]
    if placeholders:
        raise MVPError(f"Profile {path} still contains placeholders: {placeholders}")
    return profile


def model_family(profile: dict[str, Any]) -> str:
    return str(profile["model"]).split("/", 1)[0].casefold()


def assert_independent_models(
    generator: dict[str, Any], validator: dict[str, Any]
) -> None:
    if model_family(generator) == model_family(validator):
        raise MVPError(
            "Generator and validator must use different model families; found "
            f"{generator['model']} and {validator['model']}"
        )


def public_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Return manifest-safe configuration. Profiles never contain the key itself."""
    return dict(profile)


def provider_response_schema(value: Any) -> Any:
    """Return an XGrammar-compatible copy without weakening local validation."""
    if isinstance(value, dict):
        return {
            key: provider_response_schema(item)
            for key, item in value.items()
            if key not in XGRAMMAR_UNSUPPORTED_KEYWORDS
        }
    if isinstance(value, list):
        return [provider_response_schema(item) for item in value]
    return value


def _parse_content(response: dict[str, Any], *, role: str) -> dict[str, Any]:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise MVPError(
            f"{role.capitalize()} response lacks choices[0].message.content"
        ) from exc
    if isinstance(content, dict):
        value = content
    elif isinstance(content, str):
        try:
            value = json.loads(content)
        except json.JSONDecodeError as exc:
            raise MVPError(f"{role.capitalize()} returned invalid JSON: {exc}") from exc
    else:
        raise MVPError(f"{role.capitalize()} response content is not JSON text")
    if not isinstance(value, dict):
        raise MVPError(f"{role.capitalize()} response must be a JSON object")
    return value


def build_request_body(
    profile: dict[str, Any],
    *,
    system: str,
    user: str,
    response_schema: dict[str, Any],
    schema_name: str,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": profile["model"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": int(profile["max_output_tokens"]),
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": provider_response_schema(response_schema),
            },
        },
        "provider": {
            "require_parameters": bool(profile["require_parameters"]),
            "allow_fallbacks": bool(profile["allow_fallbacks"]),
            "data_collection": profile.get("data_collection", "deny"),
        },
    }
    for parameter in ("temperature", "top_p"):
        if parameter in profile:
            body[parameter] = profile[parameter]
    if "reasoning_effort" in profile:
        body["reasoning"] = {"effort": str(profile["reasoning_effort"])}
    elif "reasoning_enabled" in profile:
        body["reasoning"] = {"enabled": bool(profile["reasoning_enabled"])}
    return body


def call_structured(
    profile: dict[str, Any],
    *,
    system: str,
    user: str,
    response_schema: dict[str, Any],
    schema_name: str,
) -> dict[str, Any]:
    """Call one pinned model route and return parsed output plus raw provenance."""
    env_name = str(profile["api_key_env"])
    api_key = os.environ.get(env_name)
    if not api_key:
        raise MVPError(f"Missing credential environment variable {env_name}")
    body = build_request_body(
        profile,
        system=system,
        user=user,
        response_schema=response_schema,
        schema_name=schema_name,
    )
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if profile.get("http_referer"):
        headers["HTTP-Referer"] = str(profile["http_referer"])
    if profile.get("app_title"):
        headers["X-Title"] = str(profile["app_title"])

    encoded = json.dumps(body).encode("utf-8")
    total_attempts = int(profile["transport_attempts"])
    last_error: Exception | None = None
    for transport_attempt in range(1, total_attempts + 1):
        request = urllib.request.Request(
            str(profile["base_url"]), data=encoded, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(
                request, timeout=float(profile["timeout_seconds"])
            ) as stream:
                raw = json.loads(stream.read().decode("utf-8"))
                response_headers = {
                    key.casefold(): value
                    for key, value in stream.headers.items()
                    if key.casefold()
                    in {
                        "x-request-id",
                        "x-generation-id",
                        "x-openrouter-request-id",
                        "openrouter-processing-time",
                    }
                }
            if not isinstance(raw, dict):
                raise MVPError("OpenRouter response is not a JSON object")
            provenance = {
                "profile_id": profile["profile_id"],
                "requested_model": profile["model"],
                "canonical_slug": profile["canonical_slug"],
                "returned_model": raw.get("model"),
                "response_id": raw.get("id"),
                "provider": raw.get("provider"),
                "usage": raw.get("usage"),
                "response_headers": response_headers,
                "request_sha256": sha256_object(body),
                "transport_attempt": transport_attempt,
            }
            try:
                parsed = _parse_content(raw, role=str(profile["role"]))
            except MVPError as exc:
                raise OpenRouterResponseError(
                    str(exc), raw_response=raw, provenance=provenance
                ) from exc
            return {
                "parsed": parsed,
                "raw_response": raw,
                "provenance": provenance,
            }
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            last_error = MVPError(f"OpenRouter HTTP {exc.code}: {details[:1000]}")
            retryable = exc.code == 429 or 500 <= exc.code <= 599
            if not retryable:
                raise last_error from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
        if transport_attempt < total_attempts:
            time.sleep(min(2 ** (transport_attempt - 1), 4))
    raise MVPError(
        f"OpenRouter transport failed after {total_attempts} attempts: {last_error}"
    ) from last_error
