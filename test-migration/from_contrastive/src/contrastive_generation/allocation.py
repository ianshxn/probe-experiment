"""Permanent generator/verifier allocation for contrastive content identities."""

from __future__ import annotations

import re

from .utils import ContrastiveError


TOPIC_ID_RE = re.compile(r"^t([0-9]{3})$")
CONTENT_SPEC_ID_RE = re.compile(r"^content_p([0-9]{2})$")


def generator_for(topic_id: str, content_spec_id: str) -> str:
    """Choose the author without allowing language or sample index to confound it."""
    topic_match = TOPIC_ID_RE.fullmatch(topic_id)
    spec_match = CONTENT_SPEC_ID_RE.fullmatch(content_spec_id)
    if topic_match is None or spec_match is None:
        raise ContrastiveError(
            "Allocation requires topic ids t000-t999 and content ids content_p00-p99"
        )
    parity = int(topic_match.group(1)) + int(spec_match.group(1))
    return "generator_a" if parity % 2 == 0 else "generator_b"


def verifier_for(topic_id: str, content_spec_id: str) -> str:
    """Return the permanently opposite provider role."""
    return (
        "generator_b"
        if generator_for(topic_id, content_spec_id) == "generator_a"
        else "generator_a"
    )
