"""Pre-generation prompt loading and deterministic wrapper pre-rendering."""

from __future__ import annotations

from pathlib import Path

from .catalog import ContainerRecord
from .utils import ContrastiveError


PROMPT_MARKER = "\nUSER\n"


def load_prompt_template(path: Path) -> tuple[str, str]:
    """Load the two model-visible sections of a v2 authoring prompt."""
    raw = path.read_text(encoding="utf-8")
    if not raw.startswith("SYSTEM\n") or PROMPT_MARKER not in raw:
        raise ContrastiveError(
            f"Prompt template must contain SYSTEM and USER sections: {path}"
        )
    system, user = raw[len("SYSTEM\n"):].split(PROMPT_MARKER, 1)
    return system.strip(), user.strip()


def prerender_template(
    container: ContainerRecord,
    seed: int,
) -> tuple[str, int | None]:
    """Insert the renderer-owned purpose frame without filling content slots."""
    del seed  # Kept in the stable renderer API; matched bodies have no randomness.
    body = str(container.data["body_template"])
    if body.count("{{PURPOSE_FRAME}}") != 1:
        raise ContrastiveError(
            f"Container {container.container_id} must contain "
            "{{PURPOSE_FRAME}} exactly once"
        )
    return body.replace(
        "{{PURPOSE_FRAME}}",
        str(container.data["purpose_frame"]["text"]),
    ), None
