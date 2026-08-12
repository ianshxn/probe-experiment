"""Compile renderer-owned container twins into template-blind content specs."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from .catalog import (
    ANSWER_KEY_CANONICAL,
    CONTRACT_TYPE_CANONICAL,
    OPTION_KEY_JA,
    REGISTER_CANONICAL,
    ROLE_STRUCTURE_CANONICAL,
    SLOT_CANONICAL,
    ContainerRecord,
    ContrastiveCatalog,
)
from .utils import ContrastiveError, hash_object


@dataclass(frozen=True)
class ContentSpecRecord:
    """One localized content contract shared by two eval/deploy wrapper pairs."""

    content_spec_id: str
    pair_ids: tuple[str, str]
    language: str
    register: str
    role_structure: str
    slots: tuple[str, ...]
    slot_descriptions: tuple[tuple[str, str], ...]
    response_contract: dict[str, Any]
    model_guidance: str
    containers: tuple[ContainerRecord, ...]

    def model_projection(self) -> ModelContentSpec:
        """Create a capability-limited DTO with no renderer/private fields."""
        return ModelContentSpec(
            language=self.language,
            register=self.register,
            role_structure=self.role_structure,
            slots=self.slots,
            slot_descriptions=self.slot_descriptions,
            response_contract=copy.deepcopy(self.response_contract),
            model_guidance=self.model_guidance,
        )

    def model_payload(self) -> dict[str, Any]:
        """Return the complete and explicit model-visible spec allowlist."""
        return self.model_projection().model_payload()

    def sha256(self) -> str:
        """Hash only generation-affecting content-spec material."""
        return self.model_projection().sha256()


@dataclass(frozen=True)
class ModelContentSpec:
    """Capability-limited content DTO accepted by request construction."""

    language: str
    register: str
    role_structure: str
    slots: tuple[str, ...]
    slot_descriptions: tuple[tuple[str, str], ...]
    response_contract: dict[str, Any]
    model_guidance: str

    def model_payload(self) -> dict[str, Any]:
        return {
            "register": self.register,
            "role_structure": self.role_structure,
            "slots": [
                {"name": name, "description": description}
                for name, description in self.slot_descriptions
            ],
            "response_contract": copy.deepcopy(self.response_contract),
            "model_guidance": self.model_guidance,
        }

    def sha256(self) -> str:
        return hash_object(self.model_payload())


@dataclass(frozen=True)
class ModelTopic:
    """Capability-limited localized topic DTO accepted by request construction."""

    topic_display_name: str
    domain_display_name: str


def model_topic(topic: Any) -> ModelTopic:
    return ModelTopic(
        topic_display_name=str(topic.topic_display_name),
        domain_display_name=str(topic.domain_display_name),
    )


def _language_for(catalog: ContrastiveCatalog) -> str:
    return "ja" if catalog.topic_path.name.endswith(".ja.yaml") else "en"


def _canonical_contract(contract: dict[str, Any]) -> dict[str, Any]:
    value: dict[str, Any] = {
        "type": CONTRACT_TYPE_CANONICAL.get(
            str(contract["type"]), str(contract["type"])
        ),
        "answer_key": ANSWER_KEY_CANONICAL.get(
            str(contract["answer_key"]), str(contract["answer_key"])
        ),
    }
    if "allowed_keys" in contract:
        reverse_options = {value: key for key, value in OPTION_KEY_JA.items()}
        value["allowed_keys"] = [
            reverse_options.get(str(key), str(key))
            for key in contract["allowed_keys"]
        ]
    return value


def canonical_content_shape(spec: ContentSpecRecord) -> dict[str, Any]:
    """Return language-neutral structural fields for bilingual comparison."""
    return {
        "content_spec_id": spec.content_spec_id,
        "pair_ids": list(spec.pair_ids),
        "register": REGISTER_CANONICAL.get(spec.register, spec.register),
        "role_structure": ROLE_STRUCTURE_CANONICAL.get(
            spec.role_structure, spec.role_structure
        ),
        "slots": [SLOT_CANONICAL.get(slot, slot) for slot in spec.slots],
        "response_contract": _canonical_contract(spec.response_contract),
        "container_ids": [row.container_id for row in spec.containers],
        "classes": sorted(str(row.data["class"]) for row in spec.containers),
    }


def compile_content_specs(
    catalog: ContrastiveCatalog,
    scaffolding: dict[str, Any],
) -> tuple[ContentSpecRecord, ...]:
    """Derive one generation spec for each four-container content identity."""
    language = _language_for(catalog)
    try:
        localized = scaffolding["languages"][language]
        descriptions = localized["slot_descriptions"]
    except (KeyError, TypeError) as exc:
        raise ContrastiveError(
            f"Content scaffolding has no complete {language!r} slot-description view"
        ) from exc
    if not isinstance(descriptions, dict):
        raise ContrastiveError(
            f"Content scaffolding {language!r} slot_descriptions must be a mapping"
        )

    by_spec: dict[str, list[ContainerRecord]] = {}
    for container in catalog.containers:
        by_spec.setdefault(
            str(container.data["content_spec_id"]), []
        ).append(container)

    records: list[ContentSpecRecord] = []
    seen_content_ids: set[str] = set()
    for content_spec_id, containers in sorted(by_spec.items()):
        if len(containers) != 4:
            raise ContrastiveError(
                f"Content-spec compiler requires four containers for "
                f"{content_spec_id}"
            )
        ordered = tuple(sorted(containers, key=lambda row: row.container_id))
        left = ordered[0]
        pair_ids = tuple(sorted({
            str(container.data["pair_id"]) for container in ordered
        }))
        if len(pair_ids) != 2:
            raise ContrastiveError(
                f"Content specification {content_spec_id} must be used by two pairs"
            )
        if content_spec_id in seen_content_ids:
            raise ContrastiveError(
                f"Duplicate compiled content specification {content_spec_id}"
            )
        seen_content_ids.add(content_spec_id)

        slots = tuple(str(value) for value in left.data["slots"])
        missing = [slot for slot in slots if slot not in descriptions]
        if missing:
            raise ContrastiveError(
                f"Content specification {content_spec_id} has slots without "
                f"{language} descriptions: {missing}"
            )
        records.append(
            ContentSpecRecord(
                content_spec_id=content_spec_id,
                pair_ids=pair_ids,  # type: ignore[arg-type]
                language=language,
                register=str(left.data["register"]),
                role_structure=str(left.data["role_structure"]),
                slots=slots,
                slot_descriptions=tuple(
                    (slot, str(descriptions[slot])) for slot in slots
                ),
                response_contract=copy.deepcopy(left.data["response_contract"]),
                model_guidance=str(left.data["model_guidance"]),
                containers=ordered,
            )
        )

    expected_ids = {f"content_p{number:02d}" for number in range(1, 11)}
    if seen_content_ids != expected_ids:
        raise ContrastiveError(
            "Compiled content specification identifiers differ from content_p01-p10: "
            f"{sorted(seen_content_ids)}"
        )
    return tuple(sorted(records, key=lambda row: row.content_spec_id))


def validate_bilingual_content_specs(
    english: tuple[ContentSpecRecord, ...],
    japanese: tuple[ContentSpecRecord, ...],
) -> None:
    """Mechanically verify structural equivalence of localized spec views."""
    en_by_id = {row.content_spec_id: row for row in english}
    ja_by_id = {row.content_spec_id: row for row in japanese}
    if set(en_by_id) != set(ja_by_id):
        raise ContrastiveError(
            "English and Japanese compiled content-spec identifiers differ"
        )
    for content_spec_id in sorted(en_by_id):
        en_shape = canonical_content_shape(en_by_id[content_spec_id])
        ja_shape = canonical_content_shape(ja_by_id[content_spec_id])
        if en_shape != ja_shape:
            raise ContrastiveError(
                f"English and Japanese content specification {content_spec_id} "
                f"have different canonical structures: en={en_shape}, ja={ja_shape}"
            )


def response_schema_for_spec(spec: ModelContentSpec) -> dict[str, Any]:
    """Build the exact localized structured-output schema for one spec."""
    japanese = spec.language == "ja"
    values_name = "各項目の内容" if japanese else "slot_values"
    key_name = "解答記号" if japanese else "answer_key"
    answer_key: dict[str, Any] = {"type": "string"}
    answer_mode = ANSWER_KEY_CANONICAL.get(
        str(spec.response_contract["answer_key"]),
        str(spec.response_contract["answer_key"]),
    )
    if answer_mode == "required":
        answer_key["enum"] = list(spec.response_contract["allowed_keys"])
    else:
        answer_key["const"] = ""
    return {
        "type": "object",
        "properties": {
            values_name: {
                "type": "object",
                "properties": {
                    slot: {"type": "string"} for slot in spec.slots
                },
                "required": list(spec.slots),
                "additionalProperties": False,
            },
            key_name: answer_key,
        },
        "required": [values_name, key_name],
        "additionalProperties": False,
    }


def catalog_semantic_review_payload(
    english_catalog: ContrastiveCatalog,
    japanese_catalog: ContrastiveCatalog,
    english_specs: tuple[ContentSpecRecord, ...],
    japanese_specs: tuple[ContentSpecRecord, ...],
) -> dict[str, Any]:
    """Material whose bilingual meaning requires human review."""
    en_topics = {row.topic_id: row for row in english_catalog.topics}
    ja_topics = {row.topic_id: row for row in japanese_catalog.topics}
    en_specs = {row.content_spec_id: row for row in english_specs}
    ja_specs = {row.content_spec_id: row for row in japanese_specs}

    topics = [
        {
            "topic_id": topic_id,
            "topic_label": en_topics[topic_id].topic_label,
            "domain": en_topics[topic_id].domain,
            "en": {
                "topic_display_name": en_topics[topic_id].topic_display_name,
                "domain_display_name": en_topics[topic_id].domain_display_name,
            },
            "ja": {
                "topic_display_name": ja_topics[topic_id].topic_display_name,
                "domain_display_name": ja_topics[topic_id].domain_display_name,
            },
        }
        for topic_id in sorted(en_topics)
    ]
    content_specs = [
        {
            "content_spec_id": content_spec_id,
            "pair_ids": list(en_specs[content_spec_id].pair_ids),
            "en": en_specs[content_spec_id].model_payload(),
            "ja": ja_specs[content_spec_id].model_payload(),
        }
        for content_spec_id in sorted(en_specs)
    ]
    en_containers = {
        container.container_id: container
        for spec in english_specs
        for container in spec.containers
    }
    ja_containers = {
        container.container_id: container
        for spec in japanese_specs
        for container in spec.containers
    }
    containers = [
        {
            "container_id": container_id,
            "pair_id": en_containers[container_id].data["pair_id"],
            "class": en_containers[container_id].data["class"],
            "en": {
                "purpose_frame": en_containers[container_id].data["purpose_frame"],
                "body_template": en_containers[container_id].data["body_template"],
                "nuisance_profile": en_containers[container_id].data[
                    "nuisance_profile"
                ],
            },
            "ja": {
                "purpose_frame": ja_containers[container_id].data["purpose_frame"],
                "body_template": ja_containers[container_id].data["body_template"],
                "nuisance_profile": ja_containers[container_id].data[
                    "nuisance_profile"
                ],
            },
        }
        for container_id in sorted(en_containers)
    ]
    return {
        "schema_version": 1,
        "topics": topics,
        "content_specs": content_specs,
        "containers": containers,
    }


def catalog_semantic_review_sha256(
    english_catalog: ContrastiveCatalog,
    japanese_catalog: ContrastiveCatalog,
    english_specs: tuple[ContentSpecRecord, ...],
    japanese_specs: tuple[ContentSpecRecord, ...],
) -> str:
    return hash_object(
        catalog_semantic_review_payload(
            english_catalog,
            japanese_catalog,
            english_specs,
            japanese_specs,
        )
    )
