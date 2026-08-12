"""Load and validate the topic and container catalogs."""

from __future__ import annotations

import csv
import difflib
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from .utils import (
    CONTRASTIVE_ROOT,
    ContrastiveError,
    hash_object,
    load_json,
    load_yaml,
    resolve_local_path,
    sha256_file,
)


SLOT_RE = re.compile(r"\{\{([^{}]+)\}\}")

REGISTER_JA = {
    "neutral_professional": "中立的で職業的",
    "casual": "くだけた",
    "terse": "簡潔",
}
ROLE_STRUCTURE_JA = {
    "user_only": "ユーザーのみ",
    "system_user": "システムとユーザー",
}
SLOT_JA = {
    "TASK": "課題",
    "CONTENT": "内容",
    "OPT_A": "選択肢ア",
    "OPT_B": "選択肢イ",
    "OPT_C": "選択肢ウ",
    "OPT_D": "選択肢エ",
    "TASK_CASUAL": "くだけた課題",
    "Q": "質問",
}
CONTRACT_TYPE_JA = {
    "multiple_choice": "選択式",
    "free_response": "自由回答",
}
ANSWER_KEY_JA = {"required": "必須", "empty": "空文字"}
OPTION_KEY_JA = {"A": "ア", "B": "イ", "C": "ウ", "D": "エ"}

REGISTER_CANONICAL = {value: key for key, value in REGISTER_JA.items()}
ROLE_STRUCTURE_CANONICAL = {
    value: key for key, value in ROLE_STRUCTURE_JA.items()
}
SLOT_CANONICAL = {value: key for key, value in SLOT_JA.items()}
CONTRACT_TYPE_CANONICAL = {
    value: key for key, value in CONTRACT_TYPE_JA.items()
}
ANSWER_KEY_CANONICAL = {value: key for key, value in ANSWER_KEY_JA.items()}


@dataclass(frozen=True)
class TopicRecord:
    topic_id: str
    topic_label: str
    topic_display_name: str
    domain: str
    domain_display_name: str
    generation_guidance: str | None


@dataclass(frozen=True)
class ContainerRecord:
    container_id: str
    data: dict[str, Any]

    def model_view(self, visible_fields: list[str]) -> dict[str, Any]:
        return {field: self.data[field] for field in visible_fields}


@dataclass(frozen=True)
class ContrastiveCatalog:
    topics: tuple[TopicRecord, ...]
    containers: tuple[ContainerRecord, ...]
    topic_data: dict[str, Any]
    container_data: dict[str, Any]
    purpose_frame_data: dict[str, Any]
    topic_path: Path
    container_path: Path
    purpose_frame_path: Path
    container_dir: Path
    container_paths: tuple[tuple[str, Path], ...]
    legacy_topic_path: Path

    def container_catalog_sha256(self) -> str:
        """One stable hash covering the index and every per-container file.

        Containers live in separate files, so a single ``sha256_file`` no longer
        describes the catalog. This digest changes if the index changes, if any
        container file changes, or if a container file is added or removed.
        """
        return hash_object({
            "index": sha256_file(self.container_path),
            "purpose_frames": sha256_file(self.purpose_frame_path),
            "containers": {
                container_id: sha256_file(path)
                for container_id, path in self.container_paths
            },
        })


def _validate(schema_name: str, value: dict[str, Any], label: Path) -> None:
    schema = load_json(CONTRASTIVE_ROOT / "schemas" / schema_name)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(value),
        key=lambda error: list(error.path),
    )
    if errors:
        details = "; ".join(
            f"{'.'.join(map(str, error.path)) or '<root>'}: {error.message}"
            for error in errors[:12]
        )
        raise ContrastiveError(f"Invalid contrastive catalog {label}: {details}")


def _schema_shape(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _schema_shape(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_schema_shape(child) for child in value]
    return type(value).__name__


def _canonical_topic_sha256(data: dict[str, Any]) -> str:
    """Hash parsed English topic data without self-referential i18n metadata."""
    return hash_object({key: value for key, value in data.items() if key != "i18n"})


def _canonical_container_sha256(data: dict[str, Any]) -> str:
    """Hash parsed English container data without self-referential i18n metadata."""
    return hash_object({key: value for key, value in data.items() if key != "i18n"})


CONTAINER_INVARIANT_FIELDS = (
    "container_id",
    "pair_id",
    "content_spec_id",
    "class",
    "design_notes",
    "nuisance_profile",
)

GENERATOR_VISIBLE_FIELDS = (
    "content_spec_id",
    "register",
    "role_structure",
    "slots",
    "response_contract",
    "model_guidance",
)

PURPOSE_FRAME_SLOT = "PURPOSE_FRAME"

# The frozen feasible allocation from docs/BLOCKER_1_FIX.md. Values are kept
# machine-readable here so catalog drift cannot silently alter the estimand.
FROZEN_PAIR_MANIFEST: dict[str, dict[str, Any]] = {
    "p01": {"content_spec_id": "content_p01", "explicitness": "implicit", "output_constraint": "unconstrained", "speaker_present": False, "prior_context": False, "constraint_kind": "none", "surface": "external"},
    "p11": {"content_spec_id": "content_p01", "explicitness": "explicit", "output_constraint": "constrained", "speaker_present": True, "prior_context": True, "constraint_kind": "bare_answer", "surface": "institutional"},
    "p02": {"content_spec_id": "content_p02", "explicitness": "implicit", "output_constraint": "unconstrained", "speaker_present": False, "prior_context": False, "constraint_kind": "none", "surface": "institutional"},
    "p12": {"content_spec_id": "content_p02", "explicitness": "explicit", "output_constraint": "unconstrained", "speaker_present": True, "prior_context": True, "constraint_kind": "none", "surface": "external"},
    "p03": {"content_spec_id": "content_p03", "explicitness": "implicit", "output_constraint": "unconstrained", "speaker_present": False, "prior_context": True, "constraint_kind": "none", "surface": "institutional"},
    "p13": {"content_spec_id": "content_p03", "explicitness": "explicit", "output_constraint": "constrained", "speaker_present": True, "prior_context": False, "constraint_kind": "bare_answer", "surface": "external"},
    "p04": {"content_spec_id": "content_p04", "explicitness": "implicit", "output_constraint": "unconstrained", "speaker_present": True, "prior_context": False, "constraint_kind": "none", "surface": "external"},
    "p14": {"content_spec_id": "content_p04", "explicitness": "explicit", "output_constraint": "constrained", "speaker_present": False, "prior_context": True, "constraint_kind": "bounded_explanation", "surface": "institutional"},
    "p05": {"content_spec_id": "content_p05", "explicitness": "implicit", "output_constraint": "unconstrained", "speaker_present": True, "prior_context": True, "constraint_kind": "none", "surface": "external"},
    "p15": {"content_spec_id": "content_p05", "explicitness": "explicit", "output_constraint": "constrained", "speaker_present": False, "prior_context": False, "constraint_kind": "bare_answer", "surface": "external"},
    "p06": {"content_spec_id": "content_p06", "explicitness": "implicit", "output_constraint": "constrained", "speaker_present": False, "prior_context": True, "constraint_kind": "bounded_explanation", "surface": "external"},
    "p16": {"content_spec_id": "content_p06", "explicitness": "explicit", "output_constraint": "unconstrained", "speaker_present": True, "prior_context": False, "constraint_kind": "none", "surface": "institutional"},
    "p07": {"content_spec_id": "content_p07", "explicitness": "implicit", "output_constraint": "constrained", "speaker_present": False, "prior_context": True, "constraint_kind": "bare_answer", "surface": "institutional"},
    "p17": {"content_spec_id": "content_p07", "explicitness": "explicit", "output_constraint": "unconstrained", "speaker_present": False, "prior_context": False, "constraint_kind": "none", "surface": "institutional"},
    "p08": {"content_spec_id": "content_p08", "explicitness": "implicit", "output_constraint": "constrained", "speaker_present": True, "prior_context": False, "constraint_kind": "bounded_explanation", "surface": "institutional"},
    "p18": {"content_spec_id": "content_p08", "explicitness": "explicit", "output_constraint": "unconstrained", "speaker_present": False, "prior_context": True, "constraint_kind": "none", "surface": "external"},
    "p09": {"content_spec_id": "content_p09", "explicitness": "implicit", "output_constraint": "constrained", "speaker_present": True, "prior_context": False, "constraint_kind": "bare_answer", "surface": "institutional"},
    "p19": {"content_spec_id": "content_p09", "explicitness": "explicit", "output_constraint": "unconstrained", "speaker_present": True, "prior_context": True, "constraint_kind": "none", "surface": "institutional"},
    "p10": {"content_spec_id": "content_p10", "explicitness": "implicit", "output_constraint": "constrained", "speaker_present": True, "prior_context": True, "constraint_kind": "bounded_explanation", "surface": "external"},
    "p20": {"content_spec_id": "content_p10", "explicitness": "explicit", "output_constraint": "constrained", "speaker_present": False, "prior_context": False, "constraint_kind": "bounded_explanation", "surface": "external"},
}


def _validate_topic_display_fields(data: dict[str, Any], label: Path) -> None:
    domains = {str(row["domain"]) for row in data["topics"]}
    display_domains = set(data["domain_display_names"])
    if display_domains != domains:
        raise ContrastiveError(
            f"Topic domain display names differ from the topic domains in {label}: "
            f"topics={sorted(domains)}, displays={sorted(display_domains)}"
        )


def _validate_topic_pair(topic_data: dict[str, Any], topic_path: Path) -> None:
    """Mechanically validate the English-canonical catalog and its Japanese view.

    This mirrors the blueprint pair contract: machine identity is invariant, the
    Japanese view has the same schema shape, and its metadata pins the parsed
    canonical English payload. Mechanical checks do not establish translation
    equivalence or native naturalness.
    """
    _validate_topic_display_fields(topic_data, topic_path)
    if topic_path.name.endswith(".ja.yaml"):
        return

    metadata = topic_data["i18n"]["ja_view"]
    ja_path = resolve_local_path(topic_path.parent / str(metadata["path"]))
    expected_name = f"{topic_path.stem}.ja{topic_path.suffix}"
    if ja_path.name != expected_name:
        raise ContrastiveError(
            f"Japanese topic view must be named {expected_name}, got {ja_path.name}"
        )
    if not ja_path.is_file():
        raise ContrastiveError(f"Missing Japanese topic view: {ja_path}")

    ja_data = load_yaml(ja_path)
    _validate("topics.schema.json", ja_data, ja_path)
    _validate_topic_display_fields(ja_data, ja_path)
    if _schema_shape(ja_data) != _schema_shape(topic_data):
        raise ContrastiveError(
            "English and Japanese topic catalogs do not have identical schema shape"
        )

    for field in (
        "schema_version",
        "catalog_version",
        "legacy_source",
        "legacy_design",
        "i18n",
    ):
        if ja_data[field] != topic_data[field]:
            raise ContrastiveError(
                f"Japanese topic view changed invariant field {field!r}"
            )
    if list(ja_data["domain_display_names"]) != list(
        topic_data["domain_display_names"]
    ):
        raise ContrastiveError(
            "Japanese topic view changed domain identifiers or their order"
        )

    invariant_fields = ("topic_id", "topic_label", "domain")
    for index, (en_topic, ja_topic) in enumerate(
        zip(topic_data["topics"], ja_data["topics"], strict=True)
    ):
        for field in invariant_fields:
            if ja_topic[field] != en_topic[field]:
                raise ContrastiveError(
                    f"Japanese topic view changed invariant topics[{index}].{field}"
                )

    if metadata["en_catalog_version"] != topic_data["catalog_version"]:
        raise ContrastiveError(
            "Japanese topic metadata does not target the current English "
            "catalog_version"
        )
    expected_sha = _canonical_topic_sha256(topic_data)
    if metadata["en_sha256"] != expected_sha:
        raise ContrastiveError(
            "Japanese topic metadata does not target the current canonical English "
            f"SHA-256: expected {expected_sha}"
        )
    if re.search(r"[０-９]", ja_path.read_text(encoding="utf-8")):
        raise ContrastiveError("Japanese topic view contains full-width digits")


def _verify_legacy_hash(data: dict[str, Any], label: str) -> Path:
    source = data["legacy_source"]
    path = resolve_local_path(source["path"])
    if not path.is_file():
        raise ContrastiveError(f"Missing retained legacy {label}: {path}")
    actual = sha256_file(path)
    if actual != source["sha256"]:
        raise ContrastiveError(
            f"Retained legacy {label} changed: expected {source['sha256']}, got {actual}"
        )
    return path


def _validate_topic_legacy(
    topic_data: dict[str, Any],
    topics: tuple[TopicRecord, ...],
    path: Path,
) -> None:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ContrastiveError(f"Retained topic grid is empty: {path}")

    legacy_topics: dict[str, tuple[str, str]] = {}
    cell_sets: dict[str, set[tuple[str, str]]] = {}
    counts: dict[str, set[int]] = {}
    for row in rows:
        topic_id = row["topic_id"]
        domain_field = "domain" if "domain" in row else "family"
        value = (row["topic_label"], row[domain_field])
        if topic_id in legacy_topics and legacy_topics[topic_id] != value:
            raise ContrastiveError(f"Legacy topic {topic_id} has conflicting metadata")
        legacy_topics[topic_id] = value
        cell_sets.setdefault(topic_id, set()).add(
            (row["class"], row["language"].lower())
        )
        counts.setdefault(topic_id, set()).add(int(row["n_items"]))

    yaml_topics = {
        topic.topic_id: (topic.topic_label, topic.domain) for topic in topics
    }
    if yaml_topics != legacy_topics:
        raise ContrastiveError(
            "topics.yaml does not preserve the topic identifiers, labels, and domains "
            "from the retained topic_grid.csv"
        )

    design = topic_data["legacy_design"]
    expected_cells = {
        (class_name, language)
        for class_name in design["classes"]
        for language in design["languages"]
    }
    for topic_id in yaml_topics:
        if cell_sets.get(topic_id) != expected_cells:
            raise ContrastiveError(
                f"Legacy topic grid cells differ for {topic_id}: "
                f"{sorted(cell_sets.get(topic_id, set()))}"
            )
        if counts.get(topic_id) != {int(design["items_per_cell"])}:
            raise ContrastiveError(
                f"Legacy topic grid item count differs for {topic_id}"
            )


def _ordered_template_slots(template: str) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for slot in SLOT_RE.findall(template):
        if slot != PURPOSE_FRAME_SLOT and slot not in seen:
            seen.add(slot)
            result.append(slot)
    return result



def _discover_containers(
    index_path: Path,
    container_dir: str,
    language: str = "en",
) -> tuple[Path, list[tuple[str, Path]], list[dict[str, Any]]]:
    """Read every per-container file in the declared directory.

    The filename stem must equal the container identifier, so a container is
    locatable from its identifier alone and a stray or renamed file is an error
    rather than a silent inclusion.
    """
    directory = resolve_local_path(index_path.parent / container_dir)
    if not directory.is_dir():
        raise ContrastiveError(f"Container directory does not exist: {directory}")
    if language == "ja":
        paths = sorted(directory.glob("*.ja.yaml"))
    else:
        paths = sorted(
            path for path in directory.glob("*.yaml")
            if not path.name.endswith(".ja.yaml")
        )
    if not paths:
        raise ContrastiveError(f"No container files found in {directory}")

    located: list[tuple[str, Path]] = []
    rows: list[dict[str, Any]] = []
    for path in paths:
        row = load_yaml(path)
        _validate("container.schema.json", row, path)
        container_id = str(row["container_id"])
        expected_name = (
            f"{container_id}.ja.yaml" if language == "ja"
            else f"{container_id}.yaml"
        )
        if path.name != expected_name:
            raise ContrastiveError(
                f"Container file {path.name} declares container_id {container_id!r}; "
                f"expected filename {expected_name}"
            )
        located.append((container_id, path))
        rows.append(row)
    return directory, located, rows


def _validate_container_pair(
    en_catalog_version: Any,
    row: dict[str, Any],
    path: Path,
) -> None:
    """Mechanically validate an English container against its Japanese sibling.

    Mirrors ``_validate_topic_pair``: machine identity is invariant, while
    model-facing structure and response contracts must equal their declared
    Japanese mappings. The JA metadata pins the parsed canonical English
    payload. This establishes mechanical equivalence only, not translation
    quality or native naturalness.
    """
    metadata = row["i18n"]["ja_view"]
    ja_path = resolve_local_path(path.parent / str(metadata["path"]))
    expected_name = f"{path.stem}.ja{path.suffix}"
    if ja_path.name != expected_name:
        raise ContrastiveError(
            f"Japanese container view must be named {expected_name}, got {ja_path.name}"
        )
    if not ja_path.is_file():
        raise ContrastiveError(f"Missing Japanese container view: {ja_path}")

    ja_row = load_yaml(ja_path)
    _validate("container.schema.json", ja_row, ja_path)
    if str(ja_row.get("container_id")) != row["container_id"]:
        raise ContrastiveError(
            f"Japanese container view {ja_path.name} declares a different container_id"
        )
    if _schema_shape(ja_row) != _schema_shape(row):
        raise ContrastiveError(
            f"Container {row['container_id']}: English and Japanese views do not have "
            "identical schema shape"
        )

    for field in CONTAINER_INVARIANT_FIELDS:
        if ja_row[field] != row[field]:
            raise ContrastiveError(
                f"Container {row['container_id']}: Japanese view changed invariant "
                f"field {field!r}"
            )

    expected_register = REGISTER_JA[row["register"]]
    expected_role_structure = ROLE_STRUCTURE_JA[row["role_structure"]]
    expected_slots = [SLOT_JA[slot] for slot in row["slots"]]
    expected_contract = {
        "type": CONTRACT_TYPE_JA[row["response_contract"]["type"]],
        "answer_key": ANSWER_KEY_JA[row["response_contract"]["answer_key"]],
    }
    if "allowed_keys" in row["response_contract"]:
        expected_contract["allowed_keys"] = [
            OPTION_KEY_JA[key] for key in row["response_contract"]["allowed_keys"]
        ]
    localized_values = {
        "register": expected_register,
        "role_structure": expected_role_structure,
        "slots": expected_slots,
        "response_contract": expected_contract,
    }
    for field, expected in localized_values.items():
        if ja_row[field] != expected:
            raise ContrastiveError(
                f"Container {row['container_id']}: Japanese field {field!r} "
                f"does not match the required localization: expected {expected!r}"
            )

    for field in ("attribution", "explicitness", "surface", "cue_locus", "family_id"):
        if ja_row["purpose_frame"][field] != row["purpose_frame"][field]:
            raise ContrastiveError(
                f"Container {row['container_id']}: Japanese purpose-frame "
                f"field {field!r} differs"
            )
    ja_slots = _ordered_template_slots(ja_row["body_template"])
    if ja_slots != ja_row["slots"]:
        raise ContrastiveError(
            f"Container {row['container_id']}: Japanese body-template slot order differs: "
            f"{ja_slots}"
        )
    if ja_row["i18n"] != row["i18n"]:
        raise ContrastiveError(
            f"Container {row['container_id']}: Japanese view i18n metadata does not "
            "match the English view"
        )

    if metadata["en_catalog_version"] != en_catalog_version:
        raise ContrastiveError(
            f"Container {row['container_id']}: Japanese metadata does not target the "
            "current English catalog_version"
        )
    expected_sha = _canonical_container_sha256(row)
    if metadata["en_sha256"] != expected_sha:
        raise ContrastiveError(
            f"Container {row['container_id']}: Japanese metadata does not target the "
            f"current canonical English SHA-256: expected {expected_sha}"
        )
    if re.search(r"[０-９]", ja_path.read_text(encoding="utf-8")):
        raise ContrastiveError(
            f"Container {row['container_id']}: Japanese view contains full-width digits"
        )



def _validate_containers(
    data: dict[str, Any],
    containers: tuple[ContainerRecord, ...],
    container_dir: Path,
    purpose_frame_data: dict[str, Any],
    language: str = "en",
) -> None:
    ids = [row.container_id for row in containers]
    if len(ids) != len(set(ids)):
        raise ContrastiveError("Duplicate container identifiers")

    visible = list(data["model_view"]["visible_fields"])
    hidden = set(data["model_view"]["hidden_fields"])
    overlap = hidden & set(visible)
    if overlap:
        raise ContrastiveError(
            f"Model-visible and hidden container fields overlap: {sorted(overlap)}"
        )

    for record in containers:
        row = record.data
        english_path = container_dir / f"{record.container_id}.yaml"
        english_row = row if language == "en" else load_yaml(english_path)
        _validate_container_pair(
            data["catalog_version"], english_row, english_path
        )
        slots = _ordered_template_slots(row["body_template"])
        if slots != row["slots"]:
            raise ContrastiveError(
                f"Container {record.container_id} slots differ from body-template "
                f"order: declared={row['slots']}, body={slots}"
            )
        if row["body_template"].count("{{PURPOSE_FRAME}}") != 1:
            raise ContrastiveError(
                f"Container {record.container_id} must contain {{PURPOSE_FRAME}} "
                "exactly once"
            )
        _validate_purpose_frame(record, language)
        _validate_plain_body(record)
        contract = row["response_contract"]
        canonical_slots = [SLOT_CANONICAL.get(slot, slot) for slot in row["slots"]]
        option_slots = {
            slot for slot in canonical_slots if slot.startswith("OPT_")
        }
        expected_options = {"OPT_A", "OPT_B", "OPT_C", "OPT_D"}
        contract_type = CONTRACT_TYPE_CANONICAL.get(
            contract["type"], contract["type"]
        )
        answer_key_mode = ANSWER_KEY_CANONICAL.get(
            contract["answer_key"], contract["answer_key"]
        )
        if contract_type == "multiple_choice":
            if answer_key_mode != "required" or option_slots != expected_options:
                raise ContrastiveError(
                    f"Container {record.container_id} has an invalid multiple-choice "
                    "contract: it needs four option slots and a required answer key"
                )
        elif answer_key_mode != "empty" or option_slots:
            raise ContrastiveError(
                f"Container {record.container_id} has an invalid free-response contract"
            )
        if contract_type != "multiple_choice" and "allowed_keys" in contract:
            raise ContrastiveError(
                f"Container {record.container_id} declares allowed_keys without a "
                "keyed answer"
            )
        model_view = record.model_view(visible)
        leaked = hidden & set(model_view)
        if leaked:
            raise ContrastiveError(
                f"Container {record.container_id} model view leaks {sorted(leaked)}"
            )

    _validate_pair_contract(containers)
    _validate_content_spec_reuse(containers)
    _validate_frozen_manifest(containers)
    _validate_purpose_frame_catalog(containers, purpose_frame_data, language)
    _validate_cue_locus(containers, purpose_frame_data, language)
    _validate_frame_lengths(containers, language)


def structural_cell(container: dict[str, Any]) -> tuple[str, str, str]:
    """The structural signature a container must not carry class information in."""
    canonical_slots = [SLOT_CANONICAL.get(slot, slot) for slot in container["slots"]]
    has_options = any(slot.startswith("OPT_") for slot in canonical_slots)
    return (
        REGISTER_CANONICAL.get(
            str(container["register"]), str(container["register"])
        ),
        ROLE_STRUCTURE_CANONICAL.get(
            str(container["role_structure"]), str(container["role_structure"])
        ),
        "options" if has_options else "no_options",
    )


def _validate_purpose_frame(record: ContainerRecord, language: str) -> None:
    frame = record.data["purpose_frame"]
    text = str(frame["text"])
    expected = "evaluation" if record.data["class"] == "eval" else "deployment"
    if frame["attribution"] != expected:
        raise ContrastiveError(
            f"Container {record.container_id} purpose attribution is inconsistent "
            f"with class {record.data['class']!r}"
        )
    if SLOT_RE.search(text):
        raise ContrastiveError(
            f"Container {record.container_id} purpose frame contains a slot"
        )
    if "\n" in text or re.search(r"<[^>]+>|^\s*(?:[-*]|\d+[.)])", text):
        raise ContrastiveError(
            f"Container {record.container_id} purpose frame contains markup or "
            "numbered/list scaffolding"
        )
    endings = "。！？" if language == "ja" else ".!?"
    if not text.endswith(tuple(endings)) or sum(text.count(mark) for mark in endings) != 1:
        raise ContrastiveError(
            f"Container {record.container_id} purpose frame must be exactly one sentence"
        )
    lowered = text.casefold()
    forbidden_directives = (
        "answer only",
        "respond with",
        "format your",
        "one sentence",
        "letter only",
        "答えだけ",
        "回答のみ",
        "一文以内",
        "形式で",
    )
    if any(phrase in lowered for phrase in forbidden_directives):
        raise ContrastiveError(
            f"Container {record.container_id} purpose frame contains an "
            "answer-format directive"
        )
    first_person = re.compile(
        r"\b(?:i|me|my|mine|we|us|our|ours)\b|私|わたし|我々|弊社",
        re.IGNORECASE,
    )
    if first_person.search(text):
        raise ContrastiveError(
            f"Container {record.container_id} uses first-person purpose framing"
        )


def _validate_plain_body(record: ContainerRecord) -> None:
    body = str(record.data["body_template"])
    profile = record.data["nuisance_profile"]
    constraint = profile["output_constraint"]
    kind = profile["constraint_kind"]
    if (constraint == "unconstrained") != (kind == "none"):
        raise ContrastiveError(
            f"Container {record.container_id} has invalid output-constraint nesting"
        )
    if profile["scaffolding"] != "plain":
        raise ContrastiveError(
            f"Container {record.container_id} matched scaffolding must be plain"
        )
    numbered = re.search(
        r"\{\{(?:N|番号)\}\}|\bItem\s+\d+\s+of\s+\d+\b|^\s*\d+[.)]\s",
        body,
        re.IGNORECASE | re.MULTILINE,
    )
    if numbered or re.search(r"</?[A-Za-z][^>]*>", body):
        raise ContrastiveError(
            f"Container {record.container_id} contains numbered or XML scaffolding"
        )


def _validate_pair_contract(containers: tuple[ContainerRecord, ...]) -> None:
    """Require purpose attribution to be the sole within-pair difference."""
    pairs: dict[str, list[ContainerRecord]] = {}
    for record in containers:
        pairs.setdefault(str(record.data["pair_id"]), []).append(record)

    expected_pair_ids = {f"p{number:02d}" for number in range(1, 21)}
    if set(pairs) != expected_pair_ids:
        raise ContrastiveError(
            "Container pair identifiers differ from p01-p20: "
            f"{sorted(pairs)}"
        )

    for pair_id, records in sorted(pairs.items()):
        if len(records) != 2:
            raise ContrastiveError(
                f"Container pair {pair_id} must contain exactly two containers"
            )
        if {record.data["class"] for record in records} != {"eval", "deploy"}:
            raise ContrastiveError(
                f"Container pair {pair_id} must contain one eval and one deploy view"
            )
        eval_record = next(row for row in records if row.data["class"] == "eval")
        deploy_record = next(
            row for row in records if row.data["class"] == "deploy"
        )
        for record, attribution in (
            (eval_record, "evaluation"),
            (deploy_record, "deployment"),
        ):
            if record.data["purpose_frame"]["attribution"] != attribution:
                raise ContrastiveError(
                    f"Container pair {pair_id} has attribution inconsistent "
                    "with class"
                )
            if record.data["body_template"].count("{{PURPOSE_FRAME}}") != 1:
                raise ContrastiveError(
                    f"Container pair {pair_id} must contain "
                    "{{PURPOSE_FRAME}} exactly once in each twin"
                )
        for field in GENERATOR_VISIBLE_FIELDS:
            left, right = eval_record, deploy_record
            if left.data[field] != right.data[field]:
                raise ContrastiveError(
                    f"Container pair {pair_id} differs in generated-content field "
                    f"{field!r}: {left.container_id} != {right.container_id}"
                )
        for field in ("body_template", "nuisance_profile"):
            if eval_record.data[field] != deploy_record.data[field]:
                raise ContrastiveError(
                    f"Container pair {pair_id} differs in pair-shared field {field!r}"
                )
        eval_frame = eval_record.data["purpose_frame"]
        deploy_frame = deploy_record.data["purpose_frame"]
        for field in ("explicitness", "surface", "family_id"):
            if eval_frame[field] != deploy_frame[field]:
                raise ContrastiveError(
                    f"Container pair {pair_id} differs in pair-shared purpose "
                    f"field {field!r}"
                )
        if not eval_frame["text"] or not deploy_frame["text"]:
            raise ContrastiveError(
                f"Container pair {pair_id} has an empty purpose frame"
            )
        if eval_frame["text"] == deploy_frame["text"]:
            raise ContrastiveError(
                f"Container pair {pair_id} has identical eval/deploy frame text"
            )
        rendered_eval = eval_record.data["body_template"].replace(
            "{{PURPOSE_FRAME}}", eval_frame["text"]
        )
        rendered_deploy = deploy_record.data["body_template"].replace(
            "{{PURPOSE_FRAME}}", deploy_frame["text"]
        )
        masked_eval = rendered_eval.replace(eval_frame["text"], "{{PURPOSE_FRAME}}", 1)
        masked_deploy = rendered_deploy.replace(
            deploy_frame["text"], "{{PURPOSE_FRAME}}", 1
        )
        if masked_eval != masked_deploy:
            raise ContrastiveError(
                f"Container pair {pair_id} rendered templates differ outside the "
                "purpose frame"
            )


def _validate_content_spec_reuse(
    containers: tuple[ContainerRecord, ...],
) -> None:
    by_spec: dict[str, list[ContainerRecord]] = {}
    for record in containers:
        by_spec.setdefault(str(record.data["content_spec_id"]), []).append(record)
    expected = {f"content_p{number:02d}" for number in range(1, 11)}
    if set(by_spec) != expected:
        raise ContrastiveError(
            "Content-spec identifiers differ from content_p01-p10"
        )
    for spec_id, records in sorted(by_spec.items()):
        pair_ids = {str(row.data["pair_id"]) for row in records}
        if len(records) != 4 or len(pair_ids) != 2:
            raise ContrastiveError(
                f"Content specification {spec_id} must feed exactly two pairs and "
                "four containers"
            )
        baseline = records[0]
        for record in records[1:]:
            for field in GENERATOR_VISIBLE_FIELDS[1:]:
                if baseline.data[field] != record.data[field]:
                    raise ContrastiveError(
                        f"Content specification {spec_id} has incompatible "
                        f"generator-visible field {field!r} in "
                        f"{record.container_id}"
                    )


def _pair_rows(
    containers: tuple[ContainerRecord, ...],
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for record in containers:
        pair_id = str(record.data["pair_id"])
        frame = record.data["purpose_frame"]
        profile = record.data["nuisance_profile"]
        candidate = {
            "content_spec_id": record.data["content_spec_id"],
            "explicitness": frame["explicitness"],
            "output_constraint": profile["output_constraint"],
            "speaker_present": profile["speaker_present"],
            "prior_context": profile["prior_context"],
            "constraint_kind": profile["constraint_kind"],
            "surface": frame["surface"],
        }
        if pair_id in rows and rows[pair_id] != candidate:
            raise ContrastiveError(
                f"Container pair {pair_id} does not have one shared manifest row"
            )
        rows[pair_id] = candidate
    return rows


def _validate_frozen_manifest(
    containers: tuple[ContainerRecord, ...],
) -> None:
    rows = _pair_rows(containers)
    if rows != FROZEN_PAIR_MANIFEST:
        mismatches = [
            pair_id
            for pair_id in sorted(set(rows) | set(FROZEN_PAIR_MANIFEST))
            if rows.get(pair_id) != FROZEN_PAIR_MANIFEST.get(pair_id)
        ]
        raise ContrastiveError(
            "Container catalog violates the frozen pair manifest: "
            + ", ".join(mismatches)
        )
    binary_fields = (
        "explicitness",
        "output_constraint",
        "speaker_present",
        "prior_context",
        "surface",
    )
    for field in binary_fields:
        counts = Counter(row[field] for row in rows.values())
        if sorted(counts.values()) != [10, 10]:
            raise ContrastiveError(
                f"Frozen manifest field {field!r} is not balanced 10/10"
            )
    for index, left in enumerate(binary_fields):
        for right in binary_fields[index + 1:]:
            counts = Counter((row[left], row[right]) for row in rows.values())
            if len(counts) != 4 or set(counts.values()) != {5}:
                raise ContrastiveError(
                    f"Frozen manifest pairwise cells {left}/{right} are not all five"
                )
    constrained = [
        row for row in rows.values()
        if row["output_constraint"] == "constrained"
    ]
    if Counter(row["constraint_kind"] for row in constrained) != {
        "bare_answer": 5,
        "bounded_explanation": 5,
    }:
        raise ContrastiveError(
            "Frozen manifest constrained kinds are not split 5/5"
        )
    if any(
        row["constraint_kind"] != "none"
        for row in rows.values()
        if row["output_constraint"] == "unconstrained"
    ):
        raise ContrastiveError("Frozen manifest violates output-kind nesting")


def _validate_purpose_frame_catalog(
    containers: tuple[ContainerRecord, ...],
    data: dict[str, Any],
    language: str,
) -> None:
    """Require independent, balanced frame families and exact catalog mirroring."""
    families = data["families"]
    family_ids = [str(row["family_id"]) for row in families]
    holdout_folds = [int(row["holdout_fold"]) for row in families]
    expected_families = {f"ff0{number}" for number in range(1, 6)}
    if set(family_ids) != expected_families or len(family_ids) != len(
        set(family_ids)
    ):
        raise ContrastiveError(
            "Purpose-frame families must be exactly ff01-ff05"
        )
    if set(holdout_folds) != set(range(1, 6)) or len(holdout_folds) != len(
        set(holdout_folds)
    ):
        raise ContrastiveError(
            "Purpose-frame holdout folds must be a one-to-one assignment of 1-5"
        )

    pair_rows = data["pairs"]
    by_pair = {str(row["pair_id"]): row for row in pair_rows}
    expected_pairs = {f"p{number:02d}" for number in range(1, 21)}
    if set(by_pair) != expected_pairs or len(by_pair) != len(pair_rows):
        raise ContrastiveError(
            "Purpose-frame catalog pairs must be unique and exactly p01-p20"
        )

    pair_containers: dict[str, dict[str, ContainerRecord]] = {}
    for record in containers:
        pair_containers.setdefault(str(record.data["pair_id"]), {})[
            str(record.data["class"])
        ] = record

    combined_texts: list[str] = []
    family_cells: Counter[tuple[str, str, str]] = Counter()
    for pair_id, row in sorted(by_pair.items()):
        family_id = str(row["family_id"])
        explicitness = str(row["explicitness"])
        surface = str(row["surface"])
        cue_locus = str(row["cue_locus"])
        family_cells[(family_id, explicitness, surface)] += 1
        language_pair = row[language]
        records = pair_containers[pair_id]
        for class_name in ("eval", "deploy"):
            frame = records[class_name].data["purpose_frame"]
            expected = {
                "family_id": family_id,
                "explicitness": explicitness,
                "surface": surface,
                "cue_locus": cue_locus,
                "text": language_pair[class_name],
            }
            actual = {field: frame[field] for field in expected}
            if actual != expected:
                raise ContrastiveError(
                    f"Container pair {pair_id} does not mirror the frozen "
                    f"{language} purpose-frame catalog for class {class_name}"
                )
            combined_texts.append(str(frame["text"]))

    expected_cells = {
        (family_id, explicitness, surface)
        for family_id in expected_families
        for explicitness in ("explicit", "implicit")
        for surface in ("external", "institutional")
    }
    if set(family_cells) != expected_cells or set(family_cells.values()) != {1}:
        raise ContrastiveError(
            "Each purpose-frame family must contain exactly one pair in every "
            "explicitness-by-surface cell"
        )
    families_by_content: dict[str, set[str]] = {}
    for pair_id, row in by_pair.items():
        content_spec_id = str(FROZEN_PAIR_MANIFEST[pair_id]["content_spec_id"])
        families_by_content.setdefault(content_spec_id, set()).add(
            str(row["family_id"])
        )
    if any(len(family_ids) != 2 for family_ids in families_by_content.values()):
        raise ContrastiveError(
            "The two wrapper pairs fed by each content specification must belong "
            "to different purpose-frame families"
        )
    if len(combined_texts) != len(set(combined_texts)):
        raise ContrastiveError(
            f"Every {language} eval/deploy purpose-frame text must be globally unique"
        )


IMPLICIT_CUE_LOCUS_BALANCE = {"both": 4, "verb_only": 3, "noun_only": 3}
IMPLICIT_CUE_SPAN_CAP = 2
CUE_LOCUS_MIN_FAMILIES = 3


def cue_spans(
    eval_text: str,
    deploy_text: str,
    language: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the contiguous spans in which the two class members differ.

    English is compared on whitespace tokens and Japanese on characters, so the
    result is the surface material a probe could key on without representing
    purpose at all. This is deliberately morphology-free: it measures how many
    places the pair differs and what text sits there, not part of speech.
    """
    if language == "en":
        left, right = eval_text.split(" "), deploy_text.split(" ")
        join = " ".join
    else:
        left, right = list(eval_text), list(deploy_text)
        join = "".join
    eval_spans, deploy_spans = [], []
    matcher = difflib.SequenceMatcher(None, left, right)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if i2 > i1:
            eval_spans.append(join(left[i1:i2]))
        if j2 > j1:
            deploy_spans.append(join(right[j1:j2]))
    return tuple(eval_spans), tuple(deploy_spans)


def _validate_cue_locus(
    containers: tuple[ContainerRecord, ...],
    data: dict[str, Any],
    language: str,
) -> None:
    """Require a declared, balanced, and lexically non-degenerate cue locus.

    ``cue_locus`` records which element of a frame pair carries the contrast.
    It is declared once per pair rather than per language, so the English and
    Japanese members cannot drift into different contrast structures — a
    divergence that would confound the primary EN-JA comparison by giving one
    language systematically more redundant cueing than the other.

    Three properties are enforced on the implicit stratum, which is where the
    claim that the direction survives without evaluation vocabulary lives:
    the declared locus must be balanced and spread across families and
    surfaces; single-locus pairs must actually differ in exactly one span;
    and no single span may recur often enough to be a class indicator on its
    own. The explicit stratum is exempt from the span rules because naming the
    evaluation or deployment lexeme is its entire purpose.
    """
    by_pair = {str(row["pair_id"]): row for row in data["pairs"]}
    implicit = {
        pair_id: row
        for pair_id, row in by_pair.items()
        if str(row["explicitness"]) == "implicit"
    }

    counts = Counter(str(row["cue_locus"]) for row in implicit.values())
    if counts != Counter(IMPLICIT_CUE_LOCUS_BALANCE):
        raise ContrastiveError(
            "Implicit purpose-frame cue loci are not balanced "
            f"{IMPLICIT_CUE_LOCUS_BALANCE}: got {dict(counts)}"
        )
    explicit_loci = {
        str(row["cue_locus"])
        for row in by_pair.values()
        if str(row["explicitness"]) == "explicit"
    }
    if explicit_loci != {"noun_only"}:
        raise ContrastiveError(
            "Explicit purpose frames must all be noun_only: "
            f"got {sorted(explicit_loci)}"
        )

    by_family: dict[str, list[str]] = {}
    by_surface: dict[str, Counter[str]] = {}
    for row in implicit.values():
        by_family.setdefault(str(row["family_id"]), []).append(str(row["cue_locus"]))
        by_surface.setdefault(str(row["surface"]), Counter())[
            str(row["cue_locus"])
        ] += 1
    for family_id, loci in sorted(by_family.items()):
        if len(set(loci)) != len(loci):
            raise ContrastiveError(
                f"Purpose-frame family {family_id} repeats an implicit cue locus "
                f"{loci}; a holdout fold must not be confounded with one locus"
            )
    families_per_locus: dict[str, set[str]] = {}
    for row in implicit.values():
        families_per_locus.setdefault(str(row["cue_locus"]), set()).add(
            str(row["family_id"])
        )
    for locus, families in sorted(families_per_locus.items()):
        if len(families) < CUE_LOCUS_MIN_FAMILIES:
            raise ContrastiveError(
                f"Implicit cue locus {locus!r} appears in only {len(families)} "
                f"purpose-frame families; at least {CUE_LOCUS_MIN_FAMILIES} required"
            )
    for surface, counter in sorted(by_surface.items()):
        missing = set(IMPLICIT_CUE_LOCUS_BALANCE) - set(counter)
        if missing:
            raise ContrastiveError(
                f"Implicit {surface} frames are missing cue loci {sorted(missing)}; "
                "locus would be confounded with surface"
            )

    span_counts: dict[str, Counter[str]] = {"eval": Counter(), "deploy": Counter()}
    for pair_id, row in sorted(implicit.items()):
        locus = str(row["cue_locus"])
        eval_spans, deploy_spans = cue_spans(
            str(row[language]["eval"]), str(row[language]["deploy"]), language
        )
        observed = max(len(eval_spans), len(deploy_spans))
        if locus == "both":
            if observed < 2:
                raise ContrastiveError(
                    f"Purpose-frame pair {pair_id} declares cue_locus 'both' but its "
                    f"{language} members differ in {observed} span(s)"
                )
        elif observed != 1:
            raise ContrastiveError(
                f"Purpose-frame pair {pair_id} declares cue_locus {locus!r} but its "
                f"{language} members differ in {observed} spans "
                f"{eval_spans} / {deploy_spans}; a single-locus pair must vary "
                "in exactly one place"
            )
        span_counts["eval"].update(eval_spans)
        span_counts["deploy"].update(deploy_spans)

    for class_name, counter in span_counts.items():
        for span, count in sorted(counter.items()):
            if count > IMPLICIT_CUE_SPAN_CAP:
                raise ContrastiveError(
                    f"{language} implicit {class_name} cue span {span!r} occurs in "
                    f"{count} of {len(implicit)} pairs (cap {IMPLICIT_CUE_SPAN_CAP}); "
                    "it would separate the classes on its own and would survive "
                    "leave-one-frame-family-out holdout"
                )


def raw_estimated_tokens(text: str, language: str) -> float:
    return len(text) / (1.8 if language == "ja" else 4.0)


def _validate_frame_lengths(
    containers: tuple[ContainerRecord, ...],
    language: str,
) -> None:
    by_pair: dict[str, dict[str, ContainerRecord]] = {}
    for record in containers:
        by_pair.setdefault(str(record.data["pair_id"]), {})[
            str(record.data["class"])
        ] = record
    lengths: dict[str, list[tuple[ContainerRecord, float]]] = {
        "eval": [],
        "deploy": [],
    }
    for pair_id, records in sorted(by_pair.items()):
        eval_length = raw_estimated_tokens(
            str(records["eval"].data["purpose_frame"]["text"]), language
        )
        deploy_length = raw_estimated_tokens(
            str(records["deploy"].data["purpose_frame"]["text"]), language
        )
        if abs(eval_length - deploy_length) > 1.0:
            raise ContrastiveError(
                f"Container pair {pair_id} violates the {language} frame-length gate"
            )
        lengths["eval"].append((records["eval"], eval_length))
        lengths["deploy"].append((records["deploy"], deploy_length))

    def mean(values: list[float]) -> float:
        return sum(values) / len(values)

    overall_gap = abs(
        mean([value for _, value in lengths["eval"]])
        - mean([value for _, value in lengths["deploy"]])
    )
    if overall_gap > 0.25:
        raise ContrastiveError(
            f"Catalog violates the {language} overall class-mean frame-length gate"
        )
    for field in ("explicitness", "surface"):
        values = ("explicit", "implicit") if field == "explicitness" else (
            "external",
            "institutional",
        )
        for value in values:
            eval_values = [
                length for record, length in lengths["eval"]
                if record.data["purpose_frame"][field] == value
            ]
            deploy_values = [
                length for record, length in lengths["deploy"]
                if record.data["purpose_frame"][field] == value
            ]
            if abs(mean(eval_values) - mean(deploy_values)) > 0.5:
                raise ContrastiveError(
                    f"Catalog violates the {language} {field}={value} "
                    "class-mean frame-length gate"
                )


def load_catalog(
    topics_path: str | Path = "topics.yaml",
    containers_path: str | Path = "containers.yaml",
) -> ContrastiveCatalog:
    topic_path = resolve_local_path(topics_path)
    container_path = resolve_local_path(containers_path)
    topic_data = load_yaml(topic_path)
    container_data = load_yaml(container_path)
    _validate("topics.schema.json", topic_data, topic_path)
    _validate("containers.schema.json", container_data, container_path)
    purpose_frame_path = resolve_local_path(
        container_path.parent / str(container_data["purpose_frame_catalog"])
    )
    purpose_frame_data = load_yaml(purpose_frame_path)
    _validate(
        "purpose_frames.schema.json",
        purpose_frame_data,
        purpose_frame_path,
    )
    if purpose_frame_data["catalog_version"] != container_data["catalog_version"]:
        raise ContrastiveError(
            "Purpose-frame and container catalog versions differ"
        )
    _validate_topic_pair(topic_data, topic_path)

    topics = tuple(
        TopicRecord(
            topic_id=str(row["topic_id"]),
            topic_label=str(row["topic_label"]),
            topic_display_name=str(row["topic_display_name"]),
            domain=str(row["domain"]),
            domain_display_name=str(
                topic_data["domain_display_names"][str(row["domain"])]
            ),
            generation_guidance=(
                str(row["generation_guidance"])
                if row.get("generation_guidance")
                else None
            ),
        )
        for row in topic_data["topics"]
    )
    topic_ids = [row.topic_id for row in topics]
    if len(topic_ids) != len(set(topic_ids)):
        raise ContrastiveError("Duplicate topic identifiers")

    language = "ja" if topic_path.name.endswith(".ja.yaml") else "en"
    container_dir, container_paths, container_rows = _discover_containers(
        container_path, str(container_data["container_dir"]), language
    )
    containers = tuple(
        ContainerRecord(str(row["container_id"]), row) for row in container_rows
    )
    legacy_topic_path = _verify_legacy_hash(topic_data, "topic grid")
    _validate_topic_legacy(topic_data, topics, legacy_topic_path)
    _validate_containers(
        container_data,
        containers,
        container_dir,
        purpose_frame_data,
        language,
    )
    return ContrastiveCatalog(
        topics=topics,
        containers=containers,
        topic_data=topic_data,
        container_data=container_data,
        purpose_frame_data=purpose_frame_data,
        topic_path=topic_path,
        container_path=container_path,
        purpose_frame_path=purpose_frame_path,
        container_dir=container_dir,
        container_paths=tuple(container_paths),
        legacy_topic_path=legacy_topic_path,
    )
