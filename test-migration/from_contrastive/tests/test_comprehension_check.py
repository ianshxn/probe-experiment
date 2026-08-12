from __future__ import annotations

import copy
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


CONTRASTIVE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CONTRASTIVE_ROOT / "src"))

from contrastive_generation.calibration import (  # noqa: E402
    validate_heldout_comprehension_results,
)
from contrastive_generation.utils import ContrastiveError  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "comprehension_check_classify",
    CONTRASTIVE_ROOT / "comprehension_check" / "classify.py",
)
assert _SPEC and _SPEC.loader
classify = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(classify)

DATASET = CONTRASTIVE_ROOT / "data" / "processed" / "demo16"


class PromptTests(unittest.TestCase):
    def test_option_order_is_deterministic_and_counterbalanced(self) -> None:
        first = classify.option_order("hc_abc123")
        self.assertEqual(first, classify.option_order("hc_abc123"))
        self.assertEqual(set(first), set(classify.LABELS))
        ids = [f"hc_{index:04d}" for index in range(400)]
        eval_first = sum(
            1 for item in ids if classify.option_order(item)[0] == "eval"
        )
        self.assertGreater(eval_first, 150)
        self.assertLess(eval_first, 250)

    def test_instructions_are_monolingual_and_avoid_implicit_cues(self) -> None:
        messages, order = classify.render_messages(
            {"blind_item_id": "hc_abc123", "language": "ja", "text": "本文"}
        )
        system = messages[0]["content"]
        self.assertIn("アシスタント", system)
        self.assertNotIn("assistant's response", system)
        # The implicit contrast verbs must not appear: naming them would hand
        # the reader the answer key for the implicit stratum.
        for cue in ("精査", "照合", "参照", "処理", "受け取", "利用", "活用"):
            self.assertNotIn(cue, system, f"implicit cue {cue} leaked into instructions")
        self.assertEqual(
            system.index(f"- {order[0]}:"),
            min(system.index(f"- {label}:") for label in classify.LABELS),
        )

    def test_english_instructions_for_english_items(self) -> None:
        messages, _ = classify.render_messages(
            {"blind_item_id": "hc_abc123", "language": "en", "text": "body"}
        )
        self.assertIn("You classify prompts", messages[0]["content"])
        self.assertIn("body", messages[1]["content"])


class ParseTests(unittest.TestCase):
    def test_accepts_structured_fenced_and_bare_labels(self) -> None:
        self.assertEqual(classify.parse_label('{"label": "eval"}'), "eval")
        self.assertEqual(
            classify.parse_label('```json\n{"label":"deploy"}\n```'), "deploy"
        )
        self.assertEqual(classify.parse_label("  Deploy \n"), "deploy")

    def test_rejects_empty_and_out_of_vocabulary_content(self) -> None:
        for bad in (None, "", "unsure", '{"label": "evaluation"}'):
            with self.assertRaises(ValueError):
                classify.parse_label(bad)


class ConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config_path = CONTRASTIVE_ROOT / "comprehension_check" / "config.json"
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))

    def _write(self, payload: dict) -> Path:
        directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, directory, True)
        path = directory / "config.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_shipped_config_loads_with_distinct_lineages(self) -> None:
        loaded = classify.load_config(self.config_path, None)
        lineages = [model["gate"]["lineage"] for model in loaded["models"]]
        self.assertEqual(len(lineages), len(set(lineages)))

    def test_attestations_are_explicit_booleans(self) -> None:
        """Whether they are signed is the owner's call; silent defaults are not."""
        for model in self.config["models"]:
            for field in (
                "disjoint_from_generators_attestation",
                "disjoint_from_subject_models_attestation",
            ):
                self.assertIsInstance(model["gate"][field], bool)

    def test_repeated_lineage_is_rejected(self) -> None:
        payload = copy.deepcopy(self.config)
        payload["models"][1]["gate"]["lineage"] = payload["models"][0]["gate"]["lineage"]
        with self.assertRaises(classify.ClassifyError):
            classify.load_config(self._write(payload), None)

    def test_gate_block_shape_is_enforced(self) -> None:
        payload = copy.deepcopy(self.config)
        payload["models"][0]["gate"]["extra"] = "no"
        with self.assertRaises(classify.ClassifyError):
            classify.load_config(self._write(payload), None)


@unittest.skipUnless(DATASET.is_dir(), "demo16 dataset is not present locally")
class PacketTests(unittest.TestCase):
    def test_loads_without_the_answer_key_present(self) -> None:
        staged = Path(tempfile.mkdtemp()) / "demo16"
        self.addCleanup(shutil.rmtree, staged.parent, True)
        staged.mkdir(parents=True)
        for name in ("dataset_manifest.json", classify.ITEMS_ARTIFACT):
            shutil.copy(DATASET / name, staged / name)
        items = classify.load_items(staged)
        self.assertEqual(len(items), 400)
        self.assertFalse((staged / "heldout_comprehension_key.jsonl").exists())

    def test_tampered_packet_is_refused_before_spending(self) -> None:
        staged = Path(tempfile.mkdtemp()) / "demo16"
        self.addCleanup(shutil.rmtree, staged.parent, True)
        staged.mkdir(parents=True)
        for name in ("dataset_manifest.json", classify.ITEMS_ARTIFACT):
            shutil.copy(DATASET / name, staged / name)
        with (staged / classify.ITEMS_ARTIFACT).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"blind_item_id": "x", "language": "en", "text": "y"}) + "\n")
        with self.assertRaises(classify.ClassifyError):
            classify.load_items(staged)

    def test_limit_is_language_balanced(self) -> None:
        items = classify.select_items(classify.load_items(DATASET), 10)
        languages = [item["language"] for item in items]
        self.assertEqual(languages.count("en"), 5)
        self.assertEqual(languages.count("ja"), 5)


@unittest.skipUnless(DATASET.is_dir(), "demo16 dataset is not present locally")
class ArtifactTests(unittest.TestCase):
    """Emitted artifacts must be accepted by the real gate, unchanged."""

    def setUp(self) -> None:
        # The gate refuses paths outside the contrastive root, so run artifacts
        # have to be produced inside it. data/ is gitignored.
        self.out = Path(
            tempfile.mkdtemp(
                prefix="test_comprehension_",
                dir=CONTRASTIVE_ROOT / "data" / "calibration",
            )
        )
        self.addCleanup(shutil.rmtree, self.out, True)
        self.items = classify.load_items(DATASET)
        self.config = classify.load_config(
            CONTRASTIVE_ROOT / "comprehension_check" / "config.json", None
        )
        for model in self.config["models"]:
            model["gate"]["disjoint_from_generators_attestation"] = True
            model["gate"]["disjoint_from_subject_models_attestation"] = True
        self.key = {
            str(row["blind_item_id"]): row
            for row in (
                json.loads(line)
                for line in (DATASET / "heldout_comprehension_key.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            )
        }

    def _archive(self, reader, served: dict[str, str] | None = None) -> None:
        """Write synthetic raw records for every model x item."""
        for model in self.config["models"]:
            key = str(model["model_key"])
            for index, item in enumerate(self.items):
                blind_id = str(item["blind_item_id"])
                record = {
                    "blind_item_id": blind_id,
                    "language": item["language"],
                    "model_key": key,
                    "prediction": reader(self.key[blind_id], index),
                    "served_model": (served or {}).get(key, model["openrouter_slug"]),
                    "served_provider": "fixture",
                    "usage": {"prompt_tokens": 300, "completion_tokens": 5},
                    "attempts": 1,
                    "completed_at": "2026-08-02T00:00:00+00:00",
                    "settings_sha256": classify.settings_fingerprint(
                        classify.resolved_request(self.config, model)
                    ),
                }
                path = classify.record_path(self.out, key, blind_id)
                classify.write_atomic(path, json.dumps(record, ensure_ascii=False))

    @staticmethod
    def _competent(key_row: dict, index: int) -> str:
        """Perfect on explicit frames, ~85% on implicit, errors class-balanced."""
        if key_row["explicitness"] == "explicit":
            return key_row["class"]
        if index % 7 == 0:
            return "deploy" if key_row["class"] == "eval" else "eval"
        return key_row["class"]

    def test_emitted_artifacts_pass_the_real_gate(self) -> None:
        self._archive(self._competent)
        emitted = classify.emit_artifacts(self.out, self.config, self.items, True)
        self.assertEqual(emitted["predictions_file"], "predictions.jsonl")

        rows = [
            json.loads(line)
            for line in (self.out / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(rows), 800)
        for row in rows:
            self.assertEqual(set(row), {"blind_item_id", "model_id", "prediction"})

        report = validate_heldout_comprehension_results(
            DATASET,
            self.out / "predictions.jsonl",
            self.out / "heldout_models.json",
        )
        self.assertEqual(report["status"], "passed")
        self.assertEqual(len(report["models"]), 2)
        for model_report in report["models"].values():
            for language in ("en", "ja"):
                self.assertTrue(all(model_report["languages"][language]["gates"].values()))

    def test_attestations_are_carried_through_verbatim(self) -> None:
        config = copy.deepcopy(self.config)
        config["models"][0]["gate"]["disjoint_from_generators_attestation"] = False
        self._archive(self._competent)
        classify.emit_artifacts(self.out, config, self.items, True)
        models = json.loads((self.out / "heldout_models.json").read_text(encoding="utf-8"))
        self.assertFalse(models["models"][0]["disjoint_from_generators_attestation"])
        self.assertTrue(models["models"][1]["disjoint_from_generators_attestation"])

    def test_model_revision_resolves_from_the_served_model(self) -> None:
        self._archive(self._competent)
        classify.emit_artifacts(self.out, self.config, self.items, True)
        models = json.loads((self.out / "heldout_models.json").read_text(encoding="utf-8"))
        for model in models["models"]:
            self.assertTrue(model["model_revision"].startswith("openrouter/"))
            self.assertIn("@2026-08-02", model["model_revision"])
            self.assertEqual(set(model), classify.GATE_FIELDS)

    def test_routing_change_mid_run_is_refused(self) -> None:
        self._archive(self._competent)
        key = str(self.config["models"][0]["model_key"])
        path = classify.record_path(self.out, key, str(self.items[0]["blind_item_id"]))
        record = json.loads(path.read_text(encoding="utf-8"))
        record["served_model"] = "anthropic/claude-sonnet-4.6"
        classify.write_atomic(path, json.dumps(record))
        with self.assertRaises(classify.ClassifyError):
            classify.emit_artifacts(self.out, self.config, self.items, True)

    def test_mixed_request_settings_are_refused(self) -> None:
        self._archive(self._competent)
        key = str(self.config["models"][0]["model_key"])
        path = classify.record_path(self.out, key, str(self.items[0]["blind_item_id"]))
        record = json.loads(path.read_text(encoding="utf-8"))
        record["settings_sha256"] = "0" * 64
        classify.write_atomic(path, json.dumps(record))
        with self.assertRaises(classify.ClassifyError):
            classify.emit_artifacts(self.out, self.config, self.items, True)

    def test_transport_settings_do_not_invalidate_answers(self) -> None:
        model = self.config["models"][0]
        baseline = classify.settings_fingerprint(
            classify.resolved_request(self.config, model)
        )
        config = copy.deepcopy(self.config)
        config["request_defaults"]["max_attempts"] = 9
        config["request_defaults"]["timeout_seconds"] = 30
        self.assertEqual(
            baseline,
            classify.settings_fingerprint(
                classify.resolved_request(config, config["models"][0])
            ),
        )
        config["models"][0]["request"]["temperature"] = 0.7
        self.assertNotEqual(
            baseline,
            classify.settings_fingerprint(
                classify.resolved_request(config, config["models"][0])
            ),
        )

    def test_a_flawless_reader_fails_the_ordering_gate(self) -> None:
        """Explicit must beat implicit, so 100% everywhere is a failure, not a pass."""
        self._archive(lambda key_row, index: key_row["class"])
        classify.emit_artifacts(self.out, self.config, self.items, True)
        report = validate_heldout_comprehension_results(
            DATASET,
            self.out / "predictions.jsonl",
            self.out / "heldout_models.json",
        )
        self.assertEqual(report["status"], "failed")

    def test_partial_run_cannot_be_mistaken_for_evidence(self) -> None:
        subset = classify.select_items(self.items, 20)
        self.items = subset
        self._archive(self._competent)
        emitted = classify.emit_artifacts(self.out, self.config, subset, False)
        self.assertEqual(emitted["predictions_file"], "predictions.partial.jsonl")
        self.assertFalse((self.out / "predictions.jsonl").exists())
        with self.assertRaises(ContrastiveError):
            validate_heldout_comprehension_results(
                DATASET,
                self.out / "predictions.partial.jsonl",
                self.out / "heldout_models.json",
            )


if __name__ == "__main__":
    unittest.main()
