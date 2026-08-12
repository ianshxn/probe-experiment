from __future__ import annotations

import copy
import json
import shutil
import sys
import tempfile
import unittest
import uuid
from pathlib import Path

import numpy as np
import yaml


CONTRASTIVE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CONTRASTIVE_ROOT / "src"))
sys.path.insert(0, str(CONTRASTIVE_ROOT / "probe_check"))
sys.path.insert(
    0,
    str(
        CONTRASTIVE_ROOT
        / "probe_check"
        / "confirmations"
        / "01_format_matched"
    ),
)
sys.path.insert(
    0,
    str(CONTRASTIVE_ROOT / "probe_check" / "confirmations" / "02_cue_invariant"),
)
sys.path.insert(
    0,
    str(CONTRASTIVE_ROOT / "probe_check" / "confirmations" / "03_cue_constrained"),
)
sys.path.insert(
    0,
    str(CONTRASTIVE_ROOT / "probe_check" / "development" / "cue_constraints"),
)

from contrastive_generation.planning import load_provider_profile
from contrastive_generation.calibration import (
    validate_heldout_comprehension_results,
)
from contrastive_generation.post_generation import (
    _evaluation_contract,
    assemble_post_generation,
    validate_format_matched_controls,
    validate_lexical_controls,
)
from contrastive_generation.pre_generation import (
    apply_pre_generation_overrides,
    load_pre_generation_config,
    with_pre_generation_provider,
    write_pre_generation_plan,
)
from contrastive_generation.runner import GenerationRunner
from contrastive_generation.utils import (
    ContrastiveError,
    hash_object,
    sha256_file,
)
from extract_confirmatory_8b import (
    CUE_LOCUS_BY_PAIR,
    DEFAULT_CONTROLS,
    DEFAULT_ITEMS,
    FIT_METADATA_FIELDS,
    FROZEN_MODELS,
    validate_inputs as validate_confirmatory_extraction_inputs,
)
from build_cue_invariant_controls import validate_bank as validate_cue_invariant_bank
from build_cue_balanced_development_augmentation import validate_and_render
from build_cue_constrained_confirmatory_controls import (
    validate_bank as validate_cue_constrained_bank,
)
from cue_constrained import cue_differences, project_null, rowspace_basis


class PostGenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        suffix = uuid.uuid4().hex
        self.run_id_a = f"test_post_a_{suffix}"
        self.run_id_b = f"test_post_b_{suffix}"
        self.dataset_id = f"test_post_dataset_{suffix}"
        self.run_dir_a = CONTRASTIVE_ROOT / "runs" / self.run_id_a
        self.run_dir_b = CONTRASTIVE_ROOT / "runs" / self.run_id_b
        self.output_a = CONTRASTIVE_ROOT / "data" / "pilot" / self.run_id_a
        self.output_b = CONTRASTIVE_ROOT / "data" / "pilot" / self.run_id_b
        self.dataset_dir = (
            CONTRASTIVE_ROOT / "data" / "processed" / self.dataset_id
        )
        profile_a = load_provider_profile("tests/fixtures/provider.yaml")
        profile_b = copy.deepcopy(profile_a)
        profile_b["profile_id"] = "offline_post_test_b"
        profile_b["model"] = "deterministic-post-test-model-b"
        profile_b["model_revision"] = "post-test-revision-2"
        base = apply_pre_generation_overrides(
            with_pre_generation_provider(
                load_pre_generation_config(),
                profile_a,
            ),
            topics=["t001"],
            content_specs=["content_p01"],
        )
        second = copy.deepcopy(base)
        second["provider_profile"] = profile_b
        write_pre_generation_plan(base, self.run_id_a)
        write_pre_generation_plan(second, self.run_id_b)
        config = {
            "schema_version": 1,
            "stage": "contrastive_generation_runner",
            "run_dir_a": str(self.run_dir_a),
            "run_dir_b": str(self.run_dir_b),
            "production": False,
            "max_workers": 2,
            "lease_timeout_seconds": 30,
            "budget": {
                "max_usd": 100,
                "cost_basis": "generation_retry_scenario",
                "allow_zero_pricing": False,
            },
            "verification": {
                "temperature": 0,
                "top_p": 1,
                "extraction_max_output_tokens": 512,
                "comparison_max_output_tokens": 512,
                "schema_retries": 1,
                "japanese_surface": {
                    "min_japanese_chars": 20,
                    "max_latin_share": 0.8,
                },
                "escalation": {
                    "enabled": True,
                    "extraction_max_output_tokens": 1024,
                },
            },
        }
        GenerationRunner(config).run()

    def tearDown(self) -> None:
        for root in (
            self.run_dir_a,
            self.run_dir_b,
            self.output_a,
            self.output_b,
            self.dataset_dir,
        ):
            if root.is_dir():
                shutil.rmtree(root)

    @staticmethod
    def _jsonl(path: Path) -> list[dict]:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def test_mock_outputs_become_grouped_probe_ready_dataset(self) -> None:
        result = assemble_post_generation(
            run_dir_a=self.run_dir_a,
            run_dir_b=self.run_dir_b,
            output_dir=self.dataset_dir,
        )
        self.assertEqual(
            result["status"], "structurally_complete_probe_protocol_blocked"
        )
        items = self._jsonl(self.dataset_dir / "probe_items.jsonl")
        pairs = self._jsonl(self.dataset_dir / "matched_pairs.jsonl")
        calibration = self._jsonl(
            self.dataset_dir / "language_calibration_groups.jsonl"
        )
        self.assertEqual(len(items), 8)
        self.assertEqual(len(pairs), 4)
        self.assertEqual(len(calibration), 4)
        format_matched = self._jsonl(
            self.dataset_dir / "format_matched_controls.jsonl"
        )
        self.assertEqual(len(format_matched), 256)
        self.assertEqual(
            {
                (
                    row["language"],
                    row["surface"],
                    row["cue_family"],
                    row["intended_purpose"],
                    row["lexical_cue"],
                )
                for row in format_matched
            },
            {
                (language, surface, family, purpose, cue)
                for language in ("en", "ja")
                for surface in ("external", "institutional")
                for family in ("explicit_lexeme", "implicit_cue")
                for purpose in ("evaluation", "deployment")
                for cue in ("evaluation", "deployment")
            },
        )
        comprehension_items = self._jsonl(
            self.dataset_dir / "heldout_comprehension_items.jsonl"
        )
        comprehension_key = self._jsonl(
            self.dataset_dir / "heldout_comprehension_key.jsonl"
        )
        self.assertEqual(len(comprehension_items), 8)
        self.assertEqual(len(comprehension_key), 8)
        self.assertTrue(
            all(
                set(row) == {"blind_item_id", "language", "text"}
                for row in comprehension_items
            )
        )
        self.assertEqual({row["class"] for row in items}, {"eval", "deploy"})
        self.assertEqual({row["language"] for row in items}, {"en", "ja"})
        self.assertEqual(
            {row["frame_family_id"] for row in items},
            {"ff01", "ff02"},
        )
        for pair in pairs:
            members = {
                row["class"]: row
                for row in items
                if row["item_id"]
                in {pair["eval_item_id"], pair["deploy_item_id"]}
            }
            self.assertEqual(set(members), {"eval", "deploy"})
            self.assertEqual(
                members["eval"]["content_sha256"],
                members["deploy"]["content_sha256"],
            )
        folds = json.loads(
            (self.dataset_dir / "folds.json").read_text(encoding="utf-8")
        )
        self.assertEqual(folds["split_unit"], "frame_family_id")
        self.assertEqual(len(folds["folds"]), 5)
        manifest = json.loads(
            (self.dataset_dir / "dataset_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        for name, expected in manifest["artifacts"].items():
            self.assertEqual(sha256_file(self.dataset_dir / name), expected)

        models_path = self.dataset_dir.parent / f"{self.dataset_id}_models.json"
        results_path = self.dataset_dir.parent / f"{self.dataset_id}_results.jsonl"
        try:
            models_path.write_text(
                json.dumps(
                    {
                        "models": [
                            {
                                "model_id": "heldout_model_a",
                                "model_revision": "heldout-a-rev1",
                                "lineage": "family_a",
                                "disjoint_from_generators_attestation": True,
                                "disjoint_from_subject_models_attestation": True,
                            },
                            {
                                "model_id": "heldout_model_b",
                                "model_revision": "heldout-b-rev1",
                                "lineage": "family_b",
                                "disjoint_from_generators_attestation": True,
                                "disjoint_from_subject_models_attestation": True,
                            },
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            by_id = {
                row["blind_item_id"]: row for row in comprehension_key
            }
            results_path.write_text(
                "".join(
                    json.dumps(
                        {
                            "blind_item_id": blind_id,
                            "model_id": model_id,
                            "prediction": row["class"],
                        }
                    )
                    + "\n"
                    for model_id in ("heldout_model_a", "heldout_model_b")
                    for blind_id, row in sorted(by_id.items())
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ContrastiveError,
                "both languages, all 20 pairs",
            ):
                validate_heldout_comprehension_results(
                    self.dataset_dir,
                    results_path,
                    models_path,
                )
        finally:
            for path in (models_path, results_path):
                if path.is_file():
                    path.unlink()

    def test_postprocess_rejects_rendered_artifact_tamper(self) -> None:
        stimulus = self._jsonl(self.run_dir_a / "stimulus_manifest.jsonl")
        target = Path(stimulus[0]["derived_outputs"][0]["output_path"])
        target.write_text(
            target.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ContrastiveError, "Rendered output changed"):
            assemble_post_generation(
                run_dir_a=self.run_dir_a,
                run_dir_b=self.run_dir_b,
                output_dir=self.dataset_dir,
            )

    def test_postprocess_rejects_nonpassing_accepted_state(self) -> None:
        state_paths = sorted((self.run_dir_a / "state").glob("*.json"))
        state = json.loads(state_paths[0].read_text(encoding="utf-8"))
        state["attempts"][0]["verification"]["pass"] = False
        state_paths[0].write_text(
            json.dumps(state, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        result_path = self.run_dir_a / "generation_result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["state_bundle_sha256"] = hash_object(
            [
                {"job_id": path.stem, "sha256": sha256_file(path)}
                for path in state_paths
            ]
        )
        result_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            ContrastiveError,
            "complete passing verification",
        ):
            assemble_post_generation(
                run_dir_a=self.run_dir_a,
                run_dir_b=self.run_dir_b,
                output_dir=self.dataset_dir,
            )

    def test_lexical_control_crossing_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory(dir=CONTRASTIVE_ROOT) as temporary:
            path = Path(temporary) / "lexical.yaml"
            value = yaml.safe_load(
                (
                    CONTRASTIVE_ROOT
                    / "validation"
                    / "lexical_controls.yaml"
                ).read_text(encoding="utf-8")
            )
            value["items"][4]["intended_purpose"] = "evaluation"
            path.write_text(
                yaml.safe_dump(value, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ContrastiveError, "not crossed"):
                validate_lexical_controls(path)

    def test_format_matched_controls_are_fresh_balanced_and_full_length(self) -> None:
        value = validate_format_matched_controls()
        self.assertEqual(value["status"], "frozen_unscored")
        self.assertEqual(len(value["items"]), 256)
        self.assertTrue(all("\n\n" in row["text"] for row in value["items"]))

    def test_format_matched_cue_outside_cue_line_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=CONTRASTIVE_ROOT) as temporary:
            path = Path(temporary) / "format_matched.yaml"
            value = yaml.safe_load(
                (
                    CONTRASTIVE_ROOT
                    / "validation"
                    / "format_matched_controls.yaml"
                ).read_text(encoding="utf-8")
            )
            value["blocks"][0]["payload"]["en"] += " The audit field is blank."
            path.write_text(
                yaml.safe_dump(value, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ContrastiveError, "outside the cue line"):
                validate_format_matched_controls(path)


class EvaluationContractTests(unittest.TestCase):
    def test_fitting_recipe_has_no_silent_default_or_all_items_refit(self) -> None:
        contract = _evaluation_contract()
        status = contract["fitting_recipe_status"]
        self.assertTrue(status["no_default"])
        self.assertEqual(
            status["status"],
            "blocked_no_production_recipe_after_three_rejected_confirmations",
        )
        self.assertEqual(
            status["allowed_strata"],
            ["explicit_only", "implicit_only", "all_matched_items"],
        )
        final_direction = contract["fitting_protocol"]["final_direction"]
        self.assertIn("exactly the explicitness stratum or strata", final_direction)
        self.assertIn("Do not silently expand", final_direction)
        candidates = status["historical_confirmation_sequence"]
        self.assertEqual(
            candidates[0]["cell"],
            {"position": "mean", "layer": 6},
        )
        self.assertEqual(len(candidates), 3)
        self.assertTrue(all("rejected" in row["verdict"] for row in candidates))
        self.assertIn("No activation extraction", status["next_authorized_step"])
        self.assertEqual(
            contract["success_criteria_status"]["status"],
            "historical_contract_requires_prospective_amendment",
        )
        controls = contract["frozen_success_criteria"]["lexical_controls"]
        fresh = controls["format_matched_confirmatory_controls"]
        self.assertEqual(fresh["status"], "spent_development_only")
        self.assertEqual(fresh["frozen_source_status"], "frozen_unscored")
        self.assertEqual(fresh["items"], 256)
        self.assertEqual(fresh["absolute_zero_threshold_accuracy"], "diagnostic_only")

    def test_confirmatory_extraction_metadata_matches_shared_loader(self) -> None:
        items, controls = validate_confirmatory_extraction_inputs(
            DEFAULT_ITEMS,
            DEFAULT_CONTROLS,
        )
        self.assertEqual(len(items), 1200)
        self.assertEqual(len(controls), 256)
        self.assertIn("cue_locus", FIT_METADATA_FIELDS)
        self.assertEqual({row["pair_id"] for row in items}, set(CUE_LOCUS_BY_PAIR))
        catalog = yaml.safe_load(
            (CONTRASTIVE_ROOT / "purpose_frames.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(
            CUE_LOCUS_BY_PAIR,
            {pair["pair_id"]: pair["cue_locus"] for pair in catalog["pairs"]},
        )
        self.assertTrue(
            all(
                row["cue_locus"] in {"both", "verb_only", "noun_only"}
                for row in items
            )
        )

    def test_confirmatory_colab_notebook_wraps_frozen_extractor(self) -> None:
        notebook_path = (
            CONTRASTIVE_ROOT
            / "probe_check"
            / "confirmations"
            / "01_format_matched"
            / "colab_extract_confirmatory_8b.ipynb"
        )
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        sources = [
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        ]
        combined = "\n".join(sources)
        self.assertIn("extract_confirmatory_8b.py", combined)
        self.assertIn("LOAD_IN_4BIT = False", combined)
        self.assertIn("run_model(SWALLOW)", combined)
        self.assertIn("run_model(META)", combined)
        for model in FROZEN_MODELS:
            self.assertIn(model, combined)
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell.get("source", []))
            source = "\n".join(
                line for line in source.splitlines() if not line.startswith("%")
            )
            compile(source, f"{notebook_path.name}:cell-{index}", "exec")

    def test_second_cue_invariant_bank_is_fresh_and_balanced(self) -> None:
        bank = validate_cue_invariant_bank()
        self.assertEqual(bank["status"], "frozen_unscored")
        self.assertEqual(bank["development"]["method"], "matched_cue_difference_svd")
        self.assertEqual(len(bank["items"]), 256)
        first = validate_format_matched_controls()
        self.assertFalse(
            {row["text"] for row in bank["items"]}
            & {row["text"] for row in first["items"]}
        )

    def test_cue_invariant_colab_notebook_is_a100_no_search_wrapper(self) -> None:
        notebook_path = (
            CONTRASTIVE_ROOT
            / "probe_check"
            / "confirmations"
            / "02_cue_invariant"
            / "colab_extract_cue_invariant_confirmatory_8b.ipynb"
        )
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        combined = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        self.assertIn("extract_cue_invariant_confirmatory_8b.py", combined)
        self.assertIn("cue_invariant_development.json", combined)
        self.assertIn("--development-result", combined)
        self.assertIn("from google.colab import drive", combined)
        self.assertIn('/content/drive/MyDrive/lang-ea', combined)
        self.assertIn("stderr=subprocess.STDOUT", combined)
        self.assertIn("Full log:", combined)
        self.assertNotIn("SEARCH =", combined)
        self.assertNotIn("VS Code", combined)
        self.assertIn("A100 required", combined)
        self.assertNotIn("--load-in-4bit", combined)
        self.assertNotIn("--dtype", combined)
        self.assertIn('cfg["model_load_dtype"] == "bfloat16"', combined)
        extractor = (
            CONTRASTIVE_ROOT
            / "probe_check"
            / "confirmations"
            / "02_cue_invariant"
            / "extract_cue_invariant_confirmatory_8b.py"
        ).read_text(encoding="utf-8")
        evaluator = (
            CONTRASTIVE_ROOT
            / "probe_check"
            / "confirmations"
            / "02_cue_invariant"
            / "evaluate_cue_invariant_confirmation.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn('add_argument("--load-in-4bit"', extractor)
        self.assertNotIn('add_argument("--dtype"', extractor)
        self.assertIn("EXPECTED_HELPER_SHA256", extractor)
        self.assertIn("validate_helper()", extractor)
        self.assertIn('"dtype": torch.bfloat16', extractor)
        self.assertIn('"load_in_4bit": False', evaluator)
        self.assertIn('"A100" not in', evaluator)
        self.assertNotIn('add_argument("--bootstrap-samples"', evaluator)
        self.assertNotIn('add_argument("--seed"', evaluator)
        self.assertIn("BOOTSTRAP_SAMPLES = 2000", evaluator)
        self.assertIn("run_model(SWALLOW)", combined)
        self.assertIn("run_model(META)", combined)
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell.get("source", []))
            source = "\n".join(
                line for line in source.splitlines() if not line.startswith("%")
            )
            compile(source, f"{notebook_path.name}:cell-{index}", "exec")

    def test_cue_balanced_development_augmentation_is_balanced_and_spent(self) -> None:
        value = validate_and_render()
        self.assertEqual(value["status"], "development_only_spent_inputs")
        self.assertIsNone(value["trigger_result"]["required_selected_recipe"])
        self.assertEqual(len(value["items"]), 1536)
        self.assertEqual(
            {row["cue_template_id"] for row in value["items"]},
            {
                "administrative_middle",
                "category_bracket_middle",
                "index_footer",
            },
        )
        counts: dict[tuple[str, ...], int] = {}
        for row in value["items"]:
            key = (
                row["cue_template_id"],
                row["language"],
                row["surface"],
                row["cue_family"],
                row["intended_purpose"],
                row["lexical_cue"],
            )
            counts[key] = counts.get(key, 0) + 1
        self.assertEqual(len(counts), 96)
        self.assertEqual(set(counts.values()), {16})

    def test_cue_balanced_development_notebook_is_a100_and_no_search(self) -> None:
        notebook_path = (
            CONTRASTIVE_ROOT
            / "probe_check"
            / "development"
            / "cue_constraints"
            / "colab_extract_cue_balanced_development_8b.ipynb"
        )
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        combined = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        self.assertIn("extract_cue_balanced_development_8b.py", combined)
        self.assertIn("cue_balanced_development_augmentation.jsonl", combined)
        self.assertIn("cue_balanced_development.json", combined)
        self.assertIn("from google.colab import drive", combined)
        self.assertIn("/content/drive/MyDrive/lang-ea", combined)
        self.assertIn("A100 required", combined)
        self.assertIn("bfloat16", combined)
        self.assertIn("run_model(SWALLOW)", combined)
        self.assertIn("run_model(META)", combined)
        self.assertNotIn("SEARCH =", combined)
        self.assertNotIn("--load-in-4bit", combined)
        self.assertNotIn("--dtype", combined)
        extractor = (
            CONTRASTIVE_ROOT
            / "probe_check"
            / "development"
            / "cue_constraints"
            / "extract_cue_balanced_development_8b.py"
        ).read_text(encoding="utf-8")
        self.assertIn("EXPECTED_HELPER_SHA256", extractor)
        self.assertIn('"development_only": True', extractor)
        self.assertIn("torch.bfloat16", extractor)
        self.assertIn('"A100" not in', extractor)
        self.assertNotIn('add_argument("--load-in-4bit"', extractor)
        self.assertNotIn('add_argument("--dtype"', extractor)
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell.get("source", []))
            source = "\n".join(
                line for line in source.splitlines() if not line.startswith("%")
            )
            compile(source, f"{notebook_path.name}:cell-{index}", "exec")

    def test_hard_cue_constraint_removes_the_observed_difference_rowspace(self) -> None:
        rows = []
        activations = []
        for family_index, family in enumerate(("explicit_lexeme", "implicit_cue")):
            for language_index, language in enumerate(("en", "ja")):
                for purpose_index, purpose in enumerate(("evaluation", "deployment")):
                    base = np.array(
                        [1.0 + family_index, 2.0 + language_index, 3.0 + purpose_index, 4.0]
                    )
                    cue_shift = np.array([1.0, -2.0, 0.5, 0.0])
                    for lexical_cue, vector in (
                        ("evaluation", base + cue_shift),
                        ("deployment", base),
                    ):
                        rows.append(
                            {
                                "control_block_id": "synthetic_block",
                                "cue_family": family,
                                "language": language,
                                "intended_purpose": purpose,
                                "lexical_cue": lexical_cue,
                                "surface": "external",
                            }
                        )
                        activations.append(vector)

        differences = cue_differences(np.stack(activations), rows)
        basis, diagnostics = rowspace_basis(differences)
        projected = project_null(differences, basis)
        self.assertEqual(diagnostics["rank"], 1)
        self.assertEqual(diagnostics["nullspace_dimension"], 3)
        self.assertLess(float(np.linalg.norm(projected, axis=1).max()), 1e-12)

    def test_third_cue_constrained_bank_is_fresh_balanced_and_frozen(self) -> None:
        value = validate_cue_constrained_bank()
        self.assertEqual(value["status"], "frozen_unscored")
        self.assertEqual(value["development"]["selected_recipe"], "balanced_penalty_256")
        self.assertEqual(value["development"]["cue_penalty"], 256.0)
        self.assertEqual(len(value["items"]), 256)
        self.assertEqual(len({row["control_id"] for row in value["items"]}), 256)
        self.assertEqual(len({row["text"] for row in value["items"]}), 256)
        self.assertTrue(
            all("Record header:" in row["text"] for row in value["items"] if row["language"] == "en")
        )
        self.assertTrue(
            all("記録見出し：" in row["text"] for row in value["items"] if row["language"] == "ja")
        )

    def test_cue_constrained_confirmation_notebook_is_a100_and_no_search(self) -> None:
        notebook_path = (
            CONTRASTIVE_ROOT
            / "probe_check"
            / "confirmations"
            / "03_cue_constrained"
            / "colab_extract_cue_constrained_confirmatory_8b.ipynb"
        )
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        combined = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        self.assertIn("extract_cue_constrained_confirmatory_8b.py", combined)
        self.assertIn("cue_constrained_confirmatory_controls.jsonl", combined)
        self.assertIn("cue_constrained_tradeoff_development.json", combined)
        self.assertIn("from google.colab import drive", combined)
        self.assertIn("/content/drive/MyDrive/lang-ea", combined)
        self.assertIn("A100 required", combined)
        self.assertIn("run_model(SWALLOW)", combined)
        self.assertIn("run_model(META)", combined)
        self.assertNotIn("SEARCH =", combined)
        self.assertNotIn("--load-in-4bit", combined)
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell.get("source", []))
            source = "\n".join(
                line for line in source.splitlines() if not line.startswith("%")
            )
            compile(source, f"{notebook_path.name}:cell-{index}", "exec")

        extractor = (
            CONTRASTIVE_ROOT
            / "probe_check"
            / "confirmations"
            / "03_cue_constrained"
            / "extract_cue_constrained_confirmatory_8b.py"
        ).read_text(encoding="utf-8")
        evaluator = (
            CONTRASTIVE_ROOT
            / "probe_check"
            / "confirmations"
            / "03_cue_constrained"
            / "evaluate_cue_constrained_confirmation.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"balanced_penalty_256"', extractor)
        self.assertIn("torch.bfloat16", extractor)
        self.assertIn('"A100" not in', extractor)
        self.assertIn("FROZEN_RECIPE", evaluator)
        self.assertNotIn("for recipe in", evaluator)


if __name__ == "__main__":
    unittest.main()
