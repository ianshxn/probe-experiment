from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import sys
import threading
import unittest
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from unittest.mock import patch


CONTRASTIVE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CONTRASTIVE_ROOT / "src"))

from contrastive_generation import preflight
from contrastive_generation import pre_generation as pre_generation_module
from contrastive_generation.catalog import load_catalog
from contrastive_generation.content_specs import (
    canonical_content_shape,
    compile_content_specs,
    model_topic,
    validate_bilingual_content_specs,
)
from contrastive_generation.planning import load_provider_profile
from contrastive_generation.pre_generation import (
    _derived_containers,
    _production_grid_gate,
    _validate_manifest_grid,
    apply_pre_generation_overrides,
    build_pre_generation_plan,
    load_pre_generation_config,
    prepare_generation_handoff,
    validate_pre_generation_provider,
    validate_path_layout,
    validate_run_id,
    verify_pre_generation_run,
    with_pre_generation_provider,
    write_pre_generation_plan,
)
from contrastive_generation.preflight import (
    LATIN_RE,
    LOCKED_FILES,
    REQUEST_PROJECTION_FILES,
    audit_model_request,
    cue_hits,
    japanese_language_hits,
    model_visible_strings,
    render_model_request,
    validate_renderer_cue_contract,
    validate_scaffolding_bundle,
)
from contrastive_generation.utils import (
    ContrastiveError,
    hash_object,
    load_yaml,
    sha256_file,
)


def configured_run() -> dict:
    return with_pre_generation_provider(
        load_pre_generation_config(),
        load_provider_profile("tests/fixtures/provider.yaml"),
    )


def protected_digest() -> str:
    paths = [
        CONTRASTIVE_ROOT / "topics.yaml",
        CONTRASTIVE_ROOT / "topics.ja.yaml",
        CONTRASTIVE_ROOT / "containers.yaml",
        *sorted((CONTRASTIVE_ROOT / "containers").glob("*.yaml")),
    ]
    rows = "\n".join(
        f"{path.relative_to(CONTRASTIVE_ROOT).as_posix()}\t{sha256_file(path)}"
        for path in paths
    )
    return hashlib.sha256(rows.encode("utf-8")).hexdigest()


class ContentSpecCompilerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = validate_scaffolding_bundle(configured_run())
        cls.en_catalog = load_catalog("topics.yaml")
        cls.ja_catalog = load_catalog("topics.ja.yaml")
        cls.en_specs = compile_content_specs(
            cls.en_catalog, cls.bundle.scaffolding
        )
        cls.ja_specs = compile_content_specs(
            cls.ja_catalog, cls.bundle.scaffolding
        )

    def test_compiles_ten_bilingual_template_blind_specs(self) -> None:
        self.assertEqual(
            [row.content_spec_id for row in self.en_specs],
            [f"content_p{index:02d}" for index in range(1, 11)],
        )
        validate_bilingual_content_specs(self.en_specs, self.ja_specs)
        for en, ja in zip(self.en_specs, self.ja_specs, strict=True):
            self.assertEqual(
                canonical_content_shape(en),
                canonical_content_shape(ja),
            )
            self.assertEqual(
                {row.data["class"] for row in en.containers},
                {"eval", "deploy"},
            )
            payload = en.model_payload()
            projection = en.model_projection()
            for forbidden_attribute in (
                "containers",
                "container_id",
                "pair_id",
                "content_spec_id",
                "body_template",
                "purpose_frame",
                "nuisance_profile",
                "class",
            ):
                self.assertFalse(hasattr(projection, forbidden_attribute))
            self.assertEqual(
                set(payload),
                {
                    "register",
                    "role_structure",
                    "slots",
                    "response_contract",
                    "model_guidance",
                },
            )
            serialized = json.dumps(payload, ensure_ascii=False)
            for forbidden in (
                "body_template",
                "purpose_frame",
                "nuisance_profile",
                "container_id",
                "pair_id",
                "content_spec_id",
                "design_notes",
                "display_name",
            ):
                self.assertNotIn(forbidden, serialized)

    def test_compilation_and_planning_do_not_write_protected_sources(self) -> None:
        before = protected_digest()
        config = apply_pre_generation_overrides(
            configured_run(),
            topics=["t001"],
            content_specs=["content_p01"],
        )
        build_pre_generation_plan(config, "protected_source_check")
        self.assertEqual(before, protected_digest())

    def test_generated_content_cue_scope_narrows_the_request_scope(self) -> None:
        lexicon = self.bundle.cue_lexicon
        for language in ("en", "ja"):
            self.assertTrue(
                set(lexicon["generated_content"][language])
                <= set(lexicon["languages"][language]),
                f"{language} content scope must narrow the request scope",
            )
        # Polysemous ordinary-domain words screen out of requests but are legal
        # inside generated task text; the contrast lexemes screen out of both.
        cases = [
            ("ja", "二要素認証の設定を確認してください。", True, False),
            ("ja", "この回答は運用に用います。", True, True),
            ("en", "Update the compliance dataset.", True, False),
            ("en", "This evaluation is complete.", True, True),
        ]
        for language, text, in_request, in_content in cases:
            self.assertEqual(
                bool(preflight.text_cue_terms(text, language, lexicon)),
                in_request,
                f"request scope disagreed on {text!r}",
            )
            self.assertEqual(
                bool(
                    preflight.text_cue_terms(
                        text,
                        language,
                        lexicon,
                        scope="generated_content",
                    )
                ),
                in_content,
                f"content scope disagreed on {text!r}",
            )
        with self.assertRaisesRegex(ContrastiveError, "scope"):
            preflight.text_cue_terms("x", "en", lexicon, scope="nope")

    def test_english_cue_screen_catches_regular_inflection(self) -> None:
        lexicon = self.bundle.cue_lexicon
        # Japanese substring matching already catches every form of 評価; the
        # English screen must not be weaker, or the two languages are filtered
        # at different strengths.
        for word in (
            "evaluations",
            "assessments",
            "tests",
            "deployed",
            "deploying",
            "benchmarks",
            "grades",
            "scores",
            "probing",
        ):
            self.assertTrue(
                preflight.text_cue_terms(f"the {word} are ready", "en", lexicon),
                f"{word!r} escaped the English cue screen",
            )
        for word in ("contest", "testament", "greatest", "upgrade", "problem"):
            self.assertFalse(
                preflight.text_cue_terms(f"the {word} is here", "en", lexicon),
                f"{word!r} is not a cue term",
            )

    def test_japanese_deployment_lexeme_is_screened(self) -> None:
        # 運用 is the canonical Japanese deployment lexeme in the lexical
        # controls, so the generation screen must reject it too.
        self.assertIn(
            "運用",
            self.bundle.cue_lexicon["generated_content"]["ja"],
        )

    def test_renderer_cue_contract_is_language_paired(self) -> None:
        audit = validate_renderer_cue_contract(
            self.en_specs, self.ja_specs, self.bundle.cue_lexicon
        )
        self.assertEqual(audit["matched_bodies_checked"], 80)
        self.assertEqual(audit["body_cue_hits"], 0)
        container = next(
            container
            for spec in self.en_specs
            for container in spec.containers
        )
        original = container.data["body_template"]
        try:
            container.data["body_template"] = original + "\nEvaluation"
            with self.assertRaisesRegex(
                ContrastiveError, "outside the frame"
            ):
                validate_renderer_cue_contract(
                    self.en_specs,
                    self.ja_specs,
                    self.bundle.cue_lexicon,
                )
        finally:
            container.data["body_template"] = original


class RequestProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = configured_run()
        cls.bundle = validate_scaffolding_bundle(cls.config)
        cls.en_catalog = load_catalog("topics.yaml")
        cls.ja_catalog = load_catalog("topics.ja.yaml")
        cls.en_spec = compile_content_specs(
            cls.en_catalog, cls.bundle.scaffolding
        )[0]
        cls.ja_spec = compile_content_specs(
            cls.ja_catalog, cls.bundle.scaffolding
        )[0]

    def request(self, language: str = "en") -> dict:
        catalog = self.en_catalog if language == "en" else self.ja_catalog
        spec = self.en_spec if language == "en" else self.ja_spec
        return render_model_request(
            CONTRASTIVE_ROOT
            / self.config["paths"]["prompts"][language],
            model_topic(catalog.topics[0]),
            spec.model_projection(),
            self.bundle.scaffolding,
            self.config["generation"],
            12345,
        )

    def test_request_is_template_blind_under_private_mutations(self) -> None:
        request_before = self.request()
        container = self.en_spec.containers[0]
        original = copy.deepcopy(container.data)
        try:
            container.data["body_template"] = (
                "{{PURPOSE_FRAME}}\nPRIVATE WRAPPER {{TASK}} {{CONTENT}}"
            )
            container.data["purpose_frame"]["text"] = "Private frame."
            container.data["nuisance_profile"]["speaker_present"] = True
            container.data["class"] = "deploy"
            container.data["display_name"] = "private display"
            container.data["design_notes"] = "private design"
            request_after = self.request()
            self.assertEqual(request_before, request_after)
        finally:
            container.data.clear()
            container.data.update(original)

    def test_request_constructor_rejects_private_records(self) -> None:
        with self.assertRaisesRegex(
            ContrastiveError, "exact ModelTopic"
        ):
            render_model_request(
                CONTRASTIVE_ROOT
                / self.config["paths"]["prompts"]["en"],
                self.en_catalog.topics[0],
                self.en_spec,
                self.bundle.scaffolding,
                self.config["generation"],
                12345,
            )

    def test_wrapper_mutation_moves_derivation_not_request(self) -> None:
        request_hash = hash_object(self.request())
        container = self.en_spec.containers[0]
        original = str(container.data["body_template"])
        try:
            _, before = _derived_containers(
                self.en_spec,
                12345,
                CONTRASTIVE_ROOT / "data" / "pilot",
                "blindness",
                "t001",
                0,
                "a" * 64,
                "b" * 64,
            )
            container.data["body_template"] = (
                original + "\nPrivate wrapper suffix"
            )
            _, after = _derived_containers(
                self.en_spec,
                12345,
                CONTRASTIVE_ROOT / "data" / "pilot",
                "blindness",
                "t001",
                0,
                "a" * 64,
                "b" * 64,
            )
            self.assertNotEqual(before, after)
            self.assertEqual(request_hash, hash_object(self.request()))
        finally:
            container.data["body_template"] = original

    def test_exact_japanese_request_surface_has_no_latin(self) -> None:
        request = self.request("ja")
        self.assertEqual(japanese_language_hits(request), [])
        for pointer, text in model_visible_strings(request):
            self.assertIsNone(
                LATIN_RE.search(text),
                f"Latin text at {pointer}: {text!r}",
            )

    def test_audits_detect_cues_language_and_private_leaks(self) -> None:
        english = self.request()
        bad_cue = copy.deepcopy(english)
        bad_cue["user"] += "\nThis was evaluated."
        self.assertTrue(cue_hits(bad_cue, "en", self.bundle.cue_lexicon))

        japanese = self.request("ja")
        bad_language = copy.deepcopy(japanese)
        bad_language["response_schema"]["properties"]["EnglishKey"] = {
            "type": "string"
        }
        self.assertTrue(japanese_language_hits(bad_language))

        bad_leak = copy.deepcopy(english)
        bad_leak["user"] += (
            f"\n{self.en_spec.containers[0].data['body_template']}"
        )
        with self.assertRaisesRegex(ContrastiveError, "leaks"):
            audit_model_request(
                bad_leak,
                self.en_spec,
                self.bundle.cue_lexicon,
                self.en_catalog.topics[0],
            )
        bad_topic = copy.deepcopy(english)
        bad_topic["user"] += f"\n{self.en_catalog.topics[0].topic_id}"
        with self.assertRaisesRegex(ContrastiveError, "topic_id"):
            audit_model_request(
                bad_topic,
                self.en_spec,
                self.bundle.cue_lexicon,
                self.en_catalog.topics[0],
            )

    def test_nested_schema_surfaces_fail_closed(self) -> None:
        japanese = self.request("ja")
        schema = japanese["response_schema"]
        schema["$defs"] = {
            "補助": {
                "type": "string",
                "description": "EnglishDefinition",
            }
        }
        schema["properties"]["配列"] = {
            "type": "array",
            "items": {
                "anyOf": [
                    {"type": "string", "title": "EnglishNested"},
                    {"type": "string", "pattern": "EnglishPattern"},
                ]
            },
        }
        schema["properties"]["参照"] = {"$ref": "#/$defs/補助"}
        schema["required"].extend(["配列", "参照"])
        hit_paths = {
            row["path"] for row in japanese_language_hits(japanese)
        }
        self.assertTrue({
            "/response_schema/$defs/補助/description",
            "/response_schema/properties/配列/items/anyOf/0/title",
            "/response_schema/properties/配列/items/anyOf/1/pattern",
        } <= hit_paths)

        cue_request = self.request()
        cue_request["response_schema"]["properties"]["nested"] = {
            "type": "array",
            "items": {
                "type": "string",
                "description": "This field was evaluated.",
            },
        }
        cue_request["response_schema"]["required"].append("nested")
        self.assertTrue(
            cue_hits(cue_request, "en", self.bundle.cue_lexicon)
        )

        private_request = self.request()
        private_request["response_schema"]["properties"]["nested"] = {
            "type": "string",
            "description": self.en_spec.content_spec_id,
        }
        private_request["response_schema"]["required"].append("nested")
        with self.assertRaisesRegex(ContrastiveError, "leaks"):
            audit_model_request(
                private_request,
                self.en_spec,
                self.bundle.cue_lexicon,
                self.en_catalog.topics[0],
            )

        external_reference = self.request("ja")
        external_reference["response_schema"]["properties"]["参照"] = {
            "$ref": "https://example.invalid/schema.json"
        }
        external_reference["response_schema"]["required"].append("参照")
        with self.assertRaisesRegex(
            ContrastiveError, "External JSON-Schema reference"
        ):
            model_visible_strings(external_reference)

        nonschema_reference = self.request("ja")
        nonschema_reference["response_schema"]["properties"]["参照"] = {
            "$ref": "#/required/0"
        }
        nonschema_reference["response_schema"]["required"].append("参照")
        with self.assertRaisesRegex(
            ContrastiveError, "does not target a schema"
        ):
            model_visible_strings(nonschema_reference)

        unknown_extension = self.request("ja")
        unknown_extension["response_schema"]["x-model-note"] = "English"
        with self.assertRaisesRegex(
            ContrastiveError, "Unsupported JSON-Schema keyword"
        ):
            model_visible_strings(unknown_extension)

    def test_word_boundaries_avoid_substring_false_positives(self) -> None:
        request = self.request()
        request["user"] += "\nUse the contest details and upgrade the package."
        self.assertEqual(
            cue_hits(request, "en", self.bundle.cue_lexicon), []
        )


class FullPlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = configured_run()
        cls.plan, cls.jobs, cls.requests = build_pre_generation_plan(
            cls.config, "full_v2_test"
        )

    def test_full_bilingual_plan_has_800_shared_jobs(self) -> None:
        self.assertEqual(len(self.jobs), 800)
        self.assertEqual(len(self.requests), 2400)
        self.assertEqual(
            self.plan["counts"]["planned_generation_attempts"], 2400
        )
        self.assertEqual(self.plan["counts"]["request_artifacts"], 2400)
        self.assertEqual(self.plan["counts"]["derived_items"], 3200)
        self.assertEqual(
            self.plan["counts"]["by_language"], {"en": 400, "ja": 400}
        )
        self.assertEqual(
            self.plan["counts"]["derived_by_class"],
            {"deploy": 1600, "eval": 1600},
        )
        self.assertEqual(
            set(self.plan["counts"]["by_domain"].values()), {100}
        )
        for job in self.jobs:
            self.assertEqual(len(job["generation_attempts"]), 3)
            self.assertEqual(
                [
                    attempt["attempt_index"]
                    for attempt in job["generation_attempts"]
                ],
                [0, 1, 2],
            )
            self.assertEqual(len(job["derived_containers"]), 4)
            self.assertEqual(
                Counter(
                    row["class"] for row in job["derived_containers"]
                ),
                {"eval": 2, "deploy": 2},
            )
            self.assertEqual(
                len({row["pair_id"] for row in job["derived_containers"]}),
                2,
            )
            self.assertEqual(
                len(
                    {
                        row["frame_family_id"]
                        for row in job["derived_containers"]
                    }
                ),
                2,
            )
            self.assertTrue(
                all(
                    row["explicitness"] in {"explicit", "implicit"}
                    and row["surface"] in {"external", "institutional"}
                    for row in job["derived_containers"]
                )
            )

    def test_all_exact_requests_pass_recorded_audits(self) -> None:
        for job in self.jobs:
            attempts = job["generation_attempts"]
            self.assertEqual(
                len({attempt["seed"] for attempt in attempts}), 3
            )
            primary = self.requests[attempts[0]["attempt_id"]]
            for attempt in attempts:
                audit = attempt["preflight"]
                self.assertEqual(audit["placeholder_audit"], "passed")
                self.assertEqual(audit["cue_audit"], "passed")
                self.assertEqual(audit["language_audit"], "passed")
                self.assertEqual(audit["leakage_audit"], "passed")
                request = self.requests[attempt["attempt_id"]]
                if job["language"] == "ja":
                    self.assertEqual(
                        japanese_language_hits(request), []
                    )
                self.assertEqual(
                    {
                        **request,
                        "generation": {
                            **request["generation"],
                            "seed": primary["generation"]["seed"],
                        },
                    },
                    primary,
                )
        self.assertEqual(
            len({job["job_id"] for job in self.jobs}), 800
        )
        self.assertEqual(
            len({
                attempt["request_sha256"]
                for job in self.jobs
                for attempt in job["generation_attempts"]
            }),
            2400,
        )

    def test_selection_order_does_not_change_retained_jobs(self) -> None:
        base = configured_run()
        first = apply_pre_generation_overrides(
            base,
            topics=["t002", "t001"],
            content_specs=["content_p02", "content_p01"],
            languages=["ja", "en"],
        )
        second = apply_pre_generation_overrides(
            base,
            topics=["t001", "t002"],
            content_specs=["content_p01", "content_p02"],
            languages=["en", "ja"],
        )
        first_plan, first_jobs, first_requests = build_pre_generation_plan(
            first, "order_invariant"
        )
        second_plan, second_jobs, second_requests = build_pre_generation_plan(
            second, "order_invariant"
        )
        self.assertEqual(first_jobs, second_jobs)
        self.assertEqual(first_requests, second_requests)
        self.assertEqual(
            first_plan["plan_identity_sha256"],
            second_plan["plan_identity_sha256"],
        )

    def test_run_id_does_not_change_generation_identity(self) -> None:
        subset = apply_pre_generation_overrides(
            configured_run(),
            topics=["t001"],
            content_specs=["content_p01"],
        )
        _, first_jobs, _ = build_pre_generation_plan(subset, "run_one")
        _, second_jobs, _ = build_pre_generation_plan(subset, "run_two")
        for first, second in zip(first_jobs, second_jobs, strict=True):
            self.assertEqual(first["job_id"], second["job_id"])
            self.assertEqual(
                first["derivation_seed"], second["derivation_seed"]
            )
            self.assertEqual(
                [
                    row["request_sha256"]
                    for row in first["generation_attempts"]
                ],
                [
                    row["request_sha256"]
                    for row in second["generation_attempts"]
                ],
            )
            self.assertNotEqual(
                first["generation_attempts"][0]["request_path"],
                second["generation_attempts"][0]["request_path"],
            )

    def test_cost_report_has_baseline_and_retry_scenarios(self) -> None:
        cost = self.plan["cost_estimate"]
        self.assertEqual(
            cost["baseline"]["generation_calls"], 800
        )
        self.assertGreater(cost["baseline"]["estimated_usd"], 0)
        self.assertEqual(
            cost["generation_retry_scenario"]["attempt_multiplier"], 3
        )
        self.assertEqual(
            cost["generation_retry_scenario"]["planned_generation_calls"],
            2400,
        )
        self.assertGreaterEqual(
            cost["generation_retry_scenario"]["estimated_usd"],
            cost["baseline"]["estimated_usd"],
        )

    def test_manifest_must_equal_the_configured_cartesian_grid(self) -> None:
        subset = apply_pre_generation_overrides(
            configured_run(),
            topics=["t001", "t002"],
            content_specs=["content_p01"],
            languages=["en"],
        )
        _, jobs, _ = build_pre_generation_plan(subset, "grid_closure")
        _validate_manifest_grid(jobs, subset)
        broken = copy.deepcopy(jobs)
        broken[-1]["topic_id"] = broken[0]["topic_id"]
        with self.assertRaisesRegex(ContrastiveError, "Cartesian grid"):
            _validate_manifest_grid(broken, subset)

    def test_regeneration_schedule_is_prefix_stable(self) -> None:
        zero = apply_pre_generation_overrides(
            configured_run(),
            topics=["t001"],
            content_specs=["content_p01"],
            languages=["en"],
        )
        two = copy.deepcopy(zero)
        three = copy.deepcopy(zero)
        zero["generation"]["max_regenerations"] = 0
        two["generation"]["max_regenerations"] = 2
        three["generation"]["max_regenerations"] = 3
        _, zero_jobs, zero_requests = build_pre_generation_plan(
            zero, "retry_prefix"
        )
        _, two_jobs, two_requests = build_pre_generation_plan(
            two, "retry_prefix"
        )
        _, three_jobs, three_requests = build_pre_generation_plan(
            three, "retry_prefix"
        )
        primary_zero = zero_jobs[0]["generation_attempts"][0]
        primary_two = two_jobs[0]["generation_attempts"][0]
        self.assertEqual(zero_jobs[0]["job_id"], two_jobs[0]["job_id"])
        self.assertEqual(
            zero_jobs[0]["derivation_seed"],
            two_jobs[0]["derivation_seed"],
        )
        self.assertEqual(primary_zero, primary_two)
        self.assertEqual(
            zero_requests[primary_zero["attempt_id"]],
            two_requests[primary_two["attempt_id"]],
        )
        self.assertEqual(
            zero_jobs[0]["derived_containers"],
            two_jobs[0]["derived_containers"],
        )
        self.assertEqual(len(two_jobs[0]["generation_attempts"]), 3)
        self.assertEqual(
            two_jobs[0]["generation_attempts"],
            three_jobs[0]["generation_attempts"][:3],
        )
        for attempt in two_jobs[0]["generation_attempts"]:
            self.assertEqual(
                two_requests[attempt["attempt_id"]],
                three_requests[attempt["attempt_id"]],
            )

    def test_operational_provider_fields_do_not_change_job_identity(self) -> None:
        base = apply_pre_generation_overrides(
            configured_run(),
            topics=["t001"],
            content_specs=["content_p01"],
            languages=["en"],
        )
        changed = copy.deepcopy(base)
        changed["provider_profile"]["input_usd_per_million"] = 99
        changed["provider_profile"]["max_concurrency"] = 9
        _, first_jobs, _ = build_pre_generation_plan(
            base, "provider_projection"
        )
        _, second_jobs, _ = build_pre_generation_plan(
            changed, "provider_projection"
        )
        self.assertEqual(
            [job["job_id"] for job in first_jobs],
            [job["job_id"] for job in second_jobs],
        )
        self.assertEqual(
            [
                attempt["request_sha256"]
                for job in first_jobs
                for attempt in job["generation_attempts"]
            ],
            [
                attempt["request_sha256"]
                for job in second_jobs
                for attempt in job["generation_attempts"]
            ],
        )
        self.assertNotEqual(
            first_jobs[0]["provider_profile_sha256"],
            second_jobs[0]["provider_profile_sha256"],
        )

    def test_model_revision_changes_job_identity(self) -> None:
        base = apply_pre_generation_overrides(
            configured_run(),
            topics=["t001"],
            content_specs=["content_p01"],
            languages=["en"],
        )
        changed = copy.deepcopy(base)
        changed["provider_profile"]["model_revision"] = "test-revision-2"
        _, first_jobs, _ = build_pre_generation_plan(
            base, "model_revision"
        )
        _, second_jobs, _ = build_pre_generation_plan(
            changed, "model_revision"
        )
        self.assertNotEqual(
            first_jobs[0]["job_id"], second_jobs[0]["job_id"]
        )


class GateAndIntegrityTests(unittest.TestCase):
    def test_approved_semantic_and_calibration_unblock_production(self) -> None:
        # Exercise the approved branch with explicit test reviewer identities so
        # the test remains independent of who approved the checked-in bundle.
        config = copy.deepcopy(configured_run())
        config["mode"] = "production"
        approval_path = (
            CONTRASTIVE_ROOT
            / "prompts"
            / "pre_generation_semantic_approval.json"
        ).resolve()
        lock_path = (
            CONTRASTIVE_ROOT / "prompts" / "pre_generation.lock.json"
        ).resolve()
        real_load_json = preflight.load_json
        real_plan_load_json = pre_generation_module.load_json
        frame_report_path = (
            CONTRASTIVE_ROOT
            / config["paths"]["frame_calibration_report"]
        ).resolve()
        comprehension_report_path = (
            CONTRASTIVE_ROOT
            / config["paths"]["heldout_comprehension_report"]
        ).resolve()
        en_catalog = load_catalog()
        ja_catalog = load_catalog("topics.ja.yaml")
        target_hashes = {
            "purpose_frame_catalog_sha256": sha256_file(
                en_catalog.purpose_frame_path
            ),
            "container_catalog_sha256_en": (
                en_catalog.container_catalog_sha256()
            ),
            "container_catalog_sha256_ja": (
                ja_catalog.container_catalog_sha256()
            ),
        }

        def approved_load_json(path: Path) -> Any:
            value = real_load_json(path)
            if path == approval_path:
                return {
                    **value,
                    "status": "approved",
                    "researcher": "test researcher",
                    "native_japanese_reviewer": "test reviewer",
                    "approved_at": "2026-07-27T00:00:00+00:00",
                }
            if path == lock_path:
                return {**value, "semantic_equivalence_review": "approved"}
            return value

        def calibration_load_json(path: Path) -> Any:
            value = real_plan_load_json(path)
            metric = {
                "items": 20,
                "correct": 15,
                "balanced_accuracy": 0.75,
                "eval_accuracy": 0.75,
                "deploy_accuracy": 0.75,
                "class_accuracy_gap": 0.0,
                "wilson_95_lower_bound": 0.53,
            }
            if path == frame_report_path:
                return {
                    **value,
                    **target_hashes,
                    "status": "passed",
                    "packet_sha256": "1" * 64,
                    "results_sha256": "2" * 64,
                    "raters_sha256": "3" * 64,
                    "cohort_id": "test_confirmatory_cohort",
                    "confirmatory_raters": 5,
                    "minimum_ratings_per_item": 5,
                    "languages": {
                        language: {
                            "strata": {
                                "implicit": metric,
                                "explicit": {
                                    **metric,
                                    "correct": 18,
                                    "balanced_accuracy": 0.9,
                                    "eval_accuracy": 0.9,
                                    "deploy_accuracy": 0.9,
                                },
                            },
                            "gates": {
                                "implicit_min": True,
                                "implicit_max": True,
                                "implicit_above_chance": True,
                                "implicit_class_symmetry": True,
                                "explicit_min": True,
                                "explicit_minus_implicit": True,
                            },
                            "pass": True,
                        }
                        for language in ("en", "ja")
                    },
                }
            if path == comprehension_report_path:
                approval_record = real_load_json(approval_path)
                current_profile = config["provider_profile"]
                return {
                    **value,
                    **target_hashes,
                    "status": "passed",
                    "dataset_sha256": "4" * 64,
                    "results_sha256": "5" * 64,
                    "models_sha256": "6" * 64,
                    "minimum_items_per_language_pair_class_cell": 1,
                    "catalog_semantics_sha256": approval_record[
                        "catalog_semantics_sha256"
                    ],
                    "request_projection_sha256": approval_record[
                        "request_projection_sha256"
                    ],
                    "generation_settings_sha256": hash_object(
                        config["generation"]
                    ),
                    "generator_models": [
                        {
                            "profile_id": current_profile["profile_id"],
                            "model": current_profile["model"],
                            "model_revision": current_profile[
                                "model_revision"
                            ],
                        },
                        {
                            "profile_id": "other_test_profile",
                            "model": "other-test-model",
                            "model_revision": "other-test-revision",
                        },
                    ],
                    "models": {
                        f"model_{index}": {
                            "model_revision": f"revision_{index}",
                            "lineage": f"lineage_{index}",
                            "languages": {
                                language: {
                                    "strata": {
                                        "implicit": metric,
                                        "explicit": {
                                            **metric,
                                            "correct": 18,
                                            "balanced_accuracy": 0.9,
                                            "eval_accuracy": 0.9,
                                            "deploy_accuracy": 0.9,
                                        },
                                    },
                                    "surface": {
                                        "external": {
                                            "items": 20,
                                            "accuracy": 0.8,
                                        },
                                        "institutional": {
                                            "items": 20,
                                            "accuracy": 0.8,
                                        },
                                    },
                                    "gates": {
                                        "implicit_min": True,
                                        "implicit_above_chance": True,
                                        "implicit_class_symmetry": True,
                                        "explicit_greater_than_implicit": True,
                                    },
                                    "pass": True,
                                }
                                for language in ("en", "ja")
                            },
                            "pass": True,
                        }
                        for index in (1, 2)
                    },
                }
            return value

        with (
            patch.object(
                preflight,
                "load_json",
                side_effect=approved_load_json,
            ),
            patch.object(
                pre_generation_module,
                "load_json",
                side_effect=calibration_load_json,
            ),
        ):
            plan, jobs, _ = build_pre_generation_plan(
                config,
                "production_approved",
            )
        self.assertEqual(plan["preflight"]["status"], "passed")
        self.assertEqual(len(jobs), 800)

    def test_pending_calibration_blocks_production_after_semantic_approval(
        self,
    ) -> None:
        """Calibration is an independent gate, not a consequence of approval state.

        Semantic approval is patched in rather than read from the repository so
        this stays a test of the calibration gate. The checked-in approval
        legitimately flips to pending whenever the catalog semantics change.
        """
        config = copy.deepcopy(configured_run())
        config["mode"] = "production"
        # Test the gate itself, not whatever the shipped config happens to waive.
        config.pop("waived_gates", None)
        approval_path = (
            CONTRASTIVE_ROOT
            / "prompts"
            / "pre_generation_semantic_approval.json"
        ).resolve()
        lock_path = (
            CONTRASTIVE_ROOT / "prompts" / "pre_generation.lock.json"
        ).resolve()
        real_load_json = preflight.load_json

        def approved_load_json(path: Path) -> Any:
            value = real_load_json(path)
            if path == approval_path:
                return {
                    **value,
                    "status": "approved",
                    "researcher": "test researcher",
                    "native_japanese_reviewer": "test reviewer",
                    "approved_at": "2026-07-27T00:00:00+00:00",
                }
            if path == lock_path:
                return {**value, "semantic_equivalence_review": "approved"}
            return value

        with patch.object(
            preflight,
            "load_json",
            side_effect=approved_load_json,
        ):
            with self.assertRaisesRegex(
                ContrastiveError,
                "passed frame_only calibration",
            ):
                build_pre_generation_plan(
                    config,
                    "production_calibration_blocked",
                )

    def test_pending_semantic_review_blocks_production(self) -> None:
        config = copy.deepcopy(configured_run())
        config["mode"] = "production"
        # Test the gate itself, not whatever the shipped config happens to waive.
        config.pop("waived_gates", None)
        approval_path = (
            CONTRASTIVE_ROOT
            / "prompts"
            / "pre_generation_semantic_approval.json"
        ).resolve()
        lock_path = (
            CONTRASTIVE_ROOT / "prompts" / "pre_generation.lock.json"
        ).resolve()
        real_load_json = preflight.load_json

        def pending_load_json(path: Path) -> Any:
            value = real_load_json(path)
            if path == approval_path:
                return {
                    **value,
                    "status": "pending",
                    "researcher": None,
                    "native_japanese_reviewer": None,
                    "approved_at": None,
                }
            if path == lock_path:
                return {**value, "semantic_equivalence_review": "pending"}
            return value

        with patch.object(
            preflight,
            "load_json",
            side_effect=pending_load_json,
        ):
            with self.assertRaisesRegex(
                ContrastiveError,
                "semantic equivalence",
            ):
                build_pre_generation_plan(config, "production_blocked")

    def test_waived_gate_does_not_block_production(self) -> None:
        # Waiving one gate must not silently waive another: a gate that is still
        # armed has to keep blocking production. Which gates the shipped config
        # happens to waive is policy and is pinned separately by
        # test_shipped_waiver_policy_is_explicit, so this test removes the
        # calibration waiver rather than assuming it is absent.
        #
        # Removing it matters for more than tidiness. When this test assumed the
        # gate was armed and it was not, build_pre_generation_plan stopped
        # raising and instead ran to completion, writing a run directory and
        # artifacts as a side effect. That leaked state into a later module and
        # surfaced as an unrelated error under `unittest discover` that did not
        # reproduce when this module was run alone. A test asserting that
        # something raises should be constructed so it cannot silently succeed.
        config = copy.deepcopy(configured_run())
        config["mode"] = "production"
        config["waived_gates"].pop("heldout_comprehension", None)
        self.assertIn("semantic_approval", config["waived_gates"])
        with self.assertRaisesRegex(
            ContrastiveError,
            "passed heldout_comprehension calibration",
        ):
            build_pre_generation_plan(config, "production_waived_fallthrough")

    def test_shipped_waiver_policy_is_explicit(self) -> None:
        # The set of waived gates is a scientific decision, not a detail. Each
        # entry costs something the writeup has to declare, so a change here
        # must be deliberate and must come with a recorded rationale.
        #   semantic_approval / frame_only -- recorded 2026-08-01
        #   heldout_comprehension          -- recorded 2026-08-04, route 2 of
        #     docs/PATH_TO_GENERATION.md; see docs/GATE_DECISION_2026_08_04.md.
        #     Cost: a NULL probe result is uninterpretable for the affected
        #     stratum, and the preregistration must say so.
        config = configured_run()
        self.assertEqual(
            set(config["waived_gates"]),
            {"semantic_approval", "frame_only", "heldout_comprehension"},
        )
        for gate, waiver in config["waived_gates"].items():
            self.assertEqual(
                set(waiver),
                {"waived_by", "waived_at", "rationale"},
                f"waiver {gate} must carry exactly who, when and why",
            )
            self.assertTrue(str(waiver["rationale"]).strip())

    def test_waivers_are_recorded_in_report_and_notifications(self) -> None:
        config = apply_pre_generation_overrides(
            configured_run(),
            topics=["t001"],
            content_specs=["content_p01"],
            languages=["en"],
        )
        artifacts = pre_generation_module._assemble_pre_generation(
            config,
            "waiver_record",
        )
        report = artifacts.report
        waived = report["global_gates"]["C24_waived_gates"]
        # The report must mirror the config exactly: no waiver dropped from the
        # record, and none invented that the config did not authorise.
        self.assertEqual(set(waived), set(config["waived_gates"]))
        for gate in waived:
            self.assertEqual(
                set(waived[gate]),
                {"waived_by", "waived_at", "rationale"},
            )
        # A waived gate is never reported as passed.
        self.assertEqual(
            report["global_gates"]["C23_calibration_evidence"]["frame_only"][
                "status"
            ],
            "pending",
        )
        self.assertIn(
            "waiver",
            report["global_gates"]["C23_calibration_evidence"]["frame_only"],
        )
        codes = {row["code"] for row in artifacts.notifications}
        self.assertIn("frame_only_calibration_waived", codes)
        self.assertIn("semantic_equivalence_review_waived", codes)
        self.assertNotIn("frame_only_calibration_pending", codes)

    def test_gate_waiver_rejects_bad_input(self) -> None:
        config = copy.deepcopy(configured_run())
        with self.assertRaisesRegex(ContrastiveError, "not waivable"):
            preflight.gate_waiver(config, "japanese_surface")
        config["waived_gates"]["frame_only"]["waived_at"] = "2026-08-01"
        with self.assertRaisesRegex(ContrastiveError, "UTC offset"):
            preflight.gate_waiver(config, "frame_only")
        config["waived_gates"]["frame_only"]["waived_at"] = "not-a-timestamp"
        with self.assertRaisesRegex(ContrastiveError, "ISO-8601"):
            preflight.gate_waiver(config, "frame_only")

    def test_unwaived_config_still_blocks_production(self) -> None:
        config = copy.deepcopy(configured_run())
        config["mode"] = "production"
        config.pop("waived_gates", None)
        with self.assertRaisesRegex(
            ContrastiveError,
            "semantic equivalence",
        ):
            build_pre_generation_plan(config, "production_unwaived")

    def test_incomplete_grid_is_only_allowed_in_development(self) -> None:
        config = apply_pre_generation_overrides(
            configured_run(),
            topics=["t001"],
            content_specs=["content_p01"],
            languages=["en"],
        )
        plan, jobs, _ = build_pre_generation_plan(
            config, "development_slice"
        )
        self.assertEqual(len(jobs), 1)
        self.assertEqual(plan["preflight"]["status"], "passed_with_notices")
        config["mode"] = "production"
        with self.assertRaises(ContrastiveError):
            build_pre_generation_plan(config, "production_slice")

    def test_provider_validation_rejects_nested_placeholder_and_secret(self) -> None:
        profile = load_provider_profile("tests/fixtures/provider.yaml")
        placeholder = copy.deepcopy(profile)
        placeholder["nested"] = {"route": "https://<HOST>/v1"}
        with self.assertRaisesRegex(ContrastiveError, "placeholder"):
            validate_pre_generation_provider(placeholder)
        secret = copy.deepcopy(profile)
        secret["api_key"] = "literal-secret"
        with self.assertRaisesRegex(ContrastiveError, "literal secret"):
            validate_pre_generation_provider(secret)
        missing_reference = copy.deepcopy(profile)
        missing_reference["provider"] = "openai_compatible"
        missing_reference["structured_output_mode"] = "json_schema"
        missing_reference["auth"] = "api_key"
        with self.assertRaisesRegex(ContrastiveError, "api_key_env"):
            validate_pre_generation_provider(missing_reference)
        userinfo = copy.deepcopy(profile)
        userinfo["base_url"] = (
            "https://user:literal-secret@example.com/v1"
        )
        with self.assertRaisesRegex(ContrastiveError, "URL userinfo"):
            validate_pre_generation_provider(userinfo)
        query_secret = copy.deepcopy(profile)
        query_secret["base_url"] = (
            "https://example.com/v1?api_key=literal-secret"
        )
        with self.assertRaisesRegex(
            ContrastiveError, "sensitive query parameters"
        ):
            validate_pre_generation_provider(query_secret)
        route_secret = copy.deepcopy(profile)
        route_secret["route"] = "structured?token=literal-secret"
        with self.assertRaisesRegex(
            ContrastiveError, "sensitive query parameters"
        ):
            validate_pre_generation_provider(route_secret)
        hidden_sampling_retry = copy.deepcopy(profile)
        hidden_sampling_retry["malformed_response_attempts"] = 2
        with self.assertRaisesRegex(ContrastiveError, "unsupported fields"):
            validate_pre_generation_provider(hidden_sampling_retry)
        mutable_project_route = copy.deepcopy(profile)
        mutable_project_route["project_env"] = "PROJECT_ID"
        with self.assertRaisesRegex(ContrastiveError, "unsupported fields"):
            validate_pre_generation_provider(mutable_project_route)
        invalid_endpoint = copy.deepcopy(profile)
        invalid_endpoint["base_url"] = "not-an-absolute-url"
        with self.assertRaisesRegex(ContrastiveError, "absolute HTTP"):
            validate_pre_generation_provider(invalid_endpoint)
        insecure_endpoint = copy.deepcopy(profile)
        insecure_endpoint.update({
            "provider": "openai_compatible",
            "structured_output_mode": "json_schema",
            "auth": "api_key",
            "api_key_env": "PROVIDER_KEY",
            "base_url": "http://example.com/v1",
            "route": "structured",
        })
        with self.assertRaisesRegex(ContrastiveError, "must use HTTPS"):
            validate_pre_generation_provider(insecure_endpoint)
        prompt_only = copy.deepcopy(insecure_endpoint)
        prompt_only["base_url"] = "https://example.com/v1"
        prompt_only["structured_output_mode"] = "prompt_only"
        with self.assertRaisesRegex(
            ContrastiveError,
            "native JSON-Schema enforcement",
        ):
            validate_pre_generation_provider(prompt_only)
        mutable_revision = copy.deepcopy(profile)
        mutable_revision["model_revision"] = "latest"
        with self.assertRaisesRegex(ContrastiveError, "mutable alias"):
            validate_pre_generation_provider(mutable_revision)
        google_without_adc_reference = copy.deepcopy(profile)
        google_without_adc_reference.update({
            "provider": "google",
            "auth": "auto",
            "project": "frozen-project",
            "location": "us-central1",
        })
        with self.assertRaisesRegex(ContrastiveError, "credential_env"):
            validate_pre_generation_provider(google_without_adc_reference)
        fake_adc_reference = copy.deepcopy(google_without_adc_reference)
        fake_adc_reference["credential_env"] = "PATH"
        with self.assertRaisesRegex(
            ContrastiveError, "GOOGLE_APPLICATION_CREDENTIALS"
        ):
            validate_pre_generation_provider(fake_adc_reference)
        header_injection = copy.deepcopy(profile)
        header_injection["route"] = "v1\r\nCookie: injected"
        with self.assertRaisesRegex(ContrastiveError, "control character"):
            validate_pre_generation_provider(header_injection)
        prefixed_secret = copy.deepcopy(profile)
        prefixed_secret["model_revision"] = "sk-or-v1-abcdefghijklmnop"
        with self.assertRaisesRegex(ContrastiveError, "secret-like"):
            validate_pre_generation_provider(prefixed_secret)
        invalid_port = copy.deepcopy(profile)
        invalid_port["base_url"] = "https://example.com:99999/v1"
        with self.assertRaisesRegex(ContrastiveError, "invalid base_url port"):
            validate_pre_generation_provider(invalid_port)
        ambiguous_google = copy.deepcopy(google_without_adc_reference)
        ambiguous_google["credential_env"] = "GOOGLE_APPLICATION_CREDENTIALS"
        ambiguous_google["base_url"] = "https://example.com/v1"
        with self.assertRaisesRegex(
            ContrastiveError, "incompatible routing/auth fields"
        ):
            validate_pre_generation_provider(ambiguous_google)
        absolute_route = copy.deepcopy(insecure_endpoint)
        absolute_route["base_url"] = "https://trusted.example/v1"
        absolute_route["route"] = "http://evil.example/v1"
        with self.assertRaisesRegex(ContrastiveError, "safe relative path"):
            validate_pre_generation_provider(absolute_route)

    def test_non_finite_numbers_are_rejected(self) -> None:
        for field, value in (
            ("temperature", float("nan")),
            ("temperature", float("inf")),
            ("top_p", float("-inf")),
        ):
            config = apply_pre_generation_overrides(
                configured_run(),
                topics=["t001"],
                content_specs=["content_p01"],
                languages=["en"],
            )
            config["generation"][field] = value
            with self.subTest(field=field, value=value):
                with self.assertRaisesRegex(
                    ContrastiveError, "finite number"
                ):
                    build_pre_generation_plan(config, "non_finite")
        provider = load_provider_profile("tests/fixtures/provider.yaml")
        provider["input_usd_per_million"] = float("nan")
        with self.assertRaisesRegex(ContrastiveError, "finite number"):
            validate_pre_generation_provider(provider)

    def test_run_id_validation_rejects_path_like_values(self) -> None:
        for value in (
            ".",
            "..",
            "../escape",
            "nested/run",
            "nested\\run",
            "",
            "run.",
            "CON",
            "lpt1.txt",
        ):
            with self.subTest(value=value):
                with self.assertRaises(ContrastiveError):
                    validate_run_id(value)

    def test_mutable_paths_cannot_overlap_sources(self) -> None:
        config = configured_run()
        bad_runs = copy.deepcopy(config)
        bad_runs["paths"]["runs"] = "."
        with self.assertRaisesRegex(ContrastiveError, "paths.runs"):
            validate_path_layout(bad_runs)
        bad_output = copy.deepcopy(config)
        bad_output["paths"]["output"] = "prompts"
        with self.assertRaisesRegex(ContrastiveError, "paths.output"):
            validate_path_layout(bad_output)

    def test_context_overflow_fails_before_manifest(self) -> None:
        config = apply_pre_generation_overrides(
            configured_run(),
            topics=["t001"],
            content_specs=["content_p01"],
            languages=["en"],
        )
        config["provider_profile"]["capabilities"]["max_context_tokens"] = 100
        with self.assertRaisesRegex(ContrastiveError, "exceeds provider context"):
            build_pre_generation_plan(config, "context_failure")

    def test_generation_and_provider_relational_limits(self) -> None:
        config = apply_pre_generation_overrides(
            configured_run(),
            topics=["t001"],
            content_specs=["content_p01"],
            languages=["en"],
        )
        config["generation"]["estimated_output_tokens"] = (
            config["generation"]["max_output_tokens"] + 1
        )
        with self.assertRaisesRegex(
            ContrastiveError, "estimated_output_tokens"
        ):
            build_pre_generation_plan(config, "output_estimate_failure")

        workers = apply_pre_generation_overrides(
            configured_run(),
            topics=["t001"],
            content_specs=["content_p01"],
            languages=["en"],
        )
        workers["generation"]["max_workers"] = 2
        with self.assertRaisesRegex(
            ContrastiveError, "max_workers"
        ):
            build_pre_generation_plan(workers, "worker_failure")

        budget = apply_pre_generation_overrides(
            configured_run(),
            topics=["t001"],
            content_specs=["content_p01"],
            languages=["en"],
        )
        budget["budget"]["max_usd"] = 0
        with self.assertRaisesRegex(ContrastiveError, "authorized budget"):
            build_pre_generation_plan(budget, "budget_failure")

        sub_micro = apply_pre_generation_overrides(
            configured_run(),
            topics=["t001"],
            content_specs=["content_p01"],
            languages=["en"],
        )
        sub_micro["provider_profile"]["input_usd_per_million"] = 1e-12
        sub_micro["provider_profile"]["output_usd_per_million"] = 1e-12
        sub_micro["budget"]["max_usd"] = 0
        with self.assertRaisesRegex(ContrastiveError, "authorized budget"):
            build_pre_generation_plan(sub_micro, "sub_micro_budget_failure")

    def test_production_requires_one_sample_and_seed_support(self) -> None:
        config = configured_run()
        config["mode"] = "production"
        config["grid"]["samples_per_combination"] = 2
        with self.assertRaisesRegex(
            ContrastiveError, "exactly one sample"
        ):
            _production_grid_gate(
                config,
                selected_topic_count=40,
                selected_spec_count=10,
            )
        config["grid"]["samples_per_combination"] = 1
        config["provider_profile"]["capabilities"]["seed"] = "none"
        with self.assertRaisesRegex(ContrastiveError, "seed"):
            build_pre_generation_plan(config, "seed_failure")

    def test_hash_lock_detects_mutation(self) -> None:
        import contrastive_generation.preflight as preflight_module

        original = preflight_module.sha256_file

        def changed(path: Path) -> str:
            if path.name == "content_generation_en.txt":
                return "0" * 64
            return original(path)

        with patch.object(preflight_module, "sha256_file", side_effect=changed):
            with self.assertRaisesRegex(ContrastiveError, "Locked.*changed"):
                validate_scaffolding_bundle(configured_run())
        with patch.object(
            preflight_module,
            "request_projection_sha256",
            return_value="0" * 64,
        ):
            with self.assertRaisesRegex(
                ContrastiveError, "request-projection code"
            ):
                validate_scaffolding_bundle(configured_run())

    def test_lock_covers_every_pipeline_module_and_runtime_lock(self) -> None:
        package_root = (
            CONTRASTIVE_ROOT / "src" / "contrastive_generation"
        )
        package_modules = {
            path.relative_to(CONTRASTIVE_ROOT).as_posix()
            for path in package_root.rglob("*.py")
        }
        self.assertTrue(package_modules)
        self.assertTrue(package_modules <= LOCKED_FILES)
        self.assertTrue(
            {"pyproject.toml", "uv.lock"} <= REQUEST_PROJECTION_FILES
        )

    def test_write_verify_idempotence_and_request_tamper_detection(self) -> None:
        run_id = "test_" + uuid.uuid4().hex
        run_dir = CONTRASTIVE_ROOT / "runs" / run_id
        output_dir = CONTRASTIVE_ROOT / "data" / "pilot" / run_id
        config = apply_pre_generation_overrides(
            configured_run(),
            topics=["t001"],
            content_specs=["content_p01"],
        )
        try:
            first_dir, first_plan, first_jobs = write_pre_generation_plan(
                config, run_id
            )
            self.assertEqual(first_dir, run_dir)
            verified = verify_pre_generation_run(run_dir)
            self.assertEqual(verified["generation_jobs"], 2)
            self.assertEqual(verified["requests"], 6)
            import contrastive_generation.pre_generation as generation_module

            different_runtime = copy.deepcopy(
                first_plan["runtime_environment"]
            )
            different_runtime["python"]["version"] = "0.0.0"
            with patch.object(
                generation_module,
                "_runtime_environment",
                return_value=different_runtime,
            ):
                with self.assertRaisesRegex(
                    ContrastiveError, "Runtime environment differs"
                ):
                    verify_pre_generation_run(run_dir)
            handoff = prepare_generation_handoff(run_dir)
            self.assertFalse(
                hasattr(handoff, "request_for_submission")
            )
            with self.assertRaisesRegex(
                ContrastiveError, "Unknown preplanned generation attempt"
            ):
                handoff.submission_for(first_jobs[-1]["job_id"], 99)
            with self.assertRaisesRegex(
                ContrastiveError, "released in order"
            ):
                handoff.submission_for(first_jobs[-1]["job_id"], 1)
            submission = handoff.submission_for(
                first_jobs[-1]["job_id"], 0
            )
            exact_request = submission["request"]
            self.assertEqual(
                set(exact_request),
                {"system", "user", "response_schema", "generation"},
            )
            self.assertEqual(submission["attempt_index"], 0)
            self.assertEqual(
                submission["provider_execution_sha256"],
                first_jobs[-1]["provider_execution_sha256"],
            )
            self.assertNotIn("api_key", submission["adapter_profile"])
            with self.assertRaisesRegex(
                ContrastiveError, "implicit replay is forbidden"
            ):
                handoff.submission_for(first_jobs[-1]["job_id"], 0)
            prior_candidate = CONTRASTIVE_ROOT / (
                first_jobs[-1]["generation_attempts"][0][
                    "candidate_output_path"
                ]
            )
            prior_candidate.parent.mkdir(parents=True, exist_ok=True)
            prior_candidate.write_text(
                '{"status":"rejected"}\n', encoding="utf-8"
            )
            collision = CONTRASTIVE_ROOT / (
                first_jobs[-1]["generation_attempts"][1][
                    "candidate_output_path"
                ]
            )
            collision.parent.mkdir(parents=True, exist_ok=True)
            collision.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(
                ContrastiveError, "output already exists"
            ):
                handoff.submission_for(
                    first_jobs[-1]["job_id"], 1
                )
            collision.unlink()
            _, second_plan, second_jobs = write_pre_generation_plan(
                config, run_id
            )
            self.assertEqual(first_plan, second_plan)
            self.assertEqual(first_jobs, second_jobs)

            plan_path = run_dir / "plan.json"
            original_plan_bytes = plan_path.read_bytes()
            original_plan_text = original_plan_bytes.decode("utf-8")
            changed_plan = json.loads(original_plan_text)
            changed_plan["created_at"] = "2000-01-01T00:00:00+00:00"
            plan_path.write_text(
                json.dumps(
                    changed_plan,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ContrastiveError, "plan hash is invalid"
            ):
                verify_pre_generation_run(run_dir)
            plan_path.write_bytes(original_plan_bytes)

            request_path = CONTRASTIVE_ROOT / (
                first_jobs[-1]["generation_attempts"][1]["request_path"]
            )
            original_request_text = request_path.read_text(encoding="utf-8")
            request_path.write_text(
                original_request_text + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                ContrastiveError, "Request changed after handoff"
            ):
                handoff.submission_for(
                    first_jobs[-1]["job_id"], 1
                )
            request_path.write_text(
                original_request_text, encoding="utf-8"
            )
            request = json.loads(original_request_text)
            request["user"] += "\ntampered"
            request_path.write_text(
                json.dumps(request, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ContrastiveError, "Stored request was modified"
            ):
                verify_pre_generation_run(run_dir)
            with self.assertRaisesRegex(
                ContrastiveError, "Request changed after handoff"
            ):
                handoff.submission_for(
                    first_jobs[-1]["job_id"], 1
                )
        finally:
            if run_dir.is_dir():
                resolved = run_dir.resolve()
                resolved.relative_to((CONTRASTIVE_ROOT / "runs").resolve())
                shutil.rmtree(resolved)
            if output_dir.is_dir():
                resolved_output = output_dir.resolve()
                resolved_output.relative_to(
                    (CONTRASTIVE_ROOT / "data" / "pilot").resolve()
                )
                shutil.rmtree(resolved_output)

    def test_concurrent_divergent_writers_have_one_owner(self) -> None:
        import contrastive_generation.pre_generation as generation_module

        run_id = "test_" + uuid.uuid4().hex
        run_dir = CONTRASTIVE_ROOT / "runs" / run_id
        first = apply_pre_generation_overrides(
            configured_run(),
            topics=["t001"],
            content_specs=["content_p01"],
            languages=["en"],
        )
        second = copy.deepcopy(first)
        second["provider_profile"]["model_revision"] = "test-revision-2"
        configs = {"first": first, "second": second}
        barrier = threading.Barrier(2)
        original_assemble = generation_module._assemble_pre_generation

        def synchronized_assemble(
            config: dict,
            selected_run_id: str,
        ):
            artifacts = original_assemble(config, selected_run_id)
            barrier.wait(timeout=30)
            return artifacts

        def write(label: str) -> tuple[str, str, object]:
            try:
                _, plan, _ = write_pre_generation_plan(
                    configs[label], run_id
                )
                return label, "ok", plan
            except ContrastiveError as exc:
                return label, "error", exc

        try:
            with patch.object(
                generation_module,
                "_assemble_pre_generation",
                side_effect=synchronized_assemble,
            ):
                with ThreadPoolExecutor(max_workers=2) as executor:
                    results = list(executor.map(write, ("first", "second")))
            successes = [row for row in results if row[1] == "ok"]
            failures = [row for row in results if row[1] == "error"]
            self.assertEqual(len(successes), 1, results)
            self.assertEqual(len(failures), 1, results)
            self.assertRegex(
                str(failures[0][2]),
                "already being created|different immutable plan",
            )
            verified = verify_pre_generation_run(run_dir)
            self.assertEqual(verified["generation_jobs"], 1)
            winner = successes[0][0]
            stored = json.loads(
                (run_dir / "resolved_config.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                stored["provider_profile"]["model_revision"],
                configs[winner]["provider_profile"]["model_revision"],
            )
            _, repeated, _ = write_pre_generation_plan(
                configs[winner], run_id
            )
            self.assertEqual(
                repeated["plan_identity_sha256"],
                successes[0][2]["plan_identity_sha256"],
            )
        finally:
            if run_dir.is_dir():
                resolved = run_dir.resolve()
                resolved.relative_to(
                    (CONTRASTIVE_ROOT / "runs").resolve()
                )
                shutil.rmtree(resolved)

    def test_handoff_requires_referenced_development_credential(self) -> None:
        run_id = "test_" + uuid.uuid4().hex
        run_dir = CONTRASTIVE_ROOT / "runs" / run_id
        config = apply_pre_generation_overrides(
            configured_run(),
            topics=["t001"],
            content_specs=["content_p01"],
            languages=["en"],
        )
        variable = "CONTRASTIVE_MISSING_" + uuid.uuid4().hex.upper()
        self.assertNotIn(variable, os.environ)
        config["provider_profile"].update({
            "provider": "openai_compatible",
            "structured_output_mode": "json_schema",
            "auth": "api_key",
            "api_key_env": variable,
            "base_url": "https://generation.invalid/v1",
            "route": "structured",
        })
        try:
            write_pre_generation_plan(config, run_id)
            with self.assertRaisesRegex(
                ContrastiveError, "requires the referenced provider credential"
            ):
                prepare_generation_handoff(run_dir)
        finally:
            if run_dir.is_dir():
                resolved = run_dir.resolve()
                resolved.relative_to((CONTRASTIVE_ROOT / "runs").resolve())
                shutil.rmtree(resolved)

    def test_handoff_detects_control_plane_tamper(self) -> None:
        run_id = "test_" + uuid.uuid4().hex
        run_dir = CONTRASTIVE_ROOT / "runs" / run_id
        config = apply_pre_generation_overrides(
            configured_run(),
            topics=["t001"],
            content_specs=["content_p01"],
            languages=["en"],
        )
        try:
            _, _, jobs = write_pre_generation_plan(config, run_id)
            job_id = jobs[0]["job_id"]
            config_path = run_dir / "resolved_config.json"
            original_config = config_path.read_text(encoding="utf-8")
            handoff = prepare_generation_handoff(run_dir)
            config_path.write_text(
                original_config + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                ContrastiveError, "configuration changed"
            ):
                handoff.submission_for(job_id, 0)
            config_path.write_text(original_config, encoding="utf-8")

            manifest_path = run_dir / "manifest.jsonl"
            original_manifest = manifest_path.read_text(encoding="utf-8")
            handoff = prepare_generation_handoff(run_dir)
            manifest_path.write_text(
                original_manifest + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                ContrastiveError, "manifest changed"
            ):
                handoff.submission_for(job_id, 0)
            manifest_path.write_text(
                original_manifest, encoding="utf-8"
            )

            plan_path = run_dir / "plan.json"
            original_plan = plan_path.read_text(encoding="utf-8")
            handoff = prepare_generation_handoff(run_dir)
            plan_path.write_text(original_plan + "\n", encoding="utf-8")
            with self.assertRaisesRegex(
                ContrastiveError, "plan changed"
            ):
                handoff.submission_for(job_id, 0)
        finally:
            if run_dir.is_dir():
                resolved = run_dir.resolve()
                resolved.relative_to((CONTRASTIVE_ROOT / "runs").resolve())
                shutil.rmtree(resolved)


if __name__ == "__main__":
    unittest.main()
