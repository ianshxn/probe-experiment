from __future__ import annotations

import copy
import json
import shutil
import sys
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch


CONTRASTIVE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CONTRASTIVE_ROOT / "src"))

from contrastive_generation.allocation import generator_for, verifier_for
from contrastive_generation.lease import LeaseHeld, RunLease
from contrastive_generation.generation_cost import estimate_generation_cost
from contrastive_generation.planning import load_provider_profile
from contrastive_generation.pre_generation import (
    apply_pre_generation_overrides,
    load_pre_generation_config,
    with_pre_generation_provider,
    write_pre_generation_plan,
)
from contrastive_generation.publication import (
    PublicationConflict,
    atomic_create_text,
    render_item,
)
from contrastive_generation.providers import (
    MalformedProviderResponse,
    MockProvider,
    ProviderTransportError,
)
from contrastive_generation.runner import GenerationRunner, validate_twin_runs
from contrastive_generation.utils import (
    ContrastiveError,
    hash_object,
    sha256_file,
)
from contrastive_generation.verification import (
    _comparison_consistent,
    deterministic_qc,
    validate_verification_surfaces,
    verification_schema,
)


class AllocationAndPublicationTests(unittest.TestCase):
    def test_allocation_uses_only_permanent_identity(self) -> None:
        self.assertEqual(generator_for("t001", "content_p01"), "generator_a")
        self.assertEqual(verifier_for("t001", "content_p01"), "generator_b")
        self.assertEqual(generator_for("t002", "content_p01"), "generator_b")
        for _language in ("en", "ja"):
            for _sample_index in range(4):
                self.assertEqual(
                    generator_for("t019", "content_p06"),
                    "generator_b",
                )

    def test_render_item_requires_exact_slot_keys(self) -> None:
        self.assertEqual(
            render_item("Dear {{NAME}}, {{ACTION}}", {
                "NAME": "Rin",
                "ACTION": "please confirm.",
            }),
            "Dear Rin, please confirm.",
        )
        with self.assertRaisesRegex(ContrastiveError, "missing"):
            render_item("{{A}} {{B}}", {"A": "one"})
        with self.assertRaisesRegex(ContrastiveError, "extra"):
            render_item("{{A}}", {"A": "one", "B": "two"})

    def test_atomic_create_does_not_clobber(self) -> None:
        root = CONTRASTIVE_ROOT / "data" / f"test_publication_{uuid.uuid4().hex}"
        path = root / "value.txt"
        try:
            atomic_create_text(path, "first\n")
            with self.assertRaises(PublicationConflict):
                atomic_create_text(path, "second\n")
            self.assertEqual(path.read_text(encoding="utf-8"), "first\n")
        finally:
            if root.is_dir():
                shutil.rmtree(root)


class LeaseTests(unittest.TestCase):
    def test_second_lease_is_rejected_until_release(self) -> None:
        root = CONTRASTIVE_ROOT / "runs" / f"test_lease_{uuid.uuid4().hex}"
        root.mkdir(parents=True)
        first = RunLease(root, root, timeout_seconds=30)
        second = RunLease(root, root, timeout_seconds=30)
        try:
            first.acquire()
            with self.assertRaises(LeaseHeld):
                second.acquire()
            first.release()
            second.acquire()
            second.release()
        finally:
            if root.is_dir():
                shutil.rmtree(root)


class StartupValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        suffix = uuid.uuid4().hex
        self.first = CONTRASTIVE_ROOT / "runs" / f"test_twin_a_{suffix}"
        self.second = CONTRASTIVE_ROOT / "runs" / f"test_twin_b_{suffix}"
        self.first.mkdir(parents=True)
        self.second.mkdir(parents=True)
        self.config_a = {
            "mode": "development",
            "randomization": {"seed": 1},
            "provider_profile": {
                "profile_id": "mock_a",
                "provider": "mock",
            },
        }
        self.config_b = copy.deepcopy(self.config_a)
        self.config_b["provider_profile"]["profile_id"] = "mock_b"
        self.row = {
            "topic_id": "t001",
            "content_spec_id": "content_p01",
            "language": "en",
            "sample_index": 0,
        }

    def tearDown(self) -> None:
        for root in (self.first, self.second):
            if root.is_dir():
                shutil.rmtree(root)

    @staticmethod
    def _write(path: Path, value: object) -> None:
        path.write_text(
            json.dumps(value, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def _write_pair(self, rows_b: list[dict] | None = None) -> None:
        self._write(self.first / "resolved_config.json", self.config_a)
        self._write(self.second / "resolved_config.json", self.config_b)
        self._write(self.first / "manifest.jsonl", self.row)
        (self.second / "manifest.jsonl").write_text(
            "".join(
                json.dumps(row) + "\n"
                for row in ([self.row] if rows_b is None else rows_b)
            ),
            encoding="utf-8",
        )

    def test_rejects_non_provider_configuration_difference(self) -> None:
        self.config_b["randomization"]["seed"] = 2
        self._write_pair()
        with patch(
            "contrastive_generation.runner.verify_pre_generation_run",
            return_value={},
        ):
            with self.assertRaisesRegex(ContrastiveError, "not grid twins"):
                validate_twin_runs(self.first, self.second)

    def test_rejects_nonidentical_manifest_identity_sets(self) -> None:
        self._write_pair(rows_b=[])
        with patch(
            "contrastive_generation.runner.verify_pre_generation_run",
            return_value={},
        ):
            with self.assertRaisesRegex(ContrastiveError, "identity grids"):
                validate_twin_runs(self.first, self.second)


class DeterministicJapaneseSurfaceTests(unittest.TestCase):
    POLICY = {
        "min_japanese_chars": 20,
        "max_latin_share": 0.80,
    }
    CUE_LEXICON = {
        "languages": {
            "en": [],
            "ja": [],
        },
        # Generated content is screened against the narrow scope.
        "generated_content": {
            "en": [],
            "ja": [],
        },
    }

    def test_japanese_verification_schema_uses_localized_property_names(
        self,
    ) -> None:
        validate_verification_surfaces()
        for stage in ("extraction", "comparison"):
            schema = verification_schema(stage, "ja")
            self.assertTrue(schema["properties"])
            self.assertTrue(
                all(
                    not any("A" <= char <= "z" for char in name)
                    for name in schema["properties"]
                )
            )

    def test_verification_pass_includes_requirement_and_quality_gates(
        self,
    ) -> None:
        extraction = {
            "is_answerable_from_content": True,
            "independently_selected_option": 1,
            "leaked_answer_signal": False,
            "matches_authoring_requirements": False,
            "register_appropriate": True,
            "distractors_plausible": True,
            "distractor_notes": "",
            "register_notes": "",
        }
        comparison = {
            "pass": False,
            "option_match": True,
            "answerable": True,
            "no_leak": True,
            "matches_authoring_requirements": False,
            "register_appropriate": True,
            "distractors_plausible": True,
            "issues": ["requirements mismatch"],
        }
        self.assertTrue(
            _comparison_consistent(comparison, extraction, 1)
        )
        comparison["pass"] = True
        self.assertFalse(
            _comparison_consistent(comparison, extraction, 1)
        )

    @staticmethod
    def _schema(*slots: str) -> dict:
        return {
            "type": "object",
            "required": ["各項目の内容", "解答記号"],
            "properties": {
                "各項目の内容": {
                    "type": "object",
                    "required": list(slots),
                    "properties": {
                        slot: {"type": "string"} for slot in slots
                    },
                    "additionalProperties": False,
                },
                "解答記号": {"type": "string"},
            },
            "additionalProperties": False,
        }

    def _qc(self, values: dict[str, str]) -> dict:
        response = json.dumps(
            {"各項目の内容": values, "解答記号": "ア"},
            ensure_ascii=False,
        )
        qc, parsed = deterministic_qc(
            response,
            self._schema(*values),
            "ja",
            self.CUE_LEXICON,
            japanese_surface=self.POLICY,
        )
        self.assertIsNotNone(parsed)
        return qc

    def test_natural_japanese_with_pr_passes(self) -> None:
        qc = self._qc({
            "内容": (
                "対象のPRは#123です。変更内容を確認して、"
                "問題点と改善方法を具体的に説明してください。"
            ),
        })
        self.assertTrue(qc["checks"]["japanese_surface"])
        self.assertIn("PR", qc["checks"]["latin_tokens"])

    def test_natural_japanese_with_ascii_unit_passes(self) -> None:
        qc = self._qc({
            "内容": (
                "薄力粉は200g必要です。分量を人数に合わせて計算し、"
                "残りの材料と一緒に分かりやすく記載してください。"
            ),
        })
        self.assertTrue(qc["checks"]["japanese_surface"])
        self.assertIn("g", qc["checks"]["latin_tokens"])

    def test_code_slot_and_separate_japanese_framing_pass(self) -> None:
        qc = self._qc({
            "コード": (
                "def load_data(path):\n"
                "    try:\n"
                "        return open(path).read()\n"
                "    except OSError:\n"
                "        return None\n"
            ),
            "依頼": (
                "このコードの例外処理と戻り値を確認し、"
                "利用者に起こり得る問題と改善案を日本語で説明してください。"
            ),
        })
        self.assertTrue(qc["checks"]["japanese_surface"])
        self.assertGreater(qc["checks"]["latin_share"], 0.5)
        self.assertLess(qc["checks"]["latin_share"], 0.8)

    def test_all_english_fails_both_language_conditions(self) -> None:
        qc = self._qc({
            "内容": (
                "Please review this pull request and explain every important "
                "problem before the code is merged."
            ),
        })
        self.assertFalse(qc["checks"]["japanese_surface"])
        self.assertTrue(any(
            "insufficient Japanese script" in issue
            for issue in qc["issues"]
        ))
        self.assertTrue(any(
            "ASCII-Latin share exceeds" in issue
            for issue in qc["issues"]
        ))

    def test_latin_share_inclusive_boundary(self) -> None:
        passing = self._qc({"内容": "日" * 20 + "A" * 80})
        failing = self._qc({"内容": "日" * 20 + "A" * 81})
        self.assertEqual(passing["checks"]["latin_share"], 0.8)
        self.assertTrue(passing["checks"]["japanese_surface"])
        self.assertGreater(failing["checks"]["latin_share"], 0.8)
        self.assertFalse(failing["checks"]["japanese_surface"])

    def test_zero_denominator_records_zero_and_fails_floor(self) -> None:
        qc = self._qc({"内容": "1234 !!!"})
        self.assertEqual(qc["checks"]["latin_share"], 0.0)
        self.assertEqual(qc["checks"]["japanese_script_chars"], 0)
        self.assertFalse(qc["checks"]["japanese_surface"])

    def test_fullwidth_digits_fail_but_ascii_digits_pass(self) -> None:
        japanese = "必要な材料の数量を確認して一覧にまとめてください"
        ascii_qc = self._qc({"内容": japanese + " 123"})
        fullwidth_qc = self._qc({"内容": japanese + " １２３"})
        self.assertTrue(ascii_qc["checks"]["japanese_surface"])
        self.assertFalse(fullwidth_qc["checks"]["japanese_surface"])
        self.assertEqual(
            fullwidth_qc["checks"]["fullwidth_digit_hits"],
            ["１", "２", "３"],
        )
        self.assertTrue(any(
            "contains full-width digits" in issue
            for issue in fullwidth_qc["issues"]
        ))

    def test_latin_tokens_are_sorted_unique_whole_tokens(self) -> None:
        qc = self._qc({
            "内容": (
                "対象のPRとSHAを確認し、必要な水の量をmlで記載して、"
                "最後にPRの変更点を日本語で説明してください。"
            ),
        })
        self.assertEqual(qc["checks"]["latin_tokens"], ["PR", "SHA", "ml"])
        self.assertEqual(
            qc["checks"]["japanese_surface_policy"],
            self.POLICY,
        )


class GenerationConfigSchemaTests(unittest.TestCase):
    @staticmethod
    def _config() -> dict:
        return {
            "schema_version": 1,
            "stage": "contrastive_generation_runner",
            "run_dir_a": "runs/not_reached_a",
            "run_dir_b": "runs/not_reached_b",
            "production": False,
            "max_workers": 1,
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

    def _assert_invalid(self, config: dict) -> None:
        with self.assertRaisesRegex(
            ContrastiveError,
            "Invalid generation configuration",
        ):
            GenerationRunner(config)

    def test_japanese_surface_block_is_required(self) -> None:
        config = self._config()
        del config["verification"]["japanese_surface"]
        self._assert_invalid(config)

    def test_japanese_surface_fields_are_required(self) -> None:
        for field in ("min_japanese_chars", "max_latin_share"):
            with self.subTest(field=field):
                config = self._config()
                del config["verification"]["japanese_surface"][field]
                self._assert_invalid(config)

    def test_japanese_surface_bounds_and_extra_fields(self) -> None:
        invalid_values = (
            ("min_japanese_chars", -1),
            ("max_latin_share", -0.01),
            ("max_latin_share", 1.01),
        )
        for field, value in invalid_values:
            with self.subTest(field=field, value=value):
                config = self._config()
                config["verification"]["japanese_surface"][field] = value
                self._assert_invalid(config)
        config = self._config()
        config["verification"]["japanese_surface"]["unexpected"] = True
        self._assert_invalid(config)


class RunnerIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        suffix = uuid.uuid4().hex
        self.run_id_a = f"test_generation_a_{suffix}"
        self.run_id_b = f"test_generation_b_{suffix}"
        self.run_dir_a = CONTRASTIVE_ROOT / "runs" / self.run_id_a
        self.run_dir_b = CONTRASTIVE_ROOT / "runs" / self.run_id_b
        self.output_a = CONTRASTIVE_ROOT / "data" / "pilot" / self.run_id_a
        self.output_b = CONTRASTIVE_ROOT / "data" / "pilot" / self.run_id_b
        profile_a = load_provider_profile("tests/fixtures/provider.yaml")
        profile_b = copy.deepcopy(profile_a)
        profile_b["profile_id"] = "offline_test_b"
        profile_b["model"] = "deterministic-test-model-b"
        profile_b["model_revision"] = "test-revision-2"
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
        self.runner_config = {
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

    def tearDown(self) -> None:
        for root in (
            self.run_dir_a,
            self.run_dir_b,
            self.output_a,
            self.output_b,
        ):
            if root.is_dir():
                shutil.rmtree(root)

    def test_mock_pair_generates_renders_and_resumes(self) -> None:
        twin = validate_twin_runs(self.run_dir_a, self.run_dir_b)
        self.assertEqual(len(twin.identities), 2)
        first = GenerationRunner(self.runner_config)
        result = first.run()
        self.assertEqual(result["counts"], {"accepted": 2})
        self.assertEqual(
            result["budget_authorization"]["status"],
            "passed",
        )
        self.assertEqual(
            result["cost_estimate"][
                "verification_context_authorization"
            ]["status"],
            "passed",
        )
        self.assertEqual(
            sum(
                result["cost_estimate"]["baseline"][
                    "provider_calls"
                ].values()
            ),
            10,
        )
        policy_path = self.run_dir_a / "generation_policy.json"
        self.assertTrue(policy_path.is_file())
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        self.assertEqual(
            result["generation_policy"],
            f"runs/{self.run_id_a}/generation_policy.json",
        )
        self.assertEqual(
            result["generation_policy_sha256"],
            hash_object(policy),
        )
        states = sorted((self.run_dir_a / "state").glob("*.json"))
        for state_path in states:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(
                len(state["attempts"][0]["verification"]["pair_results"]),
                2,
            )
            self.assertTrue(
                state["attempts"][0]["verification"][
                    "purpose_frame_masked"
                ]
            )
        self.assertEqual(len(states), 2)
        recorded_policies = []
        for path in states:
            state = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "accepted")
            self.assertEqual(len(state["derived_outputs"]), 4)
            self.assertTrue(Path(state["content_output_path"]).is_file())
            self.assertEqual(
                sha256_file(Path(state["content_output_path"])),
                state["content_sha256"],
            )
            for derived in state["derived_outputs"]:
                self.assertTrue(Path(derived["output_path"]).is_file())
                self.assertEqual(
                    sha256_file(Path(derived["output_path"])),
                    derived["output_sha256"],
                )
            for attempt in state["attempts"]:
                applied = attempt["deterministic_qc"]["checks"].get(
                    "japanese_surface_policy"
                )
                if applied is not None:
                    recorded_policies.append(applied)
        self.assertEqual(
            recorded_policies,
            [self.runner_config["verification"]["japanese_surface"]],
        )
        manifest_before = (
            self.run_dir_a / "stimulus_manifest.jsonl"
        ).read_bytes()
        result_before = (
            self.run_dir_a / "generation_result.json"
        ).read_bytes()
        second = GenerationRunner(self.runner_config)
        rerun = second.run()
        self.assertEqual(rerun["counts"], {"accepted": 2})
        self.assertEqual(
            (self.run_dir_a / "stimulus_manifest.jsonl").read_bytes(),
            manifest_before,
        )
        self.assertEqual(
            (self.run_dir_a / "generation_result.json").read_bytes(),
            result_before,
        )
        self.assertTrue(
            all(not provider.calls for provider in second.providers.values())
        )

    def test_all_in_budget_blocks_before_provider_construction(self) -> None:
        changed = copy.deepcopy(self.runner_config)
        changed["budget"]["max_usd"] = 0
        providers: list[MockProvider] = []

        def factory(name: str, profile: dict) -> MockProvider:
            provider = MockProvider(name, profile)
            providers.append(provider)
            return provider

        with self.assertRaisesRegex(
            ContrastiveError,
            "All-in generation cost",
        ):
            GenerationRunner(changed, provider_factory=factory)
        self.assertEqual(providers, [])

    def test_verifier_context_overflow_blocks_cost_authorization(self) -> None:
        twin = validate_twin_runs(self.run_dir_a, self.run_dir_b)
        twin.config_a["provider_profile"]["capabilities"][
            "max_context_tokens"
        ] = 1
        twin.config_b["provider_profile"]["capabilities"][
            "max_context_tokens"
        ] = 1
        with self.assertRaisesRegex(
            ContrastiveError,
            "Verifier context upper bound",
        ):
            estimate_generation_cost(twin, self.runner_config)

    def test_resume_rejects_changed_generation_policy_before_calls(self) -> None:
        GenerationRunner(self.runner_config).run()
        changed = copy.deepcopy(self.runner_config)
        changed["verification"]["japanese_surface"]["max_latin_share"] = 0.79
        providers: list[MockProvider] = []

        def factory(name: str, profile: dict) -> MockProvider:
            provider = MockProvider(name, profile)
            providers.append(provider)
            return provider

        with self.assertRaisesRegex(ContrastiveError, "policy differs"):
            GenerationRunner(changed, provider_factory=factory).run()
        self.assertTrue(providers)
        self.assertTrue(all(not provider.calls for provider in providers))

    def test_resume_rejects_missing_policy_with_existing_state(self) -> None:
        GenerationRunner(self.runner_config).run()
        (self.run_dir_a / "generation_policy.json").unlink()
        providers: list[MockProvider] = []

        def factory(name: str, profile: dict) -> MockProvider:
            provider = MockProvider(name, profile)
            providers.append(provider)
            return provider

        with self.assertRaisesRegex(ContrastiveError, "without a bound"):
            GenerationRunner(
                self.runner_config,
                provider_factory=factory,
            ).run()
        self.assertTrue(providers)
        self.assertTrue(all(not provider.calls for provider in providers))

    def test_preexisting_planned_output_raises_publication_conflict(self) -> None:
        job = json.loads(
            (self.run_dir_a / "manifest.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()[0]
        )
        target = CONTRASTIVE_ROOT / job["content_output_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('{"foreign": true}\n', encoding="utf-8")
        with self.assertRaises(PublicationConflict):
            GenerationRunner(self.runner_config).run()

    def test_malformed_response_consumes_attempt_without_in_place_retry(self) -> None:
        providers: dict[str, MockProvider] = {}

        def factory(name: str, profile: dict) -> MockProvider:
            configured = dict(profile)
            if name == "generator_a":
                configured["text"] = "not valid json"
            provider = MockProvider(name, configured)
            providers[name] = provider
            return provider

        result = GenerationRunner(
            self.runner_config,
            provider_factory=factory,
        ).run()
        self.assertEqual(result["counts"], {"exhausted": 2})
        states = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in (self.run_dir_a / "state").glob("*.json")
        ]
        self.assertEqual(len(states), 2)
        for state in states:
            self.assertEqual(state["status"], "exhausted")
            self.assertEqual(len(state["attempts"]), 3)
            self.assertEqual(
                [attempt["outcome"] for attempt in state["attempts"]],
                ["malformed", "malformed", "malformed"],
            )
        self.assertEqual(len(providers["generator_a"].calls), 6)
        self.assertEqual(len(providers["generator_b"].calls), 0)

    def test_verifier_transport_failure_preserves_author_candidate_for_resume(
        self,
    ) -> None:
        self.runner_config["max_workers"] = 1
        verification_calls = {"count": 0}

        class OneVerificationTransportFailure(MockProvider):
            def generate(self, *args, response_schema, **kwargs):
                schema_id = str(response_schema.get("$id", ""))
                if "verification_" in schema_id:
                    verification_calls["count"] += 1
                    if verification_calls["count"] == 3:
                        raise ProviderTransportError(
                            "synthetic verifier transport outage"
                        )
                return super().generate(
                    *args,
                    response_schema=response_schema,
                    **kwargs,
                )

        with self.assertRaisesRegex(
            ProviderTransportError,
            "synthetic verifier transport outage",
        ):
            GenerationRunner(
                self.runner_config,
                provider_factory=(
                    lambda name, profile: OneVerificationTransportFailure(
                        name,
                        profile,
                    )
                ),
            ).run()

        pending_states = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in (self.run_dir_a / "state").glob("*.json")
            if json.loads(path.read_text(encoding="utf-8"))[
                "pending_verification"
            ]
            is not None
        ]
        self.assertEqual(len(pending_states), 1)
        pending = pending_states[0]["pending_verification"]
        self.assertEqual(pending["attempt_index"], 0)
        self.assertEqual(pending_states[0]["attempts"], [])
        self.assertEqual(len(pending["completed_verifications"]), 1)
        candidate_hash = pending["candidate_sha256"]
        candidate_path = Path(pending["candidate_path"])
        self.assertEqual(sha256_file(candidate_path), candidate_hash)

        result = GenerationRunner(self.runner_config).run()
        self.assertEqual(result["counts"], {"accepted": 2})
        resumed = json.loads(
            (
                self.run_dir_a
                / "state"
                / f"{pending_states[0]['job_id']}.json"
            ).read_text(encoding="utf-8")
        )
        self.assertIsNone(resumed["pending_verification"])
        self.assertEqual(resumed["selected_attempt_index"], 0)
        self.assertEqual(
            resumed["attempts"][0]["candidate_sha256"],
            candidate_hash,
        )
        self.assertEqual(
            sha256_file(Path(resumed["attempts"][0]["candidate_path"])),
            candidate_hash,
        )

    def test_circuit_breaker_allows_half_open_recovery(self) -> None:
        class TwiceFailingProvider(MockProvider):
            def __init__(self, name, config):
                super().__init__(name, config)
                self.failures_remaining = 2

            def generate(self, *args, **kwargs):
                if self.failures_remaining:
                    self.failures_remaining -= 1
                    raise ProviderTransportError("synthetic circuit failure")
                return super().generate(*args, **kwargs)

        runner = GenerationRunner(
            self.runner_config,
            provider_factory=TwiceFailingProvider,
        )
        provider = runner.providers["generator_a"]
        job = json.loads(
            (self.run_dir_a / "manifest.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()[0]
        )
        attempt = job["generation_attempts"][0]
        request = json.loads(
            (CONTRASTIVE_ROOT / attempt["request_path"]).read_text(
                encoding="utf-8"
            )
        )
        runner._current_job.job = job
        runner._current_job.attempt_index = 0

        def call():
            return runner._call_provider(
                provider,
                request["system"],
                request["user"],
                request["generation"]["temperature"],
                request["generation"]["top_p"],
                request["generation"]["max_output_tokens"],
                request["generation"]["seed"],
                request["response_schema"],
                "circuit_test",
            )

        with self.assertRaises(ProviderTransportError):
            call()
        with self.assertRaises(ProviderTransportError):
            call()
        response, _ = call()
        self.assertTrue(response.text)
        self.assertFalse(runner._circuits)

    def test_completed_malformed_response_breaks_transport_failure_streak(
        self,
    ) -> None:
        class AlternatingProvider(MockProvider):
            def __init__(self, name, config):
                super().__init__(name, config)
                self.events = ["transport", "malformed", "transport"]

            def generate(self, *args, **kwargs):
                event = self.events.pop(0)
                if event == "transport":
                    raise ProviderTransportError("synthetic transport")
                raise MalformedProviderResponse("synthetic completed response")

        runner = GenerationRunner(
            self.runner_config,
            provider_factory=AlternatingProvider,
        )
        provider = runner.providers["generator_a"]
        job = json.loads(
            (self.run_dir_a / "manifest.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()[0]
        )
        attempt = job["generation_attempts"][0]
        request = json.loads(
            (CONTRASTIVE_ROOT / attempt["request_path"]).read_text(
                encoding="utf-8"
            )
        )
        runner._current_job.job = job
        runner._current_job.attempt_index = 0

        def call():
            return runner._call_provider(
                provider,
                request["system"],
                request["user"],
                request["generation"]["temperature"],
                request["generation"]["top_p"],
                request["generation"]["max_output_tokens"],
                request["generation"]["seed"],
                request["response_schema"],
                "streak_test",
            )

        with self.assertRaises(ProviderTransportError):
            call()
        with self.assertRaises(MalformedProviderResponse):
            call()
        with self.assertRaises(ProviderTransportError):
            call()
        self.assertFalse(runner._circuits)
