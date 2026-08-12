from __future__ import annotations

import copy
import shutil
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


CONTRASTIVE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CONTRASTIVE_ROOT / "src"))

from contrastive_generation.catalog import (
    FROZEN_PAIR_MANIFEST,
    _validate_content_spec_reuse,
    _validate_cue_locus,
    _validate_frame_lengths,
    _validate_frozen_manifest,
    _validate_pair_contract,
    _validate_plain_body,
    _validate_purpose_frame,
    cue_spans,
    load_catalog,
)
from contrastive_generation.rendering import prerender_template
from contrastive_generation.utils import ContrastiveError, sha256_file


class CatalogTests(unittest.TestCase):
    def _catalog_copy(self, root: Path) -> Path:
        catalog = load_catalog()
        index = root / "containers.yaml"
        shutil.copy2(catalog.container_path, index)
        shutil.copy2(catalog.purpose_frame_path, root / "purpose_frames.yaml")
        shutil.copytree(catalog.container_dir, root / "containers")
        return index

    def _records(self):
        return tuple(copy.deepcopy(row) for row in load_catalog().containers)

    @staticmethod
    def _pair(records, pair_id):
        return [row for row in records if row.data["pair_id"] == pair_id]

    def test_catalog_preserves_source_provenance_and_manifest(self) -> None:
        catalog = load_catalog()
        self.assertEqual(len(catalog.topics), 40)
        self.assertEqual(len(catalog.containers), 40)
        self.assertEqual(
            Counter(topic.domain for topic in catalog.topics),
            {
                "technical_work": 5,
                "data_and_numbers": 5,
                "language_and_text": 5,
                "household": 5,
                "admin_and_legal": 5,
                "workplace": 5,
                "knowledge": 5,
                "personal_logistics": 5,
            },
        )
        self.assertEqual(
            Counter(container.data["class"] for container in catalog.containers),
            {"eval": 20, "deploy": 20},
        )
        self.assertEqual(
            {row.data["pair_id"] for row in catalog.containers},
            set(FROZEN_PAIR_MANIFEST),
        )
        self.assertEqual(
            Counter(
                row.data["purpose_frame"]["family_id"]
                for row in catalog.containers
            ),
            {f"ff0{number}": 8 for number in range(1, 6)},
        )
        for language_catalog in (catalog, load_catalog("topics.ja.yaml")):
            frame_texts = [
                row.data["purpose_frame"]["text"]
                for row in language_catalog.containers
            ]
            self.assertEqual(len(frame_texts), len(set(frame_texts)))
        self.assertEqual(
            sha256_file(catalog.legacy_topic_path),
            catalog.topic_data["legacy_source"]["sha256"],
        )

    def test_container_files_and_identifiers_are_exact(self) -> None:
        catalog = load_catalog()
        self.assertEqual(
            [container_id for container_id, _ in catalog.container_paths],
            [f"c{number:02d}" for number in range(1, 41)],
        )
        for container_id, path in catalog.container_paths:
            self.assertEqual(path.stem, container_id)
            self.assertEqual(path.parent, catalog.container_dir)
            self.assertTrue(path.is_file())

    def test_content_specs_feed_two_pairs_and_four_containers(self) -> None:
        catalog = load_catalog()
        for number in range(1, 11):
            spec_id = f"content_p{number:02d}"
            records = [
                row for row in catalog.containers
                if row.data["content_spec_id"] == spec_id
            ]
            self.assertEqual(len(records), 4)
            self.assertEqual(len({row.data["pair_id"] for row in records}), 2)
            self.assertEqual(
                len(
                    {
                        row.data["purpose_frame"]["family_id"]
                        for row in records
                    }
                ),
                2,
            )

    def test_catalog_hash_covers_every_container_file(self) -> None:
        catalog = load_catalog()
        baseline = catalog.container_catalog_sha256()
        with tempfile.TemporaryDirectory(dir=CONTRASTIVE_ROOT) as temporary:
            root = Path(temporary)
            index = self._catalog_copy(root)
            self.assertEqual(
                load_catalog("topics.yaml", index).container_catalog_sha256(),
                baseline,
            )
            target = root / "containers" / "c01.yaml"
            target.write_text(
                target.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            self.assertNotEqual(
                load_catalog("topics.yaml", index).container_catalog_sha256(),
                baseline,
            )

    def test_unbalanced_pair_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=CONTRASTIVE_ROOT) as temporary:
            index = self._catalog_copy(Path(temporary))
            (index.parent / "containers" / "c05.yaml").unlink()
            with self.assertRaisesRegex(
                ContrastiveError,
                "pair p03 must contain exactly two",
            ):
                load_catalog("topics.yaml", index)

    def test_container_filename_must_match_identifier(self) -> None:
        with tempfile.TemporaryDirectory(dir=CONTRASTIVE_ROOT) as temporary:
            root = Path(temporary)
            index = self._catalog_copy(root)
            (root / "containers" / "c01.yaml").rename(
                root / "containers" / "c99.yaml"
            )
            with self.assertRaises(ContrastiveError):
                load_catalog("topics.yaml", index)

    def test_renderer_inserts_only_the_purpose_frame(self) -> None:
        eval_record, deploy_record = self._pair(
            load_catalog().containers, "p01"
        )
        eval_text, eval_number = prerender_template(eval_record, 12345)
        deploy_text, deploy_number = prerender_template(deploy_record, 67890)
        self.assertIsNone(eval_number)
        self.assertIsNone(deploy_number)
        self.assertNotIn("{{PURPOSE_FRAME}}", eval_text)
        self.assertEqual(
            eval_text.replace(
                eval_record.data["purpose_frame"]["text"],
                "{{PURPOSE_FRAME}}",
                1,
            ),
            deploy_text.replace(
                deploy_record.data["purpose_frame"]["text"],
                "{{PURPOSE_FRAME}}",
                1,
            ),
        )

    def test_rejects_changed_twin_body(self) -> None:
        records = self._records()
        self._pair(records, "p01")[0].data["body_template"] += " "
        with self.assertRaisesRegex(ContrastiveError, "body_template"):
            _validate_pair_contract(records)

    def test_rejects_changed_twin_nuisance_profile(self) -> None:
        records = self._records()
        self._pair(records, "p01")[0].data["nuisance_profile"][
            "speaker_present"
        ] = True
        with self.assertRaisesRegex(ContrastiveError, "nuisance_profile"):
            _validate_pair_contract(records)

    def test_rejects_changed_twin_explicitness_or_surface(self) -> None:
        for field, value in (
            ("explicitness", "explicit"),
            ("surface", "institutional"),
        ):
            with self.subTest(field=field):
                records = self._records()
                self._pair(records, "p01")[0].data["purpose_frame"][field] = value
                with self.assertRaisesRegex(ContrastiveError, field):
                    _validate_pair_contract(records)

    def test_rejects_empty_or_identical_frames(self) -> None:
        records = self._records()
        pair = self._pair(records, "p01")
        pair[0].data["purpose_frame"]["text"] = ""
        with self.assertRaisesRegex(ContrastiveError, "empty purpose frame"):
            _validate_pair_contract(records)
        records = self._records()
        pair = self._pair(records, "p01")
        pair[1].data["purpose_frame"]["text"] = pair[0].data[
            "purpose_frame"
        ]["text"]
        with self.assertRaisesRegex(ContrastiveError, "identical"):
            _validate_pair_contract(records)

    def test_rejects_attribution_inconsistent_with_class(self) -> None:
        records = self._records()
        record = self._pair(records, "p01")[0]
        record.data["purpose_frame"]["attribution"] = "deployment"
        with self.assertRaisesRegex(ContrastiveError, "inconsistent"):
            _validate_purpose_frame(record, "en")
        with self.assertRaisesRegex(ContrastiveError, "inconsistent"):
            _validate_pair_contract(records)

    def test_rejects_missing_or_duplicated_purpose_slot(self) -> None:
        for replacement in ("", "{{PURPOSE_FRAME}}\n{{PURPOSE_FRAME}}"):
            with self.subTest(replacement=replacement):
                records = self._records()
                pair = self._pair(records, "p01")
                for record in pair:
                    record.data["body_template"] = record.data[
                        "body_template"
                    ].replace("{{PURPOSE_FRAME}}", replacement)
                with self.assertRaisesRegex(ContrastiveError, "exactly once"):
                    _validate_pair_contract(records)

    def test_rejects_invalid_output_constraint_nesting(self) -> None:
        record = self._records()[0]
        record.data["nuisance_profile"]["constraint_kind"] = "bare_answer"
        with self.assertRaisesRegex(ContrastiveError, "nesting"):
            _validate_plain_body(record)

    def test_rejects_numbered_or_xml_matched_scaffolding(self) -> None:
        for addition in ("\nItem 3 of 40", "\n<question>"):
            with self.subTest(addition=addition):
                record = self._records()[0]
                record.data["body_template"] += addition
                with self.assertRaisesRegex(
                    ContrastiveError, "numbered or XML"
                ):
                    _validate_plain_body(record)

    def test_rejects_single_container_content_spec_incompatibility(self) -> None:
        records = self._records()
        records[0].data["model_guidance"] += " Changed."
        with self.assertRaisesRegex(ContrastiveError, "incompatible"):
            _validate_content_spec_reuse(records)

    def test_matched_mutation_survives_pair_but_fails_manifest(self) -> None:
        records = self._records()
        for record in self._pair(records, "p01"):
            record.data["nuisance_profile"]["speaker_present"] = True
        _validate_pair_contract(records)
        with self.assertRaisesRegex(ContrastiveError, "frozen pair manifest"):
            _validate_frozen_manifest(records)

    def test_rejects_content_spec_assigned_to_other_than_two_pairs(self) -> None:
        records = self._records()
        for record in self._pair(records, "p20"):
            record.data["content_spec_id"] = "content_p09"
        with self.assertRaisesRegex(ContrastiveError, "exactly two pairs"):
            _validate_content_spec_reuse(records)

    def test_rejects_frame_length_violation(self) -> None:
        records = self._records()
        self._pair(records, "p01")[0].data["purpose_frame"]["text"] += (
            " This deliberately makes the frame much longer"
        )
        with self.assertRaisesRegex(ContrastiveError, "frame-length gate"):
            _validate_frame_lengths(records, "en")

    def _frames(self):
        return copy.deepcopy(load_catalog().purpose_frame_data)

    def test_cue_spans_are_morphology_free(self) -> None:
        self.assertEqual(
            cue_spans(
                "A reviewer will assess this response.",
                "A recipient will apply this response.",
                "en",
            ),
            (("reviewer", "assess"), ("recipient", "apply")),
        )
        self.assertEqual(
            cue_spans(
                "第三者が完成した回答を精査します。",
                "第三者が完成した回答を用います。",
                "ja",
            ),
            (("精査し",), ("用い",)),
        )

    def test_rejects_recurring_implicit_cue_span(self) -> None:
        """The 2.1.0 defect: one verb separating every implicit pair of a class.

        Japanese deploy frames all ended in 活用します, so a single token was a
        perfect class indicator that survived every leave-one-frame-family-out
        fold. The cap makes that state unrepresentable.
        """
        data = self._frames()
        # Leave the declared balance and every span count valid, so only the
        # frequency cap can reject: three verb_only pairs sharing one cue verb.
        for pair_id in ("p02", "p04", "p10"):
            row = next(r for r in data["pairs"] if r["pair_id"] == pair_id)
            self.assertEqual(row["cue_locus"], "verb_only")
            eval_text = row["ja"]["eval"]
            row["ja"]["deploy"] = eval_text.replace("確認", "活用").replace(
                "点検", "活用"
            ).replace("精査", "活用")
        with self.assertRaisesRegex(ContrastiveError, "cue span '活用'"):
            _validate_cue_locus(self._records(), data, "ja")

    def test_rejects_locus_inconsistent_with_the_text(self) -> None:
        data = self._frames()
        pair = next(row for row in data["pairs"] if row["pair_id"] == "p02")
        self.assertEqual(pair["cue_locus"], "verb_only")
        pair["en"]["deploy"] = "The delivery team will apply this response."
        with self.assertRaisesRegex(ContrastiveError, "must vary\n?\\s*in exactly one"):
            _validate_cue_locus(self._records(), data, "en")

    def test_rejects_locus_confounded_with_frame_family(self) -> None:
        data = self._frames()
        for pair_id, locus in (("p01", "verb_only"), ("p03", "verb_only")):
            row = next(r for r in data["pairs"] if r["pair_id"] == pair_id)
            row["cue_locus"] = locus
        with self.assertRaisesRegex(ContrastiveError, "not balanced"):
            _validate_cue_locus(self._records(), data, "en")

    def test_declared_locus_is_shared_by_both_languages(self) -> None:
        """cue_locus is declared per pair, so EN and JA cannot diverge."""
        data = self._frames()
        for row in data["pairs"]:
            self.assertIn("cue_locus", row)
            self.assertNotIn("cue_locus", row["en"])
            self.assertNotIn("cue_locus", row["ja"])


if __name__ == "__main__":
    unittest.main()
