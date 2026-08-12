from __future__ import annotations

import json
import shutil
import sys
import unittest
import uuid
from pathlib import Path


CONTRASTIVE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CONTRASTIVE_ROOT / "src"))

from contrastive_generation.calibration import (
    validate_frame_calibration_results,
    write_frame_calibration_packet,
)
from contrastive_generation.utils import ContrastiveError


class FrameCalibrationTests(unittest.TestCase):
    def setUp(self) -> None:
        suffix = uuid.uuid4().hex
        self.packet = (
            CONTRASTIVE_ROOT / "data" / "calibration" / f"test_packet_{suffix}"
        )
        self.results = (
            CONTRASTIVE_ROOT / "data" / "calibration" / f"test_results_{suffix}.jsonl"
        )
        self.raters = (
            CONTRASTIVE_ROOT / "data" / "calibration" / f"test_raters_{suffix}.json"
        )

    def tearDown(self) -> None:
        if self.packet.is_dir():
            shutil.rmtree(self.packet)
        if self.results.is_file():
            self.results.unlink()
        if self.raters.is_file():
            self.raters.unlink()

    @staticmethod
    def _read(path: Path) -> list[dict]:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def _write_passing_results(self) -> None:
        key = self._read(self.packet / "answer_key.jsonl")
        correctness: dict[str, bool] = {}
        for language in ("en", "ja"):
            for explicitness, correct_per_class in (
                ("implicit", {"eval": 8, "deploy": 7}),
                ("explicit", {"eval": 9, "deploy": 9}),
            ):
                for class_name in ("eval", "deploy"):
                    rows = sorted(
                        (
                            row
                            for row in key
                            if row["language"] == language
                            and row["explicitness"] == explicitness
                            and row["class"] == class_name
                        ),
                        key=lambda row: row["blind_item_id"],
                    )
                    for index, row in enumerate(rows):
                        correctness[row["blind_item_id"]] = (
                            index < correct_per_class[class_name]
                        )
        output = []
        for row in key:
            correct = correctness[row["blind_item_id"]]
            prediction = (
                row["class"]
                if correct
                else ("deploy" if row["class"] == "eval" else "eval")
            )
            for rater_number in range(5):
                output.append(
                    {
                        "blind_item_id": row["blind_item_id"],
                        "rater_id": f"r{rater_number + 1:02d}",
                        "prediction": prediction,
                    }
                )
        self.results.parent.mkdir(parents=True, exist_ok=True)
        self.results.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in output
            ),
            encoding="utf-8",
        )
        self._write_raters()

    def _write_raters(self) -> None:
        self.raters.parent.mkdir(parents=True, exist_ok=True)
        self.raters.write_text(
            json.dumps(
                {
                    "cohort_id": "confirmatory_test_cohort",
                    "confirmatory_cohort": True,
                    "independent_from_frame_development_attestation": True,
                    "answer_key_not_disclosed_attestation": True,
                    "paired_counterparts_not_shown_attestation": True,
                    "raters": [
                        {
                            "rater_id": f"r{number:02d}",
                            "qualified_languages": ["en", "ja"],
                            "new_confirmatory_rater_attestation": True,
                        }
                        for number in range(1, 6)
                    ],
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def test_packet_is_blinded_and_passing_results_are_gated(self) -> None:
        result = write_frame_calibration_packet(self.packet)
        self.assertEqual(result["items"], 80)
        rating_items = self._read(self.packet / "rating_items.jsonl")
        self.assertEqual(len(rating_items), 80)
        self.assertTrue(
            all(
                set(row) == {"blind_item_id", "language", "text"}
                for row in rating_items
            )
        )
        self._write_passing_results()
        report = validate_frame_calibration_results(
            self.packet,
            self.results,
            self.raters,
        )
        self.assertEqual(report["status"], "passed")
        for language in ("en", "ja"):
            implicit = report["languages"][language]["strata"]["implicit"]
            explicit = report["languages"][language]["strata"]["explicit"]
            self.assertEqual(implicit["balanced_accuracy"], 0.75)
            self.assertEqual(explicit["balanced_accuracy"], 0.9)

    def test_implicit_frames_that_are_too_explicit_fail(self) -> None:
        write_frame_calibration_packet(self.packet)
        key = self._read(self.packet / "answer_key.jsonl")
        output = [
            {
                "blind_item_id": row["blind_item_id"],
                "rater_id": f"r{rater_number + 1:02d}",
                "prediction": row["class"],
            }
            for row in key
            for rater_number in range(5)
        ]
        self.results.parent.mkdir(parents=True, exist_ok=True)
        self.results.write_text(
            "".join(json.dumps(row) + "\n" for row in output),
            encoding="utf-8",
        )
        self._write_raters()
        report = validate_frame_calibration_results(
            self.packet,
            self.results,
            self.raters,
        )
        self.assertEqual(report["status"], "failed")
        self.assertFalse(
            report["languages"]["en"]["gates"]["implicit_max"]
        )

    def test_nonconfirmatory_rater_metadata_is_rejected(self) -> None:
        write_frame_calibration_packet(self.packet)
        self._write_passing_results()
        metadata = json.loads(self.raters.read_text(encoding="utf-8"))
        metadata["independent_from_frame_development_attestation"] = False
        self.raters.write_text(
            json.dumps(metadata, sort_keys=True),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            ContrastiveError,
            "confirmatory/blinding attestations",
        ):
            validate_frame_calibration_results(
                self.packet,
                self.results,
                self.raters,
            )


if __name__ == "__main__":
    unittest.main()
