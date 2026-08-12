from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path


CONTRASTIVE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CONTRASTIVE_ROOT / "src"))

from contrastive_generation.catalog import load_catalog
from contrastive_generation.utils import ContrastiveError


class TopicLocalizationTests(unittest.TestCase):
    def _copy_pair(self, root: Path) -> tuple[Path, Path]:
        en_path = root / "topics.yaml"
        ja_path = root / "topics.ja.yaml"
        shutil.copy2(CONTRASTIVE_ROOT / "topics.yaml", en_path)
        shutil.copy2(CONTRASTIVE_ROOT / "topics.ja.yaml", ja_path)
        return en_path, ja_path

    def test_japanese_view_preserves_machine_identity(self) -> None:
        english = load_catalog()
        japanese = load_catalog("topics.ja.yaml")
        self.assertEqual(
            [
                (topic.topic_id, topic.topic_label, topic.domain)
                for topic in japanese.topics
            ],
            [
                (topic.topic_id, topic.topic_label, topic.domain)
                for topic in english.topics
            ],
        )
        self.assertEqual(japanese.topics[0].topic_display_name, "コードレビュー")
        self.assertEqual(japanese.topics[0].domain_display_name, "技術業務")

    def test_english_edit_invalidates_japanese_hash_pin(self) -> None:
        with tempfile.TemporaryDirectory(dir=CONTRASTIVE_ROOT) as temporary:
            en_path, _ = self._copy_pair(Path(temporary))
            text = en_path.read_text(encoding="utf-8").replace(
                "topic_display_name: Code review",
                "topic_display_name: Reviewing code",
                1,
            )
            en_path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(
                ContrastiveError,
                "current canonical English SHA-256",
            ):
                load_catalog(en_path)

    def test_japanese_view_cannot_change_machine_label(self) -> None:
        with tempfile.TemporaryDirectory(dir=CONTRASTIVE_ROOT) as temporary:
            en_path, ja_path = self._copy_pair(Path(temporary))
            text = ja_path.read_text(encoding="utf-8").replace(
                "topic_label: code_review",
                "topic_label: code_reviews",
                1,
            )
            ja_path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(
                ContrastiveError,
                r"topics\[0\]\.topic_label",
            ):
                load_catalog(en_path)


if __name__ == "__main__":
    unittest.main()
