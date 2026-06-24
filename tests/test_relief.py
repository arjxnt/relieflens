from __future__ import annotations

import csv
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import relieflens


class ReliefLensTests(unittest.TestCase):
    def test_taxonomy_loads_without_pyyaml(self) -> None:
        categories = relieflens.load_taxonomy(Path(__file__).resolve().parents[1] / "taxonomy.yaml")
        self.assertGreaterEqual(len(categories), 8)
        self.assertEqual(categories[0].id, "active_danger")
        self.assertEqual(categories[0].severity, "critical")
        self.assertTrue(categories[0].prompts)

    def test_score_images_returns_ranked_matches(self) -> None:
        categories = [
            relieflens.Category("danger", "Danger", "critical", "review", ["danger"]),
            relieflens.Category("routine", "Routine", "low", "defer", ["routine"]),
        ]
        results = relieflens.score_images(
            [Path("a.jpg")],
            np.array([[1.0, 0.0]], dtype=np.float32),
            ["danger", "routine"],
            np.eye(2, dtype=np.float32),
            categories,
        )
        self.assertEqual(results[0].top_id, "danger")
        self.assertEqual(results[0].severity, "critical")
        self.assertEqual(results[0].top_matches[0]["label"], "Danger")

    def test_demo_writes_review_artifacts(self) -> None:
        with TemporaryDirectory() as tmp:
            args = type(
                "Args",
                (),
                {
                    "out": tmp,
                    "taxonomy": str(Path(__file__).resolve().parents[1] / "taxonomy.yaml"),
                },
            )()
            relieflens.run_demo(args)
            out = Path(tmp)
            self.assertTrue((out / "dashboard.html").exists())
            self.assertTrue((out / "triage.jsonl").exists())
            with (out / "triage.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3)
            self.assertIn("Flood or water damage", {row["top_label"] for row in rows})


if __name__ == "__main__":
    unittest.main()
