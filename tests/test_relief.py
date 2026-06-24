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

    def test_simple_taxonomy_parser_handles_project_taxonomy(self) -> None:
        text = (Path(__file__).resolve().parents[1] / "taxonomy.yaml").read_text(encoding="utf-8")
        data = relieflens.parse_simple_taxonomy(text)
        self.assertEqual(data["categories"][0]["id"], "active_danger")
        self.assertEqual(data["categories"][0]["severity"], "critical")
        self.assertTrue(data["categories"][0]["prompts"])

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

    def test_score_images_rejects_mismatched_dimensions(self) -> None:
        categories = [relieflens.Category("danger", "Danger", "critical", "review", ["danger"])]
        with self.assertRaises(ValueError):
            relieflens.score_images(
                [Path("a.jpg")],
                np.array([[1.0, 0.0]], dtype=np.float32),
                ["danger"],
                np.array([[1.0]], dtype=np.float32),
                categories,
            )

    def test_mobileclip2_s0_uses_apple_documented_preprocess_kwargs(self) -> None:
        self.assertEqual(
            relieflens.MobileClipScorer._model_kwargs("MobileCLIP2-S0"),
            {"image_mean": (0, 0, 0), "image_std": (1, 1, 1)},
        )

    def test_image_source_prefers_relative_paths(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            image = out / "demo_images" / "sample.jpg"
            image.parent.mkdir()
            image.write_bytes(b"fake")
            self.assertEqual(relieflens.image_source(image, out), "demo_images/sample.jpg")

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
