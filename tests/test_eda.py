"""Checks for the descriptive EDA of the ranking matrices."""
import sys
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party"))

from eda.exploratory_analysis import (
    EXPECTED,
    assess_outliers,
    column_centered_pca,
    load_matrices,
    require_processed,
    summarize_datasets,
)


class MissingInputTests(unittest.TestCase):
    def test_missing_processed_file_fails(self):
        with self.assertRaises(SystemExit):
            require_processed(
                {
                    "rosmap": str(ROOT / "data" / "processed" / "does_not_exist.csv"),
                    "gse120584": str(ROOT / "data" / "processed" / "GSE120584_serum_norm.csv"),
                    "gse215789": str(ROOT / "data" / "processed" / "GSE215789_plasma_count.csv"),
                }
            )


class PcaContractTests(unittest.TestCase):
    def test_svd_matches_skill_renderer(self):
        from bioinformatics_visual_evidence_atlas.visualization.figures import _pca

        rng = np.random.default_rng(42)
        matrix = rng.normal(size=(12, 5))
        _scores, explained = column_centered_pca(matrix)
        rows = []
        for sample in range(matrix.shape[0]):
            for feature in range(matrix.shape[1]):
                rows.append(
                    {
                        "sample": f"s{sample}",
                        "feature": f"f{feature}",
                        "value": float(matrix[sample, feature]),
                        "group": "Control" if sample < 6 else "AD",
                    }
                )
        _fig, audit = _pca(rows, "toy")
        import matplotlib.pyplot as plt

        plt.close(_fig)
        self.assertAlmostEqual(audit["variance_pc1"], float(explained[0]), places=12)
        self.assertAlmostEqual(audit["variance_pc2"], float(explained[1]), places=12)

    def test_outlier_screen_does_not_drop_rows(self):
        scores = np.column_stack([np.linspace(-1, 1, 30), np.linspace(-1, 1, 30)])
        fitted = {
            "toy": {
                "scores": scores,
                "sample_id": np.array([f"s{i}" for i in range(30)]),
                "diagnosis": np.array(["AD"] * 15 + ["Control"] * 15),
            }
        }
        from eda.exploratory_analysis import DATASET_ORDER

        original = list(DATASET_ORDER)
        DATASET_ORDER[:] = ["toy"]
        try:
            table = assess_outliers(fitted)
        finally:
            DATASET_ORDER[:] = original
        self.assertEqual(int(table.loc[table["row_type"] == "dataset", "n_samples"].iloc[0]), 30)
        self.assertTrue((table["exclusion_recommended"] == "no").all())


class RankingMatrixTests(unittest.TestCase):
    def setUp(self):
        self.paths = {
            "rosmap": str(ROOT / "data" / "processed" / "ROSMAP_brain_norm.csv"),
            "gse120584": str(ROOT / "data" / "processed" / "GSE120584_serum_norm.csv"),
            "gse215789": str(ROOT / "data" / "processed" / "GSE215789_plasma_count.csv"),
        }
        if any(not Path(path).is_file() for path in self.paths.values()):
            self.skipTest("processed ranking matrices are not present")

    def test_loader_dimensions_and_summary(self):
        datasets = load_matrices(self.paths)
        summary = summarize_datasets(datasets).set_index("dataset")
        for name, expected in EXPECTED.items():
            frame = datasets[name]
            self.assertEqual(frame.shape, (expected["n_samples"], expected["n_mirnas"] + 1))
            self.assertEqual(int((frame["Diagnosis"] == "AD").sum()), expected["n_ad"])
            self.assertEqual(int((frame["Diagnosis"] == "Control").sum()), expected["n_control"])
            self.assertEqual(int(summary.loc[name, "n_missing"]), 0)
            self.assertGreaterEqual(float(summary.loc[name, "minimum"]), 0.0)
            self.assertLess(float(summary.loc[name, "maximum"]), 1.0 + 1e-9)
            self.assertEqual(summary.loc[name, "biological_source"], {"ROSMAP": "DLPFC", "GSE120584": "serum", "GSE215789": "plasma"}[name])


if __name__ == "__main__":
    unittest.main()
