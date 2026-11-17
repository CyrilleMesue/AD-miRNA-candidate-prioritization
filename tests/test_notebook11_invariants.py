"""Invariants of the extracted Notebook 11 workflow.

Sample-count checks run only when ORIGINAL_PROJECT points at the
original project. They call the same loader the pipeline uses.
"""
import os
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prioritization.original import (
    AGGREGATORS,
    FEATURE_RANKERS,
    ad_mirna_target_pipeline,
    literature_review_table,
)
from prioritization import matrices


class InventoryTests(unittest.TestCase):
    def test_thirteen_rankers(self):
        self.assertEqual(len(FEATURE_RANKERS), 13)
        self.assertEqual(len(set(FEATURE_RANKERS)), 13)

    def test_twelve_aggregators(self):
        self.assertEqual(len(AGGREGATORS), 12)
        self.assertIn("mean_rank", AGGREGATORS)
        self.assertIn("ta_weight", AGGREGATORS)
        self.assertIn("geom.mean_rank", AGGREGATORS)

    def test_gene_cutoff_argument_is_unused(self):
        import inspect
        src = inspect.getsource(ad_mirna_target_pipeline)
        self.assertEqual(src.count("gene_db_cutoff"), 1)
        self.assertIn("[:100]", src)

    def test_literature_table_does_not_reduce_to_three_mirnas(self):
        table = literature_review_table()
        names = set(table["microRNA"])
        self.assertEqual(len(table), 14)
        self.assertTrue({"hsa-mir-324-3p", "hsa-mir-574-3p", "hsa-mir-769-3p"} <= names)
        self.assertGreater(len(names), 3)

    def test_downsample_is_not_used_by_loader(self):
        src = Path(matrices.__file__).read_text()
        self.assertIn("def downsample_gse120584_unused", src)
        self.assertNotIn("downsample_gse120584_unused(", src.replace("def downsample_gse120584_unused", ""))


class OriginalMatrixTests(unittest.TestCase):
    def setUp(self):
        project = os.environ.get("ORIGINAL_PROJECT")
        if not project:
            self.skipTest("ORIGINAL_PROJECT is not set")
        self.base = Path(project) / "data" / "microRNA"
        for name in ("ROSMAP_brain_norm.csv", "GSE120584_serum_norm.csv", "GSE215789_plasma_count.csv"):
            if not (self.base / name).is_file():
                self.skipTest(f"missing {name}")

    def test_sample_counts_and_shapes(self):
        from prioritization.matrices import load_ranking_datasets
        data = load_ranking_datasets({
            "rosmap": str(self.base / "ROSMAP_brain_norm.csv"),
            "gse120584": str(self.base / "GSE120584_serum_norm.csv"),
            "gse215789": str(self.base / "GSE215789_plasma_count.csv"),
        })
        expected = {
            "ROSMAP": (378, 310, {"AD": 209, "Control": 169}),
            "GSE120584": (1309, 2522, {"AD": 1021, "Control": 288}),
            "GSE215789": (44, 1138, {"AD": 23, "Control": 21}),
        }
        for name, (nrows, ncols, counts) in expected.items():
            frame = data[name]
            self.assertEqual(frame.shape, (nrows, ncols), name)
            self.assertEqual(frame["Diagnosis"].value_counts().to_dict(), counts, name)
            self.assertFalse(frame.drop(columns=["Diagnosis"]).isna().any().any(), name)


if __name__ == "__main__":
    unittest.main()
