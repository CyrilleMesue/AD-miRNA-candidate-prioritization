#!/usr/bin/env python
"""Prepare the ROSMAP miRNA matrix from Synapse downloads. See data/raw/README.md."""
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data.prepare_rosmap_mirna import prepare_rosmap_mirna

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gct", required=True)
    p.add_argument("--assay-metadata", required=True)
    p.add_argument("--clinical", required=True)
    p.add_argument("--biospecimen", required=True)
    p.add_argument("--out", default=str(ROOT / "data/processed/ROSMAP_brain_norm.csv"))
    a = p.parse_args()
    for label, path in [("gct", a.gct), ("assay-metadata", a.assay_metadata), ("clinical", a.clinical), ("biospecimen", a.biospecimen)]:
        if not Path(path).is_file():
            raise SystemExit(f"Missing ROSMAP input ({label}): {path}")
    df = prepare_rosmap_mirna(a.gct, a.assay_metadata, a.clinical, a.biospecimen, a.out)
    print(f"Wrote {a.out} with shape {df.shape}")

if __name__ == "__main__":
    main()
