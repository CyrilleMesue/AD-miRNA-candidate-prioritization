#!/usr/bin/env python
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data.prepare_gse120584 import prepare_gse120584

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--series-matrix", required=True)
    p.add_argument("--mirbase-map", required=True)
    p.add_argument("--out", default=str(ROOT / "data/processed/GSE120584_serum_norm.csv"))
    a = p.parse_args()
    for path in (a.series_matrix, a.mirbase_map):
        if not Path(path).is_file():
            raise SystemExit(f"Missing input: {path}")
    df = prepare_gse120584(a.series_matrix, a.mirbase_map, a.out)
    print(f"Wrote {a.out} with shape {df.shape}")

if __name__ == "__main__":
    main()
