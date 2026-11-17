#!/usr/bin/env python
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data.prepare_gse215789 import prepare_gse215789

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--counts", required=True)
    p.add_argument("--series-matrix", required=True)
    p.add_argument("--sample-map", default=str(ROOT / "data/external/GSE215789_sample_name_mapping.json"))
    p.add_argument("--out", default=str(ROOT / "data/processed/GSE215789_plasma_count.csv"))
    a = p.parse_args()
    for path in (a.counts, a.series_matrix, a.sample_map):
        if not Path(path).is_file():
            raise SystemExit(f"Missing input: {path}")
    df = prepare_gse215789(a.counts, a.series_matrix, a.sample_map, a.out)
    print(f"Wrote {a.out} with shape {df.shape}")

if __name__ == "__main__":
    main()
