"""
SENTRY-MH · Script 02: Export Validation Packs
==============================================
Creates rater CSV templates from generated vignettes and the 13-item rubric.
- No patient data; scenario_text remains blank
- Writes to rater/forms/<stage>_template.csv

Usage:
    python scripts/02_export_validation.py --stage T1 --vignettes data/vignettes/all_vignettes.jsonl

Authors: Sanwal Ahmad Zafar and Assoc. prof. Wei Qin
"""
from __future__ import annotations
import argparse, json, csv
from pathlib import Path
from typing import List, Dict
import pandas as pd

RUBRIC_CSV = Path("rubric") / "sms_items.csv"

def load_vignettes(jsonl_path: Path) -> List[Dict]:
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            rows.append(json.loads(line))
    return rows

def load_rubric() -> pd.DataFrame:
    df = pd.read_csv(RUBRIC_CSV)
    if len(df) != 13:
        raise ValueError(f"Rubric must have exactly 13 items; found {len(df)}")
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=str, choices=["T1","T2"], default="T1")
    ap.add_argument("--vignettes", type=str, default="data/vignettes/all_vignettes.jsonl")
    args = ap.parse_args()

    vignettes = load_vignettes(Path(args.vignettes))
    df_rubric = load_rubric()
    item_ids = df_rubric["item_id"].tolist()

    out_dir = Path("rater/forms")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{args.stage.lower()}_template.csv"

    # Columns: metadata + one column per rubric item for rater entries
    cols = ["rater_id", "vignette_id", "domain", "language", "severity", "oncology_flag", "country"]
    cols += [f"rate::{iid}" for iid in item_ids]

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for v in vignettes:
            row = [
                "",  # rater_id to be filled
                v["vignette_id"],
                v["domain"],
                v["language"],
                v["severity"],
                bool(v["clinical_vars"]["oncology_flag"]),
                v["clinical_vars"]["country"],
            ]
            row += ["" for _ in item_ids]  # empty ratings
            writer.writerow(row)

    print("="*80)
    print("SENTRY-MH · Script 02: DONE")
    print(f"Wrote template: {out_csv} (rows: {len(vignettes)})")
    print("="*80)

if __name__ == "__main__":
    main()
