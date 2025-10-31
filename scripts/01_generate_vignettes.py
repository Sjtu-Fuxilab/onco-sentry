"""
SENTRY-MH · Script 01: Vignette Generation
==========================================
Generates the 720-slot vignette set from YAML config without authoring text.
- 6 domains × per_domain (EN) + Urdu mirrors → 720 total when per_domain=60
- Respects oncology_rate (~40%) and country weights
- Scenario text is intentionally blank to avoid synthetic content

Usage:
    python scripts/01_generate_vignettes.py --config config/example_config.yaml --out data/vignettes

Authors: Sanwal Ahmad Zafar and Assoc. prof. Wei Qin
"""

from __future__ import annotations
import argparse, json, random
from pathlib import Path
from typing import Dict, List
import pandas as pd
import yaml
from datetime import datetime

def load_config(p: Path) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_out(out_dir: Path):
    (out_dir).mkdir(parents=True, exist_ok=True)

def pick(items, probs):
    # random.choices wrapper with deterministic behavior
    return random.choices(items, weights=probs, k=1)[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config/example_config.yaml")
    ap.add_argument("--out", type=str, default="data/vignettes")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    seed = args.seed if args.seed is not None else int(cfg["project"].get("seed", 1337))
    random.seed(seed)

    out_dir = Path(args.out)
    ensure_out(out_dir)

    domains: List[str] = cfg["domains"]
    langs: List[str] = cfg["languages"]
    per_domain: int = int(cfg["vignettes"]["per_domain"])     # per-domain EN count
    oncology_rate: float = float(cfg["vignettes"]["oncology_rate"])
    cw = cfg["vignettes"]["country_weights"]
    countries = ["neutral", "us", "uk", "pk"]
    weights = [cw.get("neutral", 0.55), cw.get("us", 0.15), cw.get("uk", 0.15), cw.get("pk", 0.15)]

    # Severity sampling: even by default unless T1 config provided (not used here)
    severities = ["high", "medium", "low"]
    sev_w = [1, 1, 1]

    rows = []
    vid = 0
    for d in domains:
        for i in range(per_domain):
            # EN instance
            vid += 1
            sev = pick(severities, sev_w)
            country = pick(countries, weights)
            oncology_flag = (random.random() < oncology_rate)
            base_id = f"{d}_{sev}_en_{vid:03d}"
            rows.append(dict(
                vignette_id=base_id,
                domain=d,
                language="en",
                severity=sev,
                scenario_text="",                      # NO synthetic text
                clinical_vars=dict(age=None, role=None, living=None, oncology_flag=oncology_flag, country=country),
                ground_truth_sms={},                    # left blank to avoid introducing content
                version=1,
            ))
            # Urdu mirror
            vid += 1
            base_id_ur = base_id.replace("_en_", "_ur_")
            country_ur = pick(countries, weights)
            oncology_flag_ur = (random.random() < oncology_rate)
            rows.append(dict(
                vignette_id=base_id_ur,
                domain=d,
                language="ur",
                severity=sev,
                scenario_text="",                      # NO synthetic text
                clinical_vars=dict(age=None, role=None, living=None, oncology_flag=oncology_flag_ur, country=country_ur),
                ground_truth_sms={},
                version=1,
            ))

    # Save JSONL + manifest
    jsonl_path = Path(out_dir) / "all_vignettes.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    man = pd.DataFrame([{
        "vignette_id": r["vignette_id"],
        "domain": r["domain"],
        "language": r["language"],
        "severity": r["severity"],
        "oncology_flag": bool(r["clinical_vars"]["oncology_flag"]),
        "country": r["clinical_vars"]["country"],
    } for r in rows])
    man_path = Path(out_dir) / "manifest.csv"
    man.to_csv(man_path, index=False, encoding="utf-8")

    print("="*80)
    print("SENTRY-MH · Script 01: DONE")
    print(f"Seed: {seed} | Vignettes: {len(rows)} | JSONL: {jsonl_path}")
    print(f"Manifest: {man_path}")
    print("="*80)

if __name__ == "__main__":
    main()
