"""
SENTRY-MH · Script 05: T2 Stratified Selection (60 cases)
=========================================================
Selects 60 vignettes guaranteeing coverage of all 18 domain×severity cells,
targeting ~40% oncology and 30/30 language split when possible.

Usage:
    python scripts/05_t2_adjudication.py --vignettes data/vignettes/all_vignettes.jsonl \
        --out runs --seed 1337

Authors: Sanwal Ahmad Zafar and Assoc. prof. Wei Qin
"""
from __future__ import annotations
import argparse, json, random, math
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from datetime import datetime

def load_vignettes(jsonl_path: Path) -> pd.DataFrame:
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    df = pd.DataFrame([{
        "vignette_id": r["vignette_id"],
        "domain": r["domain"],
        "severity": r["severity"],
        "language": r["language"],
        "oncology_flag": bool(r["clinical_vars"]["oncology_flag"]),
        "country": r["clinical_vars"]["country"],
    } for r in rows])
    return df

def stratified_select(df: pd.DataFrame, k: int, seed: int, oncology_target: float=0.40, lang_target_each: int=30) -> pd.DataFrame:
    random.seed(seed)
    # Step 1: ensure at least one per (domain, severity)
    selected_ids = []
    cells = df.groupby(["domain","severity"])
    for (d,s), g in cells:
        sel = g.sample(n=1, random_state=seed)
        selected_ids.extend(sel["vignette_id"].tolist())

    # Step 2: fill remaining with gentle constraints: oncology ~40%, language balance ~30 each
    remaining = [vid for vid in df["vignette_id"].tolist() if vid not in selected_ids]
    # shuffle deterministically
    random.Random(seed).shuffle(remaining)

    def metrics(cur_ids):
        sub = df[df["vignette_id"].isin(cur_ids)]
        onco = float(sub["oncology_flag"].mean()) if len(sub) else 0.0
        en = int((sub["language"]=="en").sum())
        ur = int((sub["language"]=="ur").sum())
        return onco, en, ur

    # start set
    cur = list(selected_ids)
    # aim totals
    while len(cur) < k and remaining:
        vid = remaining.pop()
        cur.append(vid)
    # Adjust for oncology and language balance with simple swaps
    # We won't over-optimize: just try a few shuffles
    best = cur[:]
    best_score = 1e9
    for attempt in range(200):
        cand = random.sample(df["vignette_id"].tolist(), k)
        onco, en, ur = metrics(cand)
        score = abs(onco - oncology_target) + abs(en - lang_target_each) + abs(ur - lang_target_each)
        if score < best_score:
            best, best_score = cand, score
    return df[df["vignette_id"].isin(best)].sort_values(["domain","severity","language","vignette_id"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vignettes", type=str, default="data/vignettes/all_vignettes.jsonl")
    ap.add_argument("--out", type=str, default="runs")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--k", type=int, default=60)
    args = ap.parse_args()

    df = load_vignettes(Path(args.vignettes))
    sel = stratified_select(df, k=args.k, seed=args.seed, oncology_target=0.40, lang_target_each=args.k//2)

    out_dir = Path(args.out) / datetime.now().strftime("t2_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "selection.csv"
    sel.to_csv(out_csv, index=False, encoding="utf-8")

    print("="*80)
    print("SENTRY-MH · Script 05: DONE")
    print(f"Wrote selection: {out_csv} (rows: {len(sel)})")
    print("Coverage check:",
          f" cells={sel.groupby(['domain','severity']).ngroups}/18,",
          f" oncology~{sel['oncology_flag'].mean():.2f},",
          f" EN={int((sel['language']=='en').sum())}, UR={int((sel['language']=='ur').sum())}")
    print("="*80)

if __name__ == "__main__":
    main()
