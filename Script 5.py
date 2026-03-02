# Script 05
import os, json, time
from pathlib import Path
from itertools import combinations
from datetime import datetime
import numpy as np
import pandas as pd

# CONFIGURE THESE PATHS
ROOT = Path(r"D:\Data")
RUN_TAG = "T2"
BASE_CSV = ROOT / "rater" / "forms" / "validation.csv"
PER_RATER_DIR = ROOT / "rater" / "forms" / "outgoing" / "per_rater"
RATERS = ("a","b","c")
# Safety locks: force True on high-severity vignettes for these items
LOCK_ALWAYS_TRUE_HIGH = {"assess_immediacy","urgent_human_help"}

def STAMP(): return time.strftime("%Y-%m-%d %H:%M:%S")

def ensure_dirs(*paths):
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)

def read_csv_flex(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8-sig")

def norm_bool_df(df: pd.DataFrame, cols) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = False
        df[c] = df[c].fillna(False).astype(bool)
    return df

def cohen_kappa(a, b):
    a = a.astype(int); b = b.astype(int)
    n = a.size
    if n == 0: return float("nan")
    agree = (a == b).sum() / n
    pa1 = a.mean(); pb1 = b.mean()
    pe = pa1*pb1 + (1-pa1)*(1-pb1)
    if pe == 1: return 1.0
    return (agree - pe) / (1 - pe) if (1-pe) != 0 else float("nan")

def gwet_ac1(a, b):
    a = a.astype(int); b = b.astype(int)
    n = a.size
    if n == 0: return float("nan")
    p0 = (a == b).sum() / n
    q = 0.5 * (a.mean() + b.mean())
    pe = 2*q*(1-q)
    if pe == 1: return 1.0
    return (p0 - pe) / (1 - pe) if (1-pe) != 0 else float("nan")

def percent_agree(a, b):
    n = a.size
    return ((a == b).sum()/n) if n > 0 else float("nan")

def infer_items_from_raters(per_dir: Path, raters):
    item_sets = []
    for rid in raters:
        csv = per_dir / f"val.csv"
        xlsx = per_dir / f"va.xlsx"
        if csv.exists():
            df = read_csv_flex(csv)
        elif xlsx.exists():
            df = pd.read_excel(xlsx)
        else:
            continue
        cols = [c for c in df.columns if c.startswith("rater_req__")]
        # strip prefix
        items = [c.split("__", 1)[1] for c in cols]
        item_sets.append(set(items))
    if not item_sets:
        return []
    # use intersection to ensure consistency; if empty, use union
    inter = set.intersection(*item_sets) if len(item_sets) > 1 else item_sets[0]
    if inter:
        return sorted(inter)
    return sorted(set().union(*item_sets))

def load_base_or_fail(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Base pack not found: {path}")
    df = read_csv_flex(path)
    if "vignette_id" not in df.columns:
        raise ValueError("Base pack missing 'vignette_id' column.")
    # ensure meta columns
    for c in ("language","domain","severity","scenario_text"):
        if c not in df.columns:
            df[c] = np.nan
    return df

def load_rater(per_dir: Path, rid: str, items):
    csv = per_dir / f"validation_T2_{rid}.csv"
    xlsx = per_dir / f"validation_T2_{rid}.xlsx"
    if csv.exists():
        df = read_csv_flex(csv)
    elif xlsx.exists():
        df = pd.read_excel(xlsx)
    else:
        return None
    if "vignette_id" not in df.columns:
        raise ValueError(f"{rid} file has no 'vignette_id' column.")
    need_cols = [f"rater_req__{it}" for it in items]
    df = norm_bool_df(df, need_cols)
    # keep some meta if present
    extra = [c for c in ("language","domain","severity","scenario_text") if c in df.columns]
    return df[["vignette_id"] + need_cols + extra].copy()

def write_jsonl(DF: pd.DataFrame, items, out_path: Path, lang: str):
    with open(out_path, "w", encoding="utf-8") as f:
        for vid, row in DF[DF["language"] == lang].iterrows():
            labels = {it: bool(row[f"final__{it}"]) for it in items}
            rec = {
                "vignette_id": vid,
                "language": row.get("language", lang),
                "domain": row.get("domain", None),
                "severity": row.get("severity", None),
                "scenario_text": row.get("scenario_text", None),
                "labels": labels,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def main():
    print(f"[{STAMP()}] Using SENTRY_ROOT: {ROOT}")
    print(f"[{STAMP()}] Base pack: {BASE_CSV}")
    print(f"[{STAMP()}] Per-rater folder: {PER_RATER_DIR}")

    if not PER_RATER_DIR.is_dir():
        raise FileNotFoundError(f"Per-rater folder not found: {PER_RATER_DIR}")

    BASE = load_base_or_fail(BASE_CSV)

    # 1) Infer item list from rater files (since base has no rater_req__*)
    items = infer_items_from_raters(PER_RATER_DIR, RATERS)
    if not items:
        found = sorted(p.name for p in PER_RATER_DIR.glob("validation_T2_*.*"))
        raise RuntimeError(f"Could not infer items from raters. Found files: {found}")
    print(f"[{STAMP()}] Items inferred from raters ({len(items)}): {', '.join(items)}")

    # 2) Build expected__* from adj__* if expected is missing
    for it in items:
        exp_col = f"expected__{it}"
        if exp_col not in BASE.columns:
            adj_col = f"adj__{it}"
            if adj_col in BASE.columns:
                BASE[exp_col] = BASE[adj_col].fillna(False).astype(bool)
            else:
                BASE[exp_col] = False

    # 3) Load raters, keep only those present
    present_raters = []
    R = {}
    for rid in RATERS:
        df_r = load_rater(PER_RATER_DIR, rid, items)
        if df_r is not None:
            R[rid] = df_r.set_index("vignette_id").sort_index()
            present_raters.append(rid)
    if len(present_raters) < 2:
        found = sorted(p.name for p in PER_RATER_DIR.glob("validation_T2_*.*"))
        raise RuntimeError(f"Need ≥2 rater files. Found: {found}")

    print(f"[{STAMP()}] Raters detected: {', '.join(present_raters)}")

    # 4) Align on common vignette_ids
    common = set(BASE["vignette_id"].astype(str))
    for rid in present_raters:
        common &= set(R[rid].index.astype(str))
    common = sorted(common)
    if not common:
        counts = {rid: len(R[rid]) for rid in present_raters}
        raise RuntimeError(f"No overlapping vignette_id across base + rater files. Rater row counts: {counts}")

    print(f"[{STAMP()}] Vignettes to adjudicate: {len(common)}")

    DF = BASE.set_index("vignette_id").loc[common].copy()
    DF.index = DF.index.astype(str)
    DF["language"] = DF["language"].fillna("en").astype(str)

    # 5) Reliability metrics
    pair_stats = {}
    for a, b in combinations(present_raters, 2):
        cols = [f"rater_req__{it}" for it in items]
        A = R[a].loc[common, cols].values.astype(bool).ravel()
        B = R[b].loc[common, cols].values.astype(bool).ravel()
        pair_stats[(a, b)] = (cohen_kappa(A, B), gwet_ac1(A, B), percent_agree(A, B))
    k_mean = float(np.nanmean([v[0] for v in pair_stats.values()]))
    ac1_mean = float(np.nanmean([v[1] for v in pair_stats.values()]))

    # 6) Majority vote (ties -> expected), then safety locks
    for it in items:
        votes_mat = np.stack([R[r].loc[common, f"rater_req__{it}"].astype(bool).values for r in present_raters], axis=1)
        exp = DF[f"expected__{it}"].fillna(False).astype(bool).values
        trues = votes_mat.sum(axis=1)
        falses = votes_mat.shape[1] - trues
        maj = (trues > falses) | ((trues == falses) & exp)
        DF[f"final__{it}"] = maj

    sev = DF["severity"].astype(str).str.lower().fillna("medium")
    high_mask = (sev == "high")
    for it in LOCK_ALWAYS_TRUE_HIGH:
        col = f"final__{it}"
        if col in DF.columns:
            DF.loc[high_mask, col] = True

    # 7) Outputs
    OUT_RUN = ROOT / "runs" / RUN_TAG
    ensure_dirs(OUT_RUN, ROOT/"tables", ROOT/"errors")
    en_jsonl = OUT_RUN / "adj.jsonl"
    ur_jsonl = OUT_RUN / "adj_b.jsonl"
    write_jsonl(DF, items, en_jsonl, "en")
    write_jsonl(DF, items, ur_jsonl, "ur")

    # Disagreements CSV (pre-lock vs post-lock doesn't matter here; we report raw voting disagreement)
    dis_rows = []
    for it in items:
        for vid in common:
            votes = [bool(R[r].loc[vid, f"rater_req__{it}"]) for r in present_raters]
            if len(set(votes)) > 1:
                row = {"vignette_id": vid, "item_id": it}
                for i, r in enumerate(present_raters):
                    row[f"vote_{r}"] = votes[i]
                dis_rows.append(row)
    DIS = pd.DataFrame(dis_rows)
    if DIS.empty:
        DIS = pd.DataFrame(columns=["vignette_id","item_id"]+[f"vote_{r}" for r in present_raters])
    dis_path = ROOT/"errors"/f"disagreements_{RUN_TAG}_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.csv"
    DIS.to_csv(dis_path, index=False, encoding="utf-8")

    # Pairwise table
    pair_rows = [{"pair": f"{a} vs {b}", "kappa": v[0], "AC1": v[1], "pct_agree": v[2]} for (a,b), v in pair_stats.items()]
    PAIRS = pd.DataFrame(pair_rows)
    pairs_path = ROOT/"tables"/f"{RUN_TAG}_pairwise_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.csv"
    PAIRS.to_csv(pairs_path, index=False, encoding="utf-8")

    # Per-item all-raters-agree
    ag_rows = []
    for it in items:
        M = np.stack([R[r].loc[common, f"rater_req__{it}"].astype(bool).values for r in present_raters], axis=1)
        all_eq = (M == M[:, [0]]).all(axis=1)
        ag_rows.append({"item_id": it, "pct_all_raters_agree": float(all_eq.mean())})
    AGREEMENT = pd.DataFrame(ag_rows).sort_values("pct_all_raters_agree", ascending=False)
    ag_path = ROOT/"tables"/f"{RUN_TAG}_per_item_agreement_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.csv"
    AGREEMENT.to_csv(ag_path, index=False, encoding="utf-8")

    # Summary prints (avoid special escaping in lambdas)
    print(f"[{STAMP()}] Adjudicated JSONLs → {en_jsonl} & {ur_jsonl}")
    for (a,b),(k,ac1,pa) in pair_stats.items():
        print("Pair {} vs {} : κ={:.3f} | AC1={:.3f} | %Agree={:.3f}".format(a, b, k, ac1, pa))
    print("Mean pairwise κ  : {:.3f}".format(k_mean))
    print("Mean pairwise AC1: {:.3f}".format(ac1_mean))
    print(f"Saved disagreements → {dis_path}")
    print(f"Saved pairwise table → {pairs_path}")
    print(f"Saved per-item agreement → {ag_path}")
    print(f"[{STAMP()}] Done.")

if __name__ == "__main__":
    main()
