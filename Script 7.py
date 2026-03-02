# SENTRY-MH Script 07

from __future__ import annotations
import os, sys, json, re, time, warnings, shutil, itertools
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import (entropy, mannwhitneyu, fisher_exact,
                          kruskal, kendalltau, spearmanr)
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")


# CONFIG & UTILITIES

MASTER_SEED          = 42
N_BOOTSTRAP          = 200
AL_BUDGET            = 60
AUGMENTED_BUDGET     = 80
N_ROBUSTNESS_SEEDS   = 200
N_RANK_SEEDS         = 50
BUDGET_RANGE         = list(range(18, 121, 6))

# Composite score weights — BALD-motivated (Houlsby et al. 2011)
WEIGHT_SCHEMES = {
    "equal":         (0.50, 0.50),
    "entropy_heavy": (0.33, 0.67),
    "std_heavy":     (0.67, 0.33),
    "entropy_only":  (0.00, 1.00),
    "std_only":      (1.00, 0.00),
}
PRIMARY_WEIGHTS = "equal"

ONCO_THRESHOLD  = 3
ONCO_TARGET_LO  = 0.35
ONCO_TARGET_HI  = 0.45

np.random.seed(MASTER_SEED)


def STAMP() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def LOG(level: str, msg: str) -> None:
    pad = {"OK": "OK  ", "INFO": "INFO", "WARN": "WARN", "ERR": "ERR "}.get(level, level)
    print(f"[{STAMP()}] [{pad}] {msg}")

def in_jupyter() -> bool:
    return "ipykernel" in sys.modules or "JPY_PARENT_PID" in os.environ

def select_root() -> Path:
    if "SENTRY_ROOT" in os.environ:
        return Path(os.environ["SENTRY_ROOT"]).expanduser().resolve()
    hardcoded = Path(r"D:\个人文件夹\Sanwal\LLM")
    if hardcoded.exists():
        return hardcoded
    if not in_jupyter():
        argv = [a for a in sys.argv[1:] if not a.startswith("-")]
        if argv:
            return Path(argv[0]).expanduser().resolve()
    return Path.cwd()

ROOT = select_root()
if not (ROOT / "rubric").exists() and (ROOT / "SENTRY-MH" / "rubric").exists():
    ROOT = ROOT / "SENTRY-MH"

DIRS = {
    "rubric":      ROOT / "rubric",
    "vignettes":   ROOT / "data" / "vignettes",
    "runs_T1":     ROOT / "runs" / "T1",
    "rater_forms": ROOT / "rater" / "forms",
    "advanced":    ROOT / "advanced_validation",
}
for name, p in DIRS.items():
    p.mkdir(parents=True, exist_ok=True)

ADV = DIRS["advanced"]
LOG("OK", f"Root: {ROOT}")
LOG("OK", f"Output: {ADV}")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows

def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


# --- §1 LOAD SMS RUBRIC + VIGNETTES ---

LOG("INFO", "§1 Loading data ...")

sms_csv = DIRS["rubric"] / "sms_items.csv"
if not sms_csv.exists():
    for alt in [ROOT / "sms_items.csv",
                ROOT / "rubric" / "sms_items.csv",
                ROOT.parent / "rubric" / "sms_items.csv"]:
        if alt.exists():
            sms_csv = alt
            break
    else:
        raise FileNotFoundError(f"SMS rubric not found in {DIRS['rubric']}")

SMS_DF    = pd.read_csv(sms_csv)
SMS_ITEMS = SMS_DF["item_id"].tolist()
LOG("OK", f"SMS items: {len(SMS_ITEMS)}")

vignette_paths = [
    (DIRS["runs_T1"] / "frozen_vignettes_en.jsonl",
     DIRS["runs_T1"] / "frozen_vignettes_ur.jsonl"),
    (DIRS["runs_T1"] / "adjudicated_vignettes_en.jsonl",
     DIRS["runs_T1"] / "adjudicated_vignettes_ur.jsonl"),
    (DIRS["vignettes"] / "vignettes_en.jsonl",
     DIRS["vignettes"] / "vignettes_ur.jsonl"),
]

vignettes_raw = []
vig_en_path = vig_ur_path = None
for en_path, ur_path in vignette_paths:
    if en_path.exists() and ur_path.exists():
        vig_en_path, vig_ur_path = en_path, ur_path
        vignettes_raw = read_jsonl(en_path) + read_jsonl(ur_path)
        LOG("OK", f"Vignettes from: {en_path.parent.name}/")
        break
if not vignettes_raw:
    raise RuntimeError("No vignettes found! Run Scripts 01-04 first.")

VIG_DF = pd.DataFrame(vignettes_raw)

if "labels" in VIG_DF.columns:
    for item in SMS_ITEMS:
        VIG_DF[f"expected__{item}"] = VIG_DF["labels"].apply(
            lambda x: bool(x.get(item, False)) if isinstance(x, dict) else False)
elif "ground_truth_sms" in VIG_DF.columns:
    for item in SMS_ITEMS:
        col = f"expected__{item}"
        if col not in VIG_DF.columns:
            VIG_DF[col] = VIG_DF["ground_truth_sms"].apply(
                lambda x: bool(x.get(item, False)) if isinstance(x, dict) else False)
else:
    for item in SMS_ITEMS:
        ec = f"expected__{item}"
        if ec not in VIG_DF.columns:
            for alt in [f"adj__{item}", f"final__{item}"]:
                if alt in VIG_DF.columns:
                    VIG_DF[ec] = VIG_DF[alt]
                    break
            else:
                VIG_DF[ec] = False

n_en = len(VIG_DF[VIG_DF["language"] == "en"])
n_ur = len(VIG_DF[VIG_DF["language"] == "ur"])
LOG("OK", f"Total vignettes: {len(VIG_DF)}  (EN={n_en}, UR={n_ur})")
if "severity" in VIG_DF.columns:
    LOG("OK", f"Severity: {VIG_DF['severity'].value_counts().to_dict()}")


# --- §2 FEATURE MATRIX ---

LOG("INFO", "§2 Building feature matrix ...")

def compute_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    features = df.copy()
    if "domain" in df.columns:
        for domain in df["domain"].unique():
            if pd.notna(domain):
                features[f"domain_{domain}"] = (df["domain"] == domain).astype(int)
    if "severity" in df.columns:
        features["severity_ord"] = df["severity"].map(
            {"low": 0, "medium": 1, "high": 2}).fillna(1)
    if "language" in df.columns:
        features["lang_en"] = (df["language"] == "en").astype(int)
    if "scenario_text" in df.columns:
        features["text_length"] = df["scenario_text"].fillna("").str.len()
        features["text_words"]  = df["scenario_text"].fillna("").str.split().str.len()
    if "clinical_vars" in df.columns:
        features["has_oncology"] = df["clinical_vars"].apply(
            lambda x: bool(x.get("oncology_flag", False))
            if isinstance(x, dict) else False)
        features["country_cue"] = df["clinical_vars"].apply(
            lambda x: 0 if x.get("country", "neutral") == "neutral"
            else 1 if isinstance(x, dict) else 0)
    expected_cols = [c for c in df.columns if c.startswith("expected__")]
    if expected_cols:
        features["n_required_items"] = df[expected_cols].sum(axis=1)
    exclude = {"vignette_id", "scenario_text", "domain", "severity",
               "language", "ground_truth_sms", "clinical_vars", "labels"}
    feature_cols = [c for c in features.columns
                    if c not in exclude
                    and features[c].dtype in [np.int64, np.float64, int, float]]
    return features[["vignette_id"] + feature_cols]

FEAT_DF      = compute_feature_matrix(VIG_DF)
X            = FEAT_DF.drop(columns=["vignette_id"]).fillna(0).values
VIGNETTE_IDS = FEAT_DF["vignette_id"].values
META_DF      = VIG_DF[["vignette_id", "domain", "severity", "language"]].copy()

expected_cols = [f"expected__{item}" for item in SMS_ITEMS]
y_multi       = VIG_DF[expected_cols].values.astype(int)
Y_BINARY      = (y_multi.sum(axis=1) > len(SMS_ITEMS) / 2).astype(int)

ALL_DOMAINS    = sorted(VIG_DF["domain"].unique())
ALL_SEVERITIES = ["high", "medium", "low"]
ALL_COMBOS     = list(itertools.product(ALL_DOMAINS, ALL_SEVERITIES))

LOG("OK", f"Feature matrix: {X.shape}, target prevalence: {Y_BINARY.mean():.3f}")


# --- §3 BOOTSTRAP ENSEMBLE UNCERTAINTY ---

LOG("INFO", f"§3 Uncertainty estimation (seed={MASTER_SEED}, n_boot={N_BOOTSTRAP}) ...")

def compute_uncertainty(seed: int, n_boot: int,
                        w_std: float = 0.5, w_ent: float = 0.5,
                        verbose: bool = False) -> pd.DataFrame:
    """Bootstrap ensemble uncertainty with BALD-motivated composite score."""
    rng = np.random.RandomState(seed)
    predictions = []
    for i in range(n_boot):
        if verbose and i % 50 == 0:
            print(f"  bootstrap {i}/{n_boot}", end="\r")
        idx = rng.choice(len(X), size=len(X), replace=True)
        clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=i)
        clf.fit(X[idx], Y_BINARY[idx])
        predictions.append(clf.predict_proba(X)[:, 1])

    predictions  = np.array(predictions)
    pred_mean    = predictions.mean(axis=0)
    pred_std     = predictions.std(axis=0)
    pred_entropy = np.array([entropy([p, 1 - p]) if 0 < p < 1
                             else 0.0 for p in pred_mean])

    df = pd.DataFrame({
        "vignette_id": VIGNETTE_IDS,
        "pred_mean": pred_mean,
        "pred_std": pred_std,
        "pred_entropy": pred_entropy,
    }).merge(META_DF, on="vignette_id", how="left")

    std_max = df["pred_std"].max()
    ent_max = df["pred_entropy"].max()
    df["uncertainty_score"] = (
        w_std * (df["pred_std"]     / std_max if std_max > 0 else 0) +
        w_ent * (df["pred_entropy"] / ent_max if ent_max > 0 else 0)
    )
    if verbose:
        print(f"\n  Done. Mean unc = {df['uncertainty_score'].mean():.4f}")
    return df

w_std, w_ent = WEIGHT_SCHEMES[PRIMARY_WEIGHTS]
UNCERTAINTY_DF = compute_uncertainty(
    seed=MASTER_SEED, n_boot=N_BOOTSTRAP,
    w_std=w_std, w_ent=w_ent, verbose=True)

LOG("OK", f"Range: [{UNCERTAINTY_DF['uncertainty_score'].min():.4f}, "
    f"{UNCERTAINTY_DF['uncertainty_score'].max():.4f}], "
    f"Mean: {UNCERTAINTY_DF['uncertainty_score'].mean():.4f}")


# --- §4 ONCOLOGY CLASSIFICATION ---

LOG("INFO", "§4 Oncology classification (two-tier lexicon) ...")

TIER1 = [
    r"\bchemotherap\w*", r"\boncolog\w+", r"\bmalignant\b", r"\bmalignancy\b",
    r"\bleukemi\w+", r"\blymphom\w+", r"\bmesotheliom\w+", r"\bglioblaston\w+",
    r"\bglioma\w+", r"\bpalliative\s+care\b", r"\bpalliative\s+treat\w+",
    r"\btargeted\s+therap\w+", r"\btargeted\s+agent\w+",
    r"\bradiation\s+therap\w+", r"\bradiation\s+treat\w+",
    r"\bimmunotherap\w+", r"\bbone\s+marrow\b", r"\btumo[u]?r\s+board\b",
    r"\bmetastas\w+", r"\bchemo\b",
    r"\bterminal\s+cancer\b", r"\badvanced\s+cancer\b",
    r"\bcancer\s+diagnos\w+", r"\bdiagnosed\s+with\s+cancer\b",
    r"\bcancer\s+patient\b", r"\bcancer\s+survivor\b",
    r"\bbreast\s+cancer\b", r"\blung\s+cancer\b", r"\bcolon\s+cancer\b",
    r"\bcervical\s+cancer\b", r"\bovarian\s+cancer\b", r"\bprostate\s+cancer\b",
    r"\boral\s+cancer\b", r"\bliver\s+cancer\b", r"\bpancreatic\s+cancer\b",
    r"\bha?ematolog\w+",
    r"\bcycles?\s+of\s+chemo\w*",
    r"\boncology\s+(?:clinic|ward)\b", r"\bcancer\s+(?:clinic|ward)\b",
    r"\bpalliative\b",
    r"\bsera?taa?n\b", r"\bkimo?therap\w*", r"\bkhabeesa\s+rasoli\b",
    r"\bonkologi\b",
]

TIER2 = [
    r"\bcancer\b", r"\btumo[u]?r\b", r"\bneoplasm\w*",
    r"\bpost[- ]surger\w+", r"\bsurgical\s+resection\b",
    r"\btreatment\s+cycle\w*", r"\bside\s+effects?\s+of\s+treatment\b",
    r"\bhospice\b", r"\bremission\b", r"\brelapse\b", r"\brecurrence\b",
    r"\bstage\s+(?:i{1,3}v?|[1-4])\b",
    r"\bPET\s+scan\b", r"\bCT\s+scan\b", r"\bMRI\s+result\b",
    r"\bclinical\s+trial\b", r"\bsupportive\s+care\b",
    r"\bpain\s+management\b", r"\bnausea\b", r"\bfatigue\b",
    r"\bhair\s+loss\b", r"\balopecia\b",
    r"\bimmunosuppress\w+", r"\bblood\s+count\b",
    r"\bradiation\b", r"\boncologist\b", r"\bdiagnosis\b",
    r"\bmareed\b", r"\bilaaj\b", r"\bilaj\b", r"\bdard\b",
    r"\bkamzori\b", r"\btaskhees\b",
]

_T1 = [re.compile(p, re.IGNORECASE | re.UNICODE) for p in TIER1]
_T2 = [re.compile(p, re.IGNORECASE | re.UNICODE) for p in TIER2]
_TRANSLATOR_BLOCK = re.compile(r"\[TRANSLATE\s+FAITHFULLY.*?\]", re.IGNORECASE | re.DOTALL)

def clean_text(text: str) -> str:
    cleaned = _TRANSLATOR_BLOCK.sub("", text).strip()
    return cleaned if cleaned else text

def classify_oncology(row: Dict[str, Any]) -> Tuple[bool, int]:
    """Deterministic two-tier oncology classification -> (flag, score)."""
    text = clean_text(str(row.get("scenario_text", ""))).lower()
    cv = row.get("clinical_vars", {})
    score = 0
    for rx in _T1:
        if rx.search(text):
            score += 3
            break
    for rx in _T2:
        if rx.search(text):
            score += 1
    if isinstance(cv, dict) and cv.get("oncology_flag") is True:
        score += 2
    return score >= ONCO_THRESHOLD, score

for v in vignettes_raw:
    flag, sc = classify_oncology(v)
    v["_onco_flag"]  = flag
    v["_onco_score"] = sc

raw_rate     = sum(v["_onco_flag"] for v in vignettes_raw) / len(vignettes_raw)
n_onco_total = sum(v["_onco_flag"] for v in vignettes_raw)

if not (ONCO_TARGET_LO <= raw_rate <= ONCO_TARGET_HI):
    LOG("WARN", f"Rate {raw_rate*100:.1f}% outside target. Tuning threshold ...")
    scores_all = [v["_onco_score"] for v in vignettes_raw]
    for thr in sorted(set(scores_all)):
        r = sum(s >= thr for s in scores_all) / len(scores_all)
        if ONCO_TARGET_LO <= r <= ONCO_TARGET_HI:
            ONCO_THRESHOLD = thr
            for v in vignettes_raw:
                v["_onco_flag"] = v["_onco_score"] >= thr
            raw_rate = r
            n_onco_total = sum(v["_onco_flag"] for v in vignettes_raw)
            LOG("OK", f"  Threshold={thr} -> rate={r*100:.1f}%")
            break

LOG("OK", f"Oncology rate: {raw_rate*100:.1f}% ({n_onco_total}/{len(vignettes_raw)}), "
    f"threshold={ONCO_THRESHOLD}")

onco_lookup  = {v.get("vignette_id", ""): v["_onco_flag"]  for v in vignettes_raw}
score_lookup = {v.get("vignette_id", ""): v["_onco_score"] for v in vignettes_raw}

UNCERTAINTY_DF["oncology"]   = UNCERTAINTY_DF["vignette_id"].map(onco_lookup).fillna(False).astype(bool)
UNCERTAINTY_DF["onco_score"] = UNCERTAINTY_DF["vignette_id"].map(score_lookup).fillna(0).astype(int)

def update_jsonl_flags(src_path: Path) -> int:
    if src_path is None or not src_path.exists():
        return 0
    rows = read_jsonl(src_path)
    updated = 0
    for row in rows:
        vid = row.get("vignette_id", "")
        flag = onco_lookup.get(vid)
        if flag is None:
            continue
        if not isinstance(row.get("clinical_vars"), dict):
            row["clinical_vars"] = {}
        if row["clinical_vars"].get("oncology_flag") != flag:
            updated += 1
        row["clinical_vars"]["oncology_flag"] = flag
    bak = src_path.with_suffix(".jsonl.bak")
    if not bak.exists():
        shutil.copy2(src_path, bak)
    write_jsonl(src_path, rows)
    return updated

n_upd_en = update_jsonl_flags(vig_en_path)
n_upd_ur = update_jsonl_flags(vig_ur_path)
LOG("OK", f"JSONL flags updated — EN: {n_upd_en}, UR: {n_upd_ur}")

for dom in ALL_DOMAINS:
    sub = UNCERTAINTY_DF[UNCERTAINTY_DF["domain"] == dom]
    n_pos = sub["oncology"].sum()
    LOG("INFO", f"  {dom:<45}  {n_pos:>3}/{len(sub)} ({n_pos/len(sub)*100:.1f}%)")

rerun_flags    = [classify_oncology(v)[0] for v in vignettes_raw]
original_flags = [v["_onco_flag"] for v in vignettes_raw]
n_diff = sum(a != b for a, b in zip(original_flags, rerun_flags))
LOG("OK", f"Classifier determinism: {n_diff} differences (expected 0)")


# --- §5 FOUR SELECTION STRATEGIES ---

LOG("INFO", "§5 Selection strategies ...")

def coverage_stats(df: pd.DataFrame):
    if len(df) == 0:
        return 0, 0, set(ALL_DOMAINS)
    covered = df.groupby(["domain", "severity"]).ngroups
    doms    = df["domain"].nunique()
    missing = set(ALL_DOMAINS) - set(df["domain"].unique())
    return covered, doms, missing

def select_pure_al(unc_df: pd.DataFrame, budget: int) -> pd.DataFrame:
    return unc_df.nlargest(budget, "uncertainty_score")

def select_stratified_al(unc_df: pd.DataFrame, budget: int) -> pd.DataFrame:
    mandatory = []
    for domain, severity in ALL_COMBOS:
        pool = unc_df[(unc_df["domain"] == domain) & (unc_df["severity"] == severity)]
        if len(pool) > 0:
            mandatory.append(pool.nlargest(1, "uncertainty_score"))
    mandatory_df  = pd.concat(mandatory).drop_duplicates("vignette_id")
    mandatory_ids = set(mandatory_df["vignette_id"])
    remaining     = budget - len(mandatory_ids)
    topup = unc_df[~unc_df["vignette_id"].isin(mandatory_ids)].nlargest(
        remaining, "uncertainty_score")
    return pd.concat([mandatory_df, topup]).drop_duplicates("vignette_id")

def select_random(unc_df: pd.DataFrame, budget: int, seed: int) -> pd.DataFrame:
    return unc_df.sample(n=budget, random_state=seed)

def select_proportional(unc_df: pd.DataFrame, budget: int, seed: int) -> pd.DataFrame:
    cell_sizes = unc_df.groupby(["domain", "severity"]).size()
    total      = cell_sizes.sum()
    alloc = {cell: max(1, round(n / total * budget)) for cell, n in cell_sizes.items()}
    allocated = sum(alloc.values())
    if allocated > budget:
        for cell in sorted(alloc, key=lambda c: alloc[c], reverse=True):
            if allocated <= budget: break
            if alloc[cell] > 1: alloc[cell] -= 1; allocated -= 1
    elif allocated < budget:
        for cell in sorted(alloc, key=lambda c: alloc[c]):
            if allocated >= budget: break
            alloc[cell] += 1; allocated += 1
    rng = np.random.RandomState(seed)
    selected = []
    for (domain, severity), n_pick in alloc.items():
        pool = unc_df[(unc_df["domain"] == domain) & (unc_df["severity"] == severity)]
        n_pick = min(n_pick, len(pool))
        if n_pick > 0:
            selected.append(pool.sample(n=n_pick, random_state=rng.randint(int(1e9))))
    return pd.concat(selected).drop_duplicates("vignette_id") if selected else pd.DataFrame()

pure_al  = select_pure_al(UNCERTAINTY_DF, AL_BUDGET)
strat_al = select_stratified_al(UNCERTAINTY_DF, AL_BUDGET)
rand_al  = select_random(UNCERTAINTY_DF, AL_BUDGET, seed=MASTER_SEED)
prop_al  = select_proportional(UNCERTAINTY_DF, AL_BUDGET, seed=MASTER_SEED)
aug_al   = select_stratified_al(UNCERTAINTY_DF, AUGMENTED_BUDGET)

for df in [pure_al, strat_al, rand_al, prop_al, aug_al]:
    if "oncology" not in df.columns:
        df["oncology"]   = df["vignette_id"].map(onco_lookup).fillna(False).astype(bool)
        df["onco_score"] = df["vignette_id"].map(score_lookup).fillna(0).astype(int)

methods_dict = {
    "Random": rand_al, "Proportional": prop_al,
    "Pure AL": pure_al, "Stratified AL": strat_al,
}
METHOD_ORDER = ["Random", "Proportional", "Pure AL", "Stratified AL"]

print(f"\n  {'Method':<20s} {'N':>4s} {'Coverage':>10s} {'Domains':>8s} "
      f"{'Mean Unc':>10s} {'Missing Domains'}")
for name in METHOD_ORDER:
    df = methods_dict[name]
    cov, doms, missing = coverage_stats(df)
    mu = df["uncertainty_score"].mean()
    miss_str = ", ".join(sorted(missing)) if missing else "None"
    print(f"  {name:<20s} {len(df):>4d} {cov:>6d}/18   {doms:>4d}/6  "
          f"{mu:>10.4f}   {miss_str}")


# --- §6 PAIRWISE STATISTICAL COMPARISONS ---

LOG("INFO", "§6 Statistical comparisons ...")

strat_cov, strat_doms, _ = coverage_stats(strat_al)

pairwise_stats = {}
for name in ["Random", "Proportional", "Pure AL"]:
    df = methods_dict[name]
    b_cov, _, _ = coverage_stats(df)

    U_raw, p_mw = mannwhitneyu(
        strat_al["uncertainty_score"], df["uncertainty_score"], alternative="two-sided")
    U_comp  = len(strat_al) * len(df) - U_raw
    U_small = min(U_raw, U_comp)
    r_rb    = 1 - (2 * U_small) / (len(strat_al) * len(df))
    delta   = strat_al["uncertainty_score"].mean() - df["uncertainty_score"].mean()

    tf = [[strat_cov, 18 - strat_cov], [b_cov, 18 - b_cov]]
    _, p_fish = fisher_exact(tf, alternative="two-sided")

    pairwise_stats[name] = {
        "coverage": b_cov, "U": U_small, "p_mw": p_mw,
        "r_rb": r_rb, "delta": delta, "p_fisher": p_fish,
    }
    print(f"  vs {name}: cov={strat_cov}/18 vs {b_cov}/18, "
          f"U={U_small:.1f}, p={p_mw:.4f}, r_rb={r_rb:.3f}")

# Primary: Strat vs Pure + bootstrap CI
U_main, p_main = mannwhitneyu(pure_al["uncertainty_score"],
                               strat_al["uncertainty_score"], alternative="two-sided")
U_main_comp  = AL_BUDGET * AL_BUDGET - U_main
U_main_small = min(U_main, U_main_comp)
delta_main   = strat_al["uncertainty_score"].mean() - pure_al["uncertainty_score"].mean()
r_rb_main    = 1 - (2 * U_main_small) / (AL_BUDGET * AL_BUDGET)

pure_cov, pure_doms, pure_missing = coverage_stats(pure_al)
tf_main = [[strat_cov, 18 - strat_cov], [pure_cov, 18 - pure_cov]]
_, p_fisher_main = fisher_exact(tf_main, alternative="two-sided")

rng_ci = np.random.RandomState(MASTER_SEED + 9999)
s_vals = strat_al["uncertainty_score"].values
p_vals = pure_al["uncertainty_score"].values
boot_deltas = np.array([
    rng_ci.choice(s_vals, len(s_vals), replace=True).mean() -
    rng_ci.choice(p_vals, len(p_vals), replace=True).mean()
    for _ in range(5000)
])
ci_lo, ci_hi = np.percentile(boot_deltas, [2.5, 97.5])

LOG("OK", f"Primary: D={delta_main:+.4f}, 95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}], "
    f"U={U_main_small:.1f}, p={p_main:.4f}, r_rb={r_rb_main:.3f}, "
    f"Fisher p={p_fisher_main:.4f}")


# --- §7 MULTI-SEED ROBUSTNESS ---

LOG("INFO", f"§7 Robustness ({N_ROBUSTNESS_SEEDS} seeds) ...")

t0 = time.time()
robustness_rows = []

for trial in range(N_ROBUSTNESS_SEEDS):
    if trial % 20 == 0:
        elapsed = time.time() - t0
        eta = (elapsed / max(trial, 1)) * (N_ROBUSTNESS_SEEDS - trial)
        print(f"  seed {trial}/{N_ROBUSTNESS_SEEDS} "
              f"(elapsed {elapsed:.0f}s, ETA {eta:.0f}s)", end="\r")

    unc_t = compute_uncertainty(seed=trial, n_boot=50,
                                w_std=w_std, w_ent=w_ent, verbose=False)
    pure_t  = select_pure_al(unc_t, AL_BUDGET)
    strat_t = select_stratified_al(unc_t, AL_BUDGET)
    rand_t  = select_random(unc_t, AL_BUDGET, seed=trial)
    prop_t  = select_proportional(unc_t, AL_BUDGET, seed=trial)

    p_cov, _, _  = coverage_stats(pure_t)
    s_cov, _, _  = coverage_stats(strat_t)
    r_cov, _, _  = coverage_stats(rand_t)
    pr_cov, _, _ = coverage_stats(prop_t)

    robustness_rows.append({
        "seed": trial,
        "pure_coverage": p_cov, "strat_coverage": s_cov,
        "random_coverage": r_cov, "prop_coverage": pr_cov,
        "pure_mean_unc": pure_t["uncertainty_score"].mean(),
        "strat_mean_unc": strat_t["uncertainty_score"].mean(),
        "random_mean_unc": rand_t["uncertainty_score"].mean(),
        "prop_mean_unc": prop_t["uncertainty_score"].mean(),
        "delta_strat_pure": strat_t["uncertainty_score"].mean() - pure_t["uncertainty_score"].mean(),
        "coverage_gain": s_cov - p_cov,
    })

print(f"\n  Robustness complete ({time.time() - t0:.1f}s)")

ROBUST_DF = pd.DataFrame(robustness_rows)
pct_strat_full  = (ROBUST_DF["strat_coverage"] == 18).mean() * 100
pct_pure_full   = (ROBUST_DF["pure_coverage"] == 18).mean() * 100
pct_rand_full   = (ROBUST_DF["random_coverage"] == 18).mean() * 100
pct_prop_full   = (ROBUST_DF["prop_coverage"] == 18).mean() * 100
pct_strat_beats = (ROBUST_DF["coverage_gain"] > 0).mean() * 100

LOG("OK", f"18/18 rate: Strat={pct_strat_full:.0f}%, Pure={pct_pure_full:.0f}%, "
    f"Rand={pct_rand_full:.0f}%, Prop={pct_prop_full:.0f}%")
LOG("OK", f"Strat beats Pure: {pct_strat_beats:.1f}% of seeds")


# --- §8 BUDGET-COVERAGE FRONTIER ---

LOG("INFO", f"§8 Budget frontier ({BUDGET_RANGE[0]}-{BUDGET_RANGE[-1]}) ...")

frontier_rows = []
for budget in BUDGET_RANGE:
    for method_name, selector in [
        ("Pure AL",       lambda u, b: select_pure_al(u, b)),
        ("Stratified AL", lambda u, b: select_stratified_al(u, b)),
    ]:
        sel = selector(UNCERTAINTY_DF, budget)
        cov, _, _ = coverage_stats(sel)
        frontier_rows.append({
            "budget": budget, "method": method_name,
            "coverage": cov, "mean_unc": sel["uncertainty_score"].mean(),
            "n_selected": len(sel),
        })

FRONTIER_DF = pd.DataFrame(frontier_rows)

strat_full = FRONTIER_DF[(FRONTIER_DF["method"] == "Stratified AL") & (FRONTIER_DF["coverage"] == 18)]
pure_full  = FRONTIER_DF[(FRONTIER_DF["method"] == "Pure AL") & (FRONTIER_DF["coverage"] == 18)]
min_strat = int(strat_full["budget"].min()) if len(strat_full) > 0 else ">120"
min_pure  = int(pure_full["budget"].min()) if len(pure_full) > 0 else ">120"
LOG("OK", f"Min budget for 18/18: Strat={min_strat}, Pure={min_pure}")


# --- §9 RANK STABILITY ---

LOG("INFO", f"§9 Rank stability ({N_RANK_SEEDS} seeds) ...")

rank_arrays = []
for rs in range(N_RANK_SEEDS):
    if rs % 10 == 0:
        print(f"  rank seed {rs}/{N_RANK_SEEDS}", end="\r")
    unc_r = compute_uncertainty(seed=rs, n_boot=50,
                                w_std=w_std, w_ent=w_ent, verbose=False)
    rank_arrays.append(unc_r["uncertainty_score"].rank(ascending=False).values)

print(f"\n  Computing pairwise correlations ...")

tau_values, rho_values, jaccard_values = [], [], []
top60_sets = [set(np.argsort(ra)[:60]) for ra in rank_arrays]

for i in range(len(rank_arrays)):
    for j in range(i + 1, len(rank_arrays)):
        tau, _ = kendalltau(rank_arrays[i], rank_arrays[j])
        rho, _ = spearmanr(rank_arrays[i], rank_arrays[j])
        tau_values.append(tau)
        rho_values.append(rho)
        inter = len(top60_sets[i] & top60_sets[j])
        union = len(top60_sets[i] | top60_sets[j])
        jaccard_values.append(inter / union if union > 0 else 0)

tau_arr     = np.array(tau_values)
rho_arr     = np.array(rho_values)
jaccard_arr = np.array(jaccard_values)

RANK_DF = pd.DataFrame({
    "kendall_tau": tau_values,
    "spearman_rho": rho_values,
    "top60_jaccard": jaccard_values,
})

LOG("OK", f"Kendall tau: {tau_arr.mean():.4f} +/- {tau_arr.std():.4f}")
LOG("OK", f"Spearman rho: {rho_arr.mean():.4f} +/- {rho_arr.std():.4f}")
LOG("OK", f"Top-60 Jaccard: {jaccard_arr.mean():.4f} +/- {jaccard_arr.std():.4f}")


# --- §10 SENSITIVITY ANALYSIS ---

LOG("INFO", "§10 Sensitivity analysis ...")

sensitivity_rows = []
for scheme_name, (ws, we) in WEIGHT_SCHEMES.items():
    unc_s = UNCERTAINTY_DF.copy()
    std_max = unc_s["pred_std"].max()
    ent_max = unc_s["pred_entropy"].max()
    unc_s["uncertainty_score"] = (
        ws * (unc_s["pred_std"] / std_max if std_max > 0 else 0) +
        we * (unc_s["pred_entropy"] / ent_max if ent_max > 0 else 0)
    )
    pure_s  = select_pure_al(unc_s, AL_BUDGET)
    strat_s = select_stratified_al(unc_s, AL_BUDGET)
    p_cov, _, _ = coverage_stats(pure_s)
    s_cov, _, _ = coverage_stats(strat_s)

    sensitivity_rows.append({
        "scheme": scheme_name, "w_std": ws, "w_entropy": we,
        "pure_coverage": p_cov, "strat_coverage": s_cov,
        "pure_mean": pure_s["uncertainty_score"].mean(),
        "strat_mean": strat_s["uncertainty_score"].mean(),
        "delta": strat_s["uncertainty_score"].mean() - pure_s["uncertainty_score"].mean(),
        "coverage_gain": s_cov - p_cov,
    })
    marker = " <- PRIMARY" if scheme_name == PRIMARY_WEIGHTS else ""
    print(f"  {scheme_name:.<20} pure={p_cov}/18  strat={s_cov}/18{marker}")

SENSITIVITY_DF = pd.DataFrame(sensitivity_rows)
all_strat_wins = all(r["coverage_gain"] > 0 for r in sensitivity_rows)
LOG("OK", f"Strat wins all {len(WEIGHT_SCHEMES)} schemes: {'YES' if all_strat_wins else 'NO'}")


# --- §11 ONCOLOGY STATISTICS ---

LOG("INFO", "§11 Oncology statistics ...")

onco_grp    = UNCERTAINTY_DF[UNCERTAINTY_DF["oncology"] == True]["uncertainty_score"]
notonco_grp = UNCERTAINTY_DF[UNCERTAINTY_DF["oncology"] == False]["uncertainty_score"]

onco_mean, notonco_mean = onco_grp.mean(), notonco_grp.mean()
onco_sd,   notonco_sd   = onco_grp.std(),  notonco_grp.std()
mean_diff = onco_mean - notonco_mean

if len(onco_grp) > 1 and len(notonco_grp) > 1:
    U_onco, p_onco = mannwhitneyu(onco_grp, notonco_grp, alternative="two-sided")
    r_rb_onco = U_onco / (len(onco_grp) * len(notonco_grp))
else:
    U_onco, p_onco, r_rb_onco = np.nan, np.nan, np.nan

rng_onco = np.random.default_rng(1337)
boot_onco_diffs = np.array([
    rng_onco.choice(onco_grp.values, len(onco_grp), replace=True).mean() -
    rng_onco.choice(notonco_grp.values, len(notonco_grp), replace=True).mean()
    for _ in range(5000)
])
onco_ci_lo, onco_ci_hi = np.percentile(boot_onco_diffs, [2.5, 97.5])

LOG("OK", f"Onco (n={len(onco_grp)}): mean={onco_mean:.4f}, SD={onco_sd:.4f}")
LOG("OK", f"Non-onco (n={len(notonco_grp)}): mean={notonco_mean:.4f}, SD={notonco_sd:.4f}")
LOG("OK", f"D={mean_diff:+.4f}, 95% CI [{onco_ci_lo:.4f}, {onco_ci_hi:.4f}], "
    f"U={U_onco:.0f}, p={p_onco:.4f}")

method_onco_repr = {}
for mname in METHOD_ORDER:
    mdf = methods_dict[mname]
    n_o = int(mdf["oncology"].sum())
    method_onco_repr[mname] = {
        "n_onco": n_o, "n_total": len(mdf),
        "pct": round(n_o / len(mdf) * 100, 1),
    }

domain_onco_stats = []
for dom in ALL_DOMAINS:
    sub = UNCERTAINTY_DF[UNCERTAINTY_DF["domain"] == dom]
    o_sub = sub[sub["oncology"]]
    n_sub = sub[~sub["oncology"]]
    row = {
        "domain": dom, "n": len(sub),
        "n_onco": int(len(o_sub)), "n_non": int(len(n_sub)),
        "pct_onco": len(o_sub) / len(sub) * 100,
        "mean_unc_onco": float(o_sub["uncertainty_score"].mean()) if len(o_sub) > 0 else np.nan,
        "mean_unc_non": float(n_sub["uncertainty_score"].mean()) if len(n_sub) > 0 else np.nan,
    }
    if len(o_sub) > 1 and len(n_sub) > 1:
        U_d, p_d = mannwhitneyu(o_sub["uncertainty_score"],
                                 n_sub["uncertainty_score"], alternative="two-sided")
        row["U"] = float(U_d)
        row["p"] = float(p_d)
    else:
        row["U"] = np.nan
        row["p"] = np.nan
    domain_onco_stats.append(row)


# --- §12 SAVE ALL OUTPUTS ---

LOG("INFO", "§12 Saving outputs ...")

UNCERTAINTY_DF.to_csv(ADV / "uncertainty_scores.csv",           index=False)
rand_al.to_csv(       ADV / "random_al_selection.csv",          index=False)
prop_al.to_csv(       ADV / "proportional_al_selection.csv",    index=False)
pure_al.to_csv(       ADV / "pure_al_selection.csv",            index=False)
strat_al.to_csv(      ADV / "stratified_al_selection.csv",      index=False)
aug_al.to_csv(        ADV / "augmented_al_selection.csv",       index=False)
ROBUST_DF.to_csv(     ADV / "robustness_multi_seed.csv",        index=False)
SENSITIVITY_DF.to_csv(ADV / "sensitivity_weights.csv",          index=False)
FRONTIER_DF.to_csv(   ADV / "budget_frontier.csv",              index=False)
RANK_DF.to_csv(       ADV / "rank_stability.csv",               index=False)

for fname in sorted(ADV.glob("*.csv")):
    n = len(pd.read_csv(fname))
    LOG("OK", f"  {fname.name:<42s}  {n:>5} rows")

# Validation pack
validation_pack = VIG_DF[VIG_DF["vignette_id"].isin(strat_al["vignette_id"])].copy()
validation_pack["rater_id"] = ""
for item in SMS_ITEMS:
    validation_pack[f"rater_req__{item}"] = ""
validation_pack = validation_pack.merge(
    UNCERTAINTY_DF[["vignette_id", "uncertainty_score", "pred_std",
                    "pred_entropy", "oncology", "onco_score"]],
    on="vignette_id", how="left")
keep_cols = ["vignette_id", "domain", "severity", "language", "scenario_text",
             "uncertainty_score", "pred_std", "pred_entropy",
             "oncology", "onco_score", "rater_id"] + \
            [f"rater_req__{item}" for item in SMS_ITEMS]
keep_cols = [c for c in keep_cols if c in validation_pack.columns]
val_path = DIRS["rater_forms"] / "validation_stratified_al.csv"
validation_pack[keep_cols].to_csv(val_path, index=False)
LOG("OK", f"  Validation pack -> {val_path.name}")

# Summary JSON
summary = {
    "script_version": "07_v4_unified",
    "timestamp": datetime.now().isoformat(),
    "master_seed": MASTER_SEED,
    "n_bootstrap": N_BOOTSTRAP,
    "n_robustness_seeds": N_ROBUSTNESS_SEEDS,
    "n_rank_seeds": N_RANK_SEEDS,
    "weight_scheme": PRIMARY_WEIGHTS,
    "total_vignettes": len(VIG_DF),
    "uncertainty": {
        "mean": float(UNCERTAINTY_DF["uncertainty_score"].mean()),
        "std": float(UNCERTAINTY_DF["uncertainty_score"].std()),
        "min": float(UNCERTAINTY_DF["uncertainty_score"].min()),
        "max": float(UNCERTAINTY_DF["uncertainty_score"].max()),
    },
    "primary_comparison": {
        "pure_al_coverage": pure_cov,
        "strat_al_coverage": strat_cov,
        "pure_al_mean_unc": float(pure_al["uncertainty_score"].mean()),
        "strat_al_mean_unc": float(strat_al["uncertainty_score"].mean()),
        "delta": float(delta_main),
        "delta_95ci": [float(ci_lo), float(ci_hi)],
        "mann_whitney_U": float(U_main_small),
        "mann_whitney_P": float(p_main),
        "rank_biserial": float(r_rb_main),
        "fisher_P": float(p_fisher_main),
        "pure_domains_missed": sorted(pure_missing),
    },
    "pairwise_comparisons": {
        name: {
            "coverage": stats["coverage"],
            "U": float(stats["U"]), "p_mw": float(stats["p_mw"]),
            "r_rb": float(stats["r_rb"]), "delta": float(stats["delta"]),
            "p_fisher": float(stats["p_fisher"]),
        }
        for name, stats in pairwise_stats.items()
    },
    "robustness": {
        "n_seeds": N_ROBUSTNESS_SEEDS,
        "pct_strat_full_cov": float(pct_strat_full),
        "pct_pure_full_cov": float(pct_pure_full),
        "pct_random_full_cov": float(pct_rand_full),
        "pct_prop_full_cov": float(pct_prop_full),
        "mean_coverage_gain": float(ROBUST_DF["coverage_gain"].mean()),
        "std_coverage_gain": float(ROBUST_DF["coverage_gain"].std()),
        "mean_delta": float(ROBUST_DF["delta_strat_pure"].mean()),
        "pct_strat_beats_pure": float(pct_strat_beats),
    },
    "rank_stability": {
        "n_seeds": N_RANK_SEEDS,
        "kendall_tau": {"mean": float(tau_arr.mean()), "std": float(tau_arr.std()),
                        "min": float(tau_arr.min()), "max": float(tau_arr.max())},
        "spearman_rho": {"mean": float(rho_arr.mean()), "std": float(rho_arr.std()),
                         "min": float(rho_arr.min()), "max": float(rho_arr.max())},
        "top60_jaccard": {"mean": float(jaccard_arr.mean()), "std": float(jaccard_arr.std()),
                          "min": float(jaccard_arr.min())},
    },
    "sensitivity": {
        "all_schemes_strat_wins": all_strat_wins,
        "schemes_tested": list(WEIGHT_SCHEMES.keys()),
    },
    "budget_frontier": {
        "min_budget_strat_full_cov": min_strat,
        "min_budget_pure_full_cov": min_pure,
    },
    "oncology": {
        "classification_method": "deterministic_two_tier_lexicon",
        "threshold": ONCO_THRESHOLD,
        "corpus_rate": round(raw_rate, 4),
        "n_onco": int(n_onco_total),
        "n_total": len(vignettes_raw),
        "stats": {
            "n_onco_grp": int(len(onco_grp)),
            "n_notonco_grp": int(len(notonco_grp)),
            "mean_onco": round(onco_mean, 4),
            "mean_notonco": round(notonco_mean, 4),
            "sd_onco": round(onco_sd, 4),
            "sd_notonco": round(notonco_sd, 4),
            "diff": round(mean_diff, 4),
            "ci_lo": round(onco_ci_lo, 4),
            "ci_hi": round(onco_ci_hi, 4),
            "U": round(float(U_onco), 1) if not np.isnan(U_onco) else None,
            "p": round(float(p_onco), 6) if not np.isnan(p_onco) else None,
            "r_rb": round(float(r_rb_onco), 3) if not np.isnan(r_rb_onco) else None,
        },
        "method_representation": method_onco_repr,
        "domain_breakdown": domain_onco_stats,
    },
    "all_domains": ALL_DOMAINS,
    "all_severities": ALL_SEVERITIES,
    "method_order": METHOD_ORDER,
    "al_budget": AL_BUDGET,
    "dpi_recommended": 1200,
}

summary_path = ADV / "validation_summary.json"
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
LOG("OK", f"  Summary JSON -> {summary_path.name}")


# --- §13 FINAL REPORT ---

LOG("OK", "Script 07 v4 COMPLETE")
print(f"""
  PRIMARY RESULTS (N={AL_BUDGET}, seed={MASTER_SEED}):
    Pure AL:       coverage {pure_cov}/18, mean unc = {pure_al['uncertainty_score'].mean():.4f}
    Stratified AL: coverage {strat_cov}/18, mean unc = {strat_al['uncertainty_score'].mean():.4f}
    D = {delta_main:+.4f}, 95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]
    MW U = {U_main_small:.0f}, p = {p_main:.4f}, r_rb = {r_rb_main:.3f}
    Fisher p = {p_fisher_main:.4f}

  ROBUSTNESS ({N_ROBUSTNESS_SEEDS} seeds):
    18/18 rate: Strat={pct_strat_full:.0f}%, Pure={pct_pure_full:.0f}%
    Mean gain: +{ROBUST_DF['coverage_gain'].mean():.1f} cells

  RANK STABILITY:
    Kendall tau = {tau_arr.mean():.3f} +/- {tau_arr.std():.3f}
    Spearman rho = {rho_arr.mean():.3f} +/- {rho_arr.std():.3f}
    Top-60 Jaccard = {jaccard_arr.mean():.3f} +/- {jaccard_arr.std():.3f}

  BUDGET FRONTIER:
    Min budget for 18/18: Strat={min_strat}, Pure={min_pure}

  ONCOLOGY:
    Rate: {n_onco_total}/{len(vignettes_raw)} ({raw_rate*100:.1f}%), threshold={ONCO_THRESHOLD}
    Onco mean={onco_mean:.3f}, Non-onco mean={notonco_mean:.3f}
    D={mean_diff:+.3f}, p={p_onco:.4f}
    Strat AL onco: {strat_al['oncology'].sum()}/{len(strat_al)} ({strat_al['oncology'].sum()/len(strat_al)*100:.1f}%) vs corpus {raw_rate*100:.1f}%

  SENSITIVITY: Strat wins all {len(WEIGHT_SCHEMES)} schemes: {'YES' if all_strat_wins else 'NO'}

  All outputs -> {ADV}/
""")
