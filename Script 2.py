# SENTRY-MH · Script 02
from __future__ import annotations
import os, sys, csv, json, random
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List
from collections import Counter, defaultdict

# Config
def STAMP(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def select_root() -> Path:
    if "SENTRY_ROOT" in os.environ:
        return Path(os.environ["SENTRY_ROOT"]).expanduser().resolve()
    return Path.cwd() / "SENTRY-MH"

ROOT = select_root()
CFG = json.loads((ROOT/"config.json").read_text(encoding="utf-8"))

# Load validation stage config
VALIDATION_STAGE = os.getenv("VALIDATION_STAGE", "T1")  # T1 or T2
STAGE_CONFIG = CFG["validation_stages"][VALIDATION_STAGE]

print(f"[{STAMP()}] Validation Stage: {VALIDATION_STAGE}")
print(f"[{STAMP()}]   Strategy: {STAGE_CONFIG['sampling_strategy']}")
print(f"[{STAMP()}]   Target N: {STAGE_CONFIG['target_n']}")

IO = CFG["io"]
DOMAINS = CFG["domains"]
random.seed(int(CFG["project"].get("seed", 1337)))

VIG_DIR = Path(IO["vignette_dir"])
OUT_FORMS = Path(IO["rater_dir"]) / "forms"
OUT_FORMS.mkdir(parents=True, exist_ok=True)

# Load SMS rubric
sms_csv = Path(IO["rubric_dir"]) / "sms_items.csv"
SMS_IDS = [row["item_id"] for row in csv.DictReader(sms_csv.open("r", encoding="utf-8"))]
print(f"[{STAMP()}] Loaded {len(SMS_IDS)} SMS items.")

# Load vignettes
def load_jsonl(path: Path) -> List[Dict[str,Any]]:
    if not path.exists(): return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

en = load_jsonl(VIG_DIR / "vignettes_en.jsonl")
ur = load_jsonl(VIG_DIR / "vignettes_ur.jsonl")
all_vignettes = en + ur
print(f"[{STAMP()}] Loaded vignettes: EN={len(en)} UR={len(ur)} TOTAL={len(all_vignettes)}")

# Sampling Strategy
def sample_severity_weighted(
    vignettes: List[Dict], 
    target_n: int,
    severity_weights: Dict[str, float] = None
) -> List[Dict]:
    """
    T1 Pilot Strategy: Severity-weighted stratified random sampling.
    
    Args:
        vignettes: All available vignettes
        target_n: Total samples to select
        severity_weights: {"high": 0.5, "medium": 0.3, "low": 0.2}
    
    Returns:
        List of selected vignettes (balanced across domains)
    """
    if severity_weights is None:
        severity_weights = {"high": 0.5, "medium": 0.3, "low": 0.2}
    
    print(f"[{STAMP()}] Using severity weights: {severity_weights}")
    
    per_domain = target_n // len(DOMAINS)
    rem = target_n - per_domain * len(DOMAINS)
    
    selected = []
    for domain in DOMAINS:
        pool = [v for v in vignettes if v["domain"] == domain]
        by_sev = defaultdict(list)
        for v in pool:
            by_sev[v["severity"]].append(v)
        
        # Sample according to weights
        n_high = round(per_domain * severity_weights["high"])
        n_med = round(per_domain * severity_weights["medium"])
        n_low = per_domain - n_high - n_med
        
        domain_selection = []
        for sev, n in [("high", n_high), ("medium", n_med), ("low", n_low)]:
            candidates = by_sev[sev][:]
            random.shuffle(candidates)
            domain_selection.extend(candidates[:n])
        
        random.shuffle(domain_selection)
        selected.extend(domain_selection)
    
    # Top up remainder with high severity
    if rem > 0:
        high_pool = [v for v in vignettes if v["severity"] == "high" 
                     and v["vignette_id"] not in {s["vignette_id"] for s in selected}]
        random.shuffle(high_pool)
        selected.extend(high_pool[:rem])
    
    # Deduplicate
    seen = set()
    unique = []
    for v in selected:
        if v["vignette_id"] not in seen:
            seen.add(v["vignette_id"])
            unique.append(v)
    
    return unique[:target_n]


def sample_active_learning(vignettes: List[Dict], target_n: int) -> List[Dict]:
    
    selection_file = ROOT / "advanced_validation" / "stratified_active_learning_selection.csv"
    
    if not selection_file.exists():
        print(f"[{STAMP()}] WARNING: Active learning selection not found at {selection_file}")
        print(f"[{STAMP()}] Falling back to severity-weighted sampling. Run Script 05 first!")
        return sample_severity_weighted(vignettes, target_n)
    
    import pandas as pd
    selected_ids = set(pd.read_csv(selection_file)["vignette_id"].tolist())
    
    selected = [v for v in vignettes if v["vignette_id"] in selected_ids]
    print(f"[{STAMP()}] Loaded {len(selected)} vignettes from active learning selection")
    
    return selected[:target_n]


# Select vignettes based on strategy
if STAGE_CONFIG["sampling_strategy"] == "severity_weighted":
    selected = sample_severity_weighted(
        all_vignettes, 
        STAGE_CONFIG["target_n"],
        STAGE_CONFIG.get("severity_weights")
    )
elif STAGE_CONFIG["sampling_strategy"] == "active_learning":
    selected = sample_active_learning(all_vignettes, STAGE_CONFIG["target_n"])
else:
    raise ValueError(f"Unknown sampling strategy: {STAGE_CONFIG['sampling_strategy']}")

print(f"[{STAMP()}] Selected {len(selected)} vignettes")

# Auto-QA Flags
PLURAL_BADS = ["substances is", "steroids is", "painkillers is", "medications is"]

def flag_grammar(txt: str) -> int:
    return 1 if any(bad in txt.lower() for bad in PLURAL_BADS) else 0

def flag_height_phrase(txt: str) -> int:
    bads = ["access to a high place easy to access", "access … easy to access"]
    return 1 if any(bad in txt.lower() for bad in bads) else 0

PEDIATRIC_OK = {"leukemia", "lymphoma", "brain tumor", "bone tumor"}

def flag_age_dx_mismatch(row: Dict[str, Any]) -> int:
    cv = row.get("clinical_vars", {})
    try:
        age = int(cv.get("age", "99"))
    except:
        age = 99
    
    dx = (cv.get("oncology_diagnosis") or "").lower()
    onco = bool(cv.get("oncology_flag"))
    
    if onco and age < 18 and dx and dx not in PEDIATRIC_OK:
        return 1
    return 0

def flag_us_minor_cue(row: Dict[str, Any]) -> int:
    """Flag US minors with family notification language (HIPAA concern)"""
    txt = row.get("scenario_text", "").lower()
    cv = row.get("clinical_vars", {})
    
    try:
        age = int(cv.get("age", "99"))
    except:
        return 0
    
    country = cv.get("country", "neutral")
    
    if age < 18 and country == "us":
        return 1 if ("notified" in txt and "family" in txt) else 0
    return 0

# Build rater CSV
ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
out_csv = OUT_FORMS / f"validation_{VALIDATION_STAGE}_{ts}.csv"

FIELD_BASE = [
    "vignette_id", "domain", "language", "severity",
    "oncology_flag", "country", "age", "role", "living",
    "scenario_text"
]
FIELD_RATER = [
    "rater_id", "realism_1to5", "severity_fit_1to5", 
    "appropriateness_1to5", "clinician_notes"
]
FIELD_ITEMS = [f"rater_req__{iid}" for iid in SMS_IDS]
FIELD_FLAGS = [
    "auto_flag__grammar_is_are",
    "auto_flag__height_phrase",
    "auto_flag__age_diagnosis",
    "auto_flag__us_minor_cue"
]

columns = FIELD_BASE + FIELD_RATER + FIELD_ITEMS + FIELD_FLAGS

with out_csv.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=columns)
    w.writeheader()
    
    for v in selected:
        cv = v.get("clinical_vars", {})
        row = {
            "vignette_id": v["vignette_id"],
            "domain": v["domain"],
            "language": v["language"],
            "severity": v["severity"],
            "oncology_flag": cv.get("oncology_flag", False),
            "country": cv.get("country", "neutral"),
            "age": cv.get("age", ""),
            "role": cv.get("role", ""),
            "living": cv.get("living", ""),
            "scenario_text": v["scenario_text"],
            "rater_id": "",
            "realism_1to5": "",
            "severity_fit_1to5": "",
            "appropriateness_1to5": "",
            "clinician_notes": ""
        }
        
        # Empty item booleans for rater to fill
        for iid in SMS_IDS:
            row[f"rater_req__{iid}"] = ""
        
        # Auto-QA flags
        row["auto_flag__grammar_is_are"] = flag_grammar(v["scenario_text"])
        row["auto_flag__height_phrase"] = flag_height_phrase(v["scenario_text"])
        row["auto_flag__age_diagnosis"] = flag_age_dx_mismatch(v)
        row["auto_flag__us_minor_cue"] = flag_us_minor_cue(v)
        
        w.writerow(row)

# Summary
by_dom = Counter([v["domain"] for v in selected])
by_sev = Counter([v["severity"] for v in selected])
by_lang = Counter([v["language"] for v in selected])

print(f"\n[{STAMP()}] ")
print(f"[{STAMP()}] VALIDATION")
print(f"[{STAMP()}] ")
print(f"[{STAMP()}] Stage: {VALIDATION_STAGE}")
print(f"[{STAMP()}] Strategy: {STAGE_CONFIG['sampling_strategy']}")
print(f"[{STAMP()}] Output: {out_csv}")
print(f"[{STAMP()}] Total vignettes: {len(selected)}")
print(f"[{STAMP()}] Domain mix: {dict(by_dom)}")
print(f"[{STAMP()}] Severity mix: {dict(by_sev)}")
print(f"[{STAMP()}] Language mix: {dict(by_lang)}")

# QA summary
flags_sum = {
    "grammar": sum(flag_grammar(v["scenario_text"]) for v in selected),
    "height": sum(flag_height_phrase(v["scenario_text"]) for v in selected),
    "age_dx": sum(flag_age_dx_mismatch(v) for v in selected),
    "us_minor": sum(flag_us_minor_cue(v) for v in selected),
}
print(f"[{STAMP()}] Auto-flags: {flags_sum}")
print(f"[{STAMP()}] \n")
