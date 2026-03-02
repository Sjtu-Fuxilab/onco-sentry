# Scipt 06

import os, json, glob, textwrap
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Config
SENTRY_ROOT = os.getenv("SENTRY_ROOT", r"D:\Data")
ROOT = Path(SENTRY_ROOT)
FORMS_DIR = ROOT / "rater" / "forms"
TABLES_DIR = ROOT / "tables"; TABLES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = ROOT / "reports"; REPORTS_DIR.mkdir(parents=True, exist_ok=True)

FAST_MODE = True 

# bootstrap sizes
if FAST_MODE:
    N_BOOT_GLOBAL = 400
    N_BOOT_ITEM   = 200
    N_BOOT_LANG   = 300
    N_BOOT_DOM    = 150
    N_BOOT_SEV    = 150
else:
    N_BOOT_GLOBAL = 2000
    N_BOOT_ITEM   = 1200
    N_BOOT_LANG   = 1500
    N_BOOT_DOM    = 800
    N_BOOT_SEV    = 800

def STAMP(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# File helpers
def latest_base_csv():
    hits = sorted(FORMS_DIR.glob("validation_T2_*.csv"), key=lambda p: p.stat().st_mtime)
    if not hits: raise FileNotFoundError("No T2 base CSV under rater/forms")
    return hits[-1]

def latest_per_rater_dir():
    outs = sorted(FORMS_DIR.glob("outgoing_T2_*"), key=lambda p: p.stat().st_mtime)
    if not outs: raise FileNotFoundError("No outgoing_T2_* folder under rater/forms")
    d = outs[-1] / "per_rater"
    if not d.exists(): raise FileNotFoundError(f"per_rater missing under {outs[-1]}")
    return d

# Column cleaning
PREFIXES = ("rater_req__", "final__", "adj__", "expected__")

def _is_item_col(c):
    return any(c.startswith(p) for p in PREFIXES)

def _base_item_name(c):
    for p in PREFIXES:
        if c.startswith(p):
            core = c[len(p):]
            if core.endswith("_x") or core.endswith("_y"):
                core = core[:-2]
            return p + core
    return c

def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Coalesce *_x/*_y duplicates into a single boolean column per item."""
    df = df.copy()
    # Build groups by canonical name
    groups = {}
    for c in df.columns:
        if _is_item_col(c):
            canon = _base_item_name(c)
            groups.setdefault(canon, []).append(c)

    # Coalesce groups
    for canon, cols in groups.items():
        if len(cols) == 1 and cols[0] == canon:
            # ensure boolean
            df[canon] = df[canon].fillna(False).astype(str).str.lower().isin(("1","true","t","yes","y"))
            continue
        # combine multiple sources
        vals = None
        for c in cols:
            v = df[c]
            if vals is None:
                vals = v
            else:
                vals = vals.combine_first(v)
        vals = vals.fillna(False).astype(str).str.lower().isin(("1","true","t","yes","y"))
        df[canon] = vals
        # drop the extra variants
        for c in cols:
            if c != canon and c in df.columns:
                df.drop(columns=[c], inplace=True, errors="ignore")

    # Finally, drop any lingering *_x/*_y columns not caught above
    drop_me = [c for c in df.columns if c.endswith("_x") or c.endswith("_y")]
    if drop_me:
        df.drop(columns=drop_me, inplace=True, errors="ignore")
    return df

def load_per_rater_frames(per_rater_dir: Path):
    files = sorted([p for p in per_rater_dir.glob("*.csv") if "validation_T2_" in p.name])
    if not files: raise FileNotFoundError(f"No per-rater CSVs in {per_rater_dir}")
    frames, raters = [], []
    for p in files:
        df = pd.read_csv(p, encoding="utf-8")
        df = clean_frame(df)
        rid = p.stem.replace("validation_T2_", "")
        df["rater_id"] = rid
        frames.append(df); raters.append(rid)
    return raters, frames

def infer_items_from_df(df):
    items = []
    for c in df.columns:
        if _is_item_col(c):
            # ensure no suffix
            if c.endswith("_x") or c.endswith("_y"): 
                continue
            # keep canonical
            items.append(c.split("__",1)[1])
    return sorted(set(items))

def common_items(frames):
    sets = [set(infer_items_from_df(df)) for df in frames]
    common = set.intersection(*sets) if sets else set()
    return sorted(common)

# Metrics
def cohen_kappa(y1, y2): return float(cohen_kappa_score(y1, y2))

def gwet_ac1_binary(y1, y2):
    y1 = np.asarray(y1, int); y2 = np.asarray(y2, int)
    if y1.size == 0: return np.nan
    po = np.mean(y1 == y2)
    pbar = 0.5*(np.mean(y1==1) + np.mean(y2==1))
    pe = 2*pbar*(1-pbar)
    if (1-pe) <= 0: return np.nan
    return (po - pe) / (1 - pe)

def pairwise_arrays(df_sub):
    pairs = {}
    raters = sorted(df_sub["rater_id"].unique())
    units = df_sub[["vignette_id","item_id"]].drop_duplicates()
    for i in range(len(raters)):
        for j in range(i+1, len(raters)):
            r1, r2 = raters[i], raters[j]
            a = df_sub[df_sub["rater_id"]==r1][["vignette_id","item_id","label"]]
            b = df_sub[df_sub["rater_id"]==r2][["vignette_id","item_id","label"]]
            m = units.merge(a, on=["vignette_id","item_id"], how="left") \
                     .merge(b, on=["vignette_id","item_id"], how="left", suffixes=("_a","_b"))
            m = m.dropna(subset=["label_a","label_b"])
            pairs[(r1,r2)] = (m["label_a"].astype(int).values, m["label_b"].astype(int).values)
    return pairs

def compute_pairwise_stats(df_sub):
    pairs = pairwise_arrays(df_sub)
    out = {}
    for (r1,r2),(y1,y2) in pairs.items():
        if len(y1)==0:
            out[(r1,r2)] = (np.nan, np.nan, np.nan)
        else:
            k = cohen_kappa(y1,y2); ac1 = gwet_ac1_binary(y1,y2); pa = float(np.mean(y1==y2))
            out[(r1,r2)] = (k, ac1, pa)
    ks  = [v[0] for v in out.values() if pd.notna(v[0])]
    acs = [v[1] for v in out.values() if pd.notna(v[1])]
    return (float(np.mean(ks)) if ks else np.nan,
            float(np.mean(acs)) if acs else np.nan,
            out)

def bootstrap_ci_stat(df_sub, stat_fn, n_boot, seed):
    if n_boot <= 0: return (stat_fn(df_sub), np.nan, np.nan)
    rng = np.random.default_rng(seed)
    v_ids = df_sub["vignette_id"].dropna().unique().tolist()
    if len(v_ids) < 2: return (np.nan, np.nan, np.nan)
    stats=[]
    for _ in range(n_boot):
        samp = rng.choice(v_ids, size=len(v_ids), replace=True)
        boot = df_sub[df_sub["vignette_id"].isin(samp)]
        stats.append(stat_fn(boot))
    stats = np.array(stats, float)
    return float(np.mean(stats)), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))

def stat_kappa_mean(df_sub): return compute_pairwise_stats(df_sub)[0]
def stat_ac1_mean(df_sub):   return compute_pairwise_stats(df_sub)[1]

# Long votes assembly
def build_long_votes(frames, items, base_csv):
    # start from per-rater; ensure meta present from base
    base = pd.read_csv(base_csv, encoding="utf-8")[["vignette_id","language","domain","severity"]].drop_duplicates("vignette_id")
    longs=[]
    for df in frames:
        rid = df["rater_id"].iloc[0]
        if "vignette_id" not in df: raise ValueError(f"{rid}: missing vignette_id")
        df_meta = df.merge(base, on="vignette_id", how="left", suffixes=("","_base"))
        for it in items:
            # look for columns by prefix priority
            col=None
            for p in PREFIXES:
                name = f"{p}{it}"
                if name in df_meta.columns:
                    col = name; break
            if col is None: continue
            chunk = df_meta[["vignette_id","language","domain","severity", col]].copy()
            chunk.rename(columns={col:"label"}, inplace=True)
            chunk["item_id"]=it; chunk["rater_id"]=rid
            # normalize labels -> bool
            if chunk["label"].dtype==object:
                chunk["label"]=chunk["label"].astype(str).str.lower().isin(("1","true","t","yes","y"))
            else:
                chunk["label"]=chunk["label"].fillna(False).astype(bool)
            # normalize language
            chunk["language"]=chunk["language"].fillna("").astype(str).str.lower().replace({"english":"en","eng":"en","urdu":"ur"})
            chunk.loc[~chunk["language"].isin(["en","ur"]), "language"]=""
            chunk["severity"]=chunk["severity"].fillna("").astype(str).str.lower()
            longs.append(chunk)
    return pd.concat(longs, ignore_index=True) if longs else pd.DataFrame(columns=["vignette_id","language","domain","severity","label","item_id","rater_id"])

# Plot helpers
def bar_with_ci(ax, labels, means, ci_los, ci_his, title, ylabel):
    x = np.arange(len(labels))
    ax.bar(x, means)
    yerr = np.array([np.array(means)-np.array(ci_los), np.array(ci_his)-np.array(means)])
    ax.errorbar(x, means, yerr=yerr, fmt="none", capsize=4)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title(title); ax.set_ylabel(ylabel); ax.set_ylim(0, 1.0)

def table_fig(ax, df, title):
    ax.axis("off"); ax.set_title(title, pad=12)
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.2)

# Main
def main():
    print(f"[{STAMP()}] Using SENTRY_ROOT: {ROOT}")
    base_csv = latest_base_csv()
    per_dir  = latest_per_rater_dir()
    print(f"[{STAMP()}] Base CSV: {base_csv}")
    print(f"[{STAMP()}] Per-rater dir: {per_dir}")

    raters, frames = load_per_rater_frames(per_dir)
    items = common_items(frames)
    if not items: raise RuntimeError("No item columns found after cleaning.")
    print(f"[{STAMP()}] Items ({len(items)}): {', '.join(items)}")
    print(f"[{STAMP()}] Raters: {', '.join(raters)}")

    # Build long DF
    long_votes = build_long_votes(frames, items, base_csv)

    # Language split sanity on these 200
    base_df = pd.read_csv(base_csv, encoding="utf-8")
    v_ids   = pd.concat([f[["vignette_id"]] for f in frames], ignore_index=True).drop_duplicates()
    lang_counts = base_df.merge(v_ids, on="vignette_id", how="inner")["language"].str.lower().value_counts()
    n_en = int(lang_counts.get("en", 0)); n_ur = int(lang_counts.get("ur", 0))
    print(f"[{STAMP()}] T2 language split: EN={n_en} UR={n_ur}")

    # Global stats + CIs (fast)
    k_mean, ac_mean, pairs = compute_pairwise_stats(long_votes)
    k_boot_mean, k_lo, k_hi = bootstrap_ci_stat(long_votes, stat_kappa_mean, N_BOOT_GLOBAL, seed=42)
    ac_boot_mean, ac_lo, ac_hi = bootstrap_ci_stat(long_votes, stat_ac1_mean, N_BOOT_GLOBAL, seed=43)

    # Per-item
    per_item=[]
    for it in items:
        sub = long_votes[long_votes["item_id"]==it]
        km, am, _ = compute_pairwise_stats(sub)
        km_b, km_lo, km_hi = bootstrap_ci_stat(sub, stat_kappa_mean, N_BOOT_ITEM, seed=101+hash(it)%37)
        am_b, am_lo, am_hi = bootstrap_ci_stat(sub, stat_ac1_mean, N_BOOT_ITEM, seed=141+hash(it)%37)
        per_item.append([it, km, km_lo, km_hi, am, am_lo, am_hi])
    per_item = pd.DataFrame(per_item, columns=["item_id","kappa","k_lo","k_hi","ac1","ac1_lo","ac1_hi"]).sort_values("kappa", ascending=False)

    # By language
    by_lang=[]
    for lang in ["en","ur"]:
        sub = long_votes[long_votes["language"]==lang]
        if sub.empty: continue
        km, am, _ = compute_pairwise_stats(sub)
        km_b, km_lo, km_hi = bootstrap_ci_stat(sub, stat_kappa_mean, N_BOOT_LANG, seed=211 if lang=="en" else 212)
        am_b, am_lo, am_hi = bootstrap_ci_stat(sub, stat_ac1_mean, N_BOOT_LANG, seed=221 if lang=="en" else 222)
        by_lang.append([lang, km, km_lo, km_hi, am, am_lo, am_hi, sub[["vignette_id","item_id"]].drop_duplicates().shape[0]])
    by_lang = pd.DataFrame(by_lang, columns=["language","kappa","k_lo","k_hi","ac1","ac1_lo","ac1_hi","units"])

    # By domain
    by_dom=[]
    for dom in sorted(long_votes["domain"].dropna().unique()):
        sub = long_votes[long_votes["domain"]==dom]
        km, am, _ = compute_pairwise_stats(sub)
        km_b, km_lo, km_hi = bootstrap_ci_stat(sub, stat_kappa_mean, N_BOOT_DOM, seed=301+hash(dom)%53)
        am_b, am_lo, am_hi = bootstrap_ci_stat(sub, stat_ac1_mean, N_BOOT_DOM, seed=351+hash(dom)%53)
        by_dom.append([dom, km, km_lo, km_hi, am, am_lo, am_hi, sub[["vignette_id","item_id"]].drop_duplicates().shape[0]])
    by_dom = pd.DataFrame(by_dom, columns=["domain","kappa","k_lo","k_hi","ac1","ac1_lo","ac1_hi","units"]).sort_values("kappa", ascending=False)

    # By severity
    by_sev=[]
    for sev in ["low","medium","high"]:
        sub = long_votes[long_votes["severity"]==sev]
        if sub.empty: continue
        km, am, _ = compute_pairwise_stats(sub)
        km_b, km_lo, km_hi = bootstrap_ci_stat(sub, stat_kappa_mean, N_BOOT_SEV, seed=401+["low","medium","high"].index(sev))
        am_b, am_lo, am_hi = bootstrap_ci_stat(sub, stat_ac1_mean, N_BOOT_SEV, seed=451+["low","medium","high"].index(sev))
        by_sev.append([sev, km, km_lo, km_hi, am, am_lo, am_hi, sub[["vignette_id","item_id"]].drop_duplicates().shape[0]])
    by_sev = pd.DataFrame(by_sev, columns=["severity","kappa","k_lo","k_hi","ac1","ac1_lo","ac1_hi","units"]).sort_values("severity")

    # Save CSVs
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    per_item_csv = TABLES_DIR / f"T2_per_item_{ts}.csv"
    by_lang_csv  = TABLES_DIR / f"T2_by_language_{ts}.csv"
    by_dom_csv   = TABLES_DIR / f"T2_by_domain_{ts}.csv"
    by_sev_csv   = TABLES_DIR / f"T2_by_severity_{ts}.csv"
    long_csv     = TABLES_DIR / f"T2_long_votes_{ts}.csv"
    per_item.to_csv(per_item_csv, index=False, encoding="utf-8")
    by_lang.to_csv(by_lang_csv, index=False, encoding="utf-8")
    by_dom.to_csv(by_dom_csv, index=False, encoding="utf-8")
    by_sev.to_csv(by_sev_csv, index=False, encoding="utf-8")
    long_votes.to_csv(long_csv, index=False, encoding="utf-8")

    # PDF
    pdf_path = REPORTS_DIR / f"T2_IRR_Report_{ts}.pdf"
    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(8.5, 11))
        txt = f"""T2 Inter-Rater Reliability Report (FAST MODE)
        Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
        Raters: {', '.join(raters)}
        Items: {len(items)}   |   Vignettes: {len(v_ids := v_ids if 'v_ids' in locals() else pd.concat([f[['vignette_id']] for f in frames]).drop_duplicates().shape[0])}
        Language split (BASE): EN={n_en}  UR={n_ur}

        GLOBAL
          - Mean pairwise κ: {k_mean:.3f}  (95% CI [{k_lo:.3f}, {k_hi:.3f}])
          - Mean pairwise AC1: {ac_mean:.3f}  (95% CI [{ac_lo:.3f}, {ac_hi:.3f}])
        """
        fig.text(0.08, 0.95, "T2 Inter-Rater Reliability (κ / AC1)", fontsize=16, weight="bold", ha="left")
        fig.text(0.08, 0.92, f"SENTRY_ROOT: {ROOT}", fontsize=8, ha="left")
        fig.text(0.08, 0.87, textwrap.fill(txt, 100), fontsize=10, va="top")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        if not per_item.empty:
            fig, ax = plt.subplots(figsize=(11, 6))
            bar_with_ci(ax,
                        per_item["item_id"].tolist(),
                        per_item["kappa"].tolist(),
                        per_item["k_lo"].tolist(),
                        per_item["k_hi"].tolist(),
                        "Per-item Cohen's κ (95% CI)", "κ")
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

            fig, ax = plt.subplots(figsize=(11, 6))
            bar_with_ci(ax,
                        per_item["item_id"].tolist(),
                        per_item["ac1"].tolist(),
                        per_item["ac1_lo"].tolist(),
                        per_item["ac1_hi"].tolist(),
                        "Per-item Gwet's AC1 (95% CI)", "AC1")
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        if not by_lang.empty:
            fig, ax = plt.subplots(figsize=(8.5, 3.5))
            df_show = by_lang.copy()
            df_show["κ (CI)"]   = df_show.apply(lambda r: f"{r['kappa']:.3f} [{r['k_lo']:.3f}, {r['k_hi']:.3f}]", axis=1)
            df_show["AC1 (CI)"] = df_show.apply(lambda r: f"{r['ac1']:.3f} [{r['ac1_lo']:.3f}, {r['ac1_hi']:.3f}]", axis=1)
            df_show = df_show[["language","units","κ (CI)","AC1 (CI)"]]
            table_fig(ax, df_show, "By language"); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        if not by_dom.empty:
            fig, ax = plt.subplots(figsize=(11, 6))
            bar_with_ci(ax,
                        by_dom["domain"].tolist(),
                        by_dom["kappa"].tolist(),
                        by_dom["k_lo"].tolist(),
                        by_dom["k_hi"].tolist(),
                        "By domain: Cohen's κ (95% CI)", "κ")
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

            fig, ax = plt.subplots(figsize=(11, 6))
            bar_with_ci(ax,
                        by_dom["domain"].tolist(),
                        by_dom["ac1"].tolist(),
                        by_dom["ac1_lo"].tolist(),
                        by_dom["ac1_hi"].tolist(),
                        "By domain: Gwet's AC1 (95% CI)", "AC1")
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        if not by_sev.empty:
            fig, ax = plt.subplots(figsize=(8.5, 3.5))
            df_show = by_sev.copy()
            df_show["κ (CI)"]   = df_show.apply(lambda r: f"{r['kappa']:.3f} [{r['k_lo']:.3f}, {r['k_hi']:.3f}]", axis=1)
            df_show["AC1 (CI)"] = df_show.apply(lambda r: f"{r['ac1']:.3f} [{r['ac1_lo']:.3f}, {r['ac1_hi']:.3f}]", axis=1)
            df_show = df_show[["severity","units","κ (CI)","AC1 (CI)"]]
            table_fig(ax, df_show, "By severity"); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    print(f"\n[{STAMP()}] Saved PDF → {pdf_path}")
    print(f"[{STAMP()}] Tables in → {TABLES_DIR}")

if __name__ == "__main__":
    main()
