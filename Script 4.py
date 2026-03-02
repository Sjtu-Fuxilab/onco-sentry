# Script 04
import pandas as pd, json, os
from pathlib import Path

ROOT = Path(r"D:\Data")
RUN   = ROOT/"runs"/"T1"
V_EN  = RUN/"frozen_vignettes_en.jsonl"
V_UR  = RUN/"frozen_vignettes_ur.jsonl"
SC    = ROOT/"tables"/"scored_long_cells.csv" 

sc = pd.read_csv(SC)
# Unanimous (or simple majority) consensus → adjudicated bool
cons = (sc.groupby(["vignette_id","item_id"])["rater_bool"]
          .mean().rename("p_true").reset_index())
cons["adj"] = cons["p_true"].round().astype(bool) 

# pivot to wide adj__* columns
adj_wide = cons.pivot(index="vignette_id", columns="item_id", values="adj")
adj_wide.columns = [f"adj__{c}" for c in adj_wide.columns]
adj_wide = adj_wide.reset_index()

def promote(jsonl_in: Path, jsonl_out: Path):
    rows = [json.loads(x) for x in jsonl_in.read_text(encoding="utf-8").splitlines()]
    df = pd.DataFrame(rows)
    out = df.merge(adj_wide, on="vignette_id", how="left")
    # Fill any missing adj__* with expected__* (safety)
    for c in [c for c in out.columns if c.startswith("adj__")]:
        alt = "expected__" + c.split("adj__")[1]
        if alt in out.columns:
            out[c] = out[c].where(out[c].notna(), out[alt])
        out[c] = out[c].fillna(False).astype(bool)
    jsonl_out.write_text(
        "\n".join(out.apply(lambda r: json.dumps(r.to_dict(), ensure_ascii=False), axis=1)),
        encoding="utf-8"
    )

promote(V_EN, RUN/"adjudicated_vignettes_en.jsonl")
promote(V_UR, RUN/"adjudicated_vignettes_ur.jsonl")
print("Adjudicated JSONLs written to runs/T1/")
