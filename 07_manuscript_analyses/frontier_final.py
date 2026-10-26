# -*- coding: utf-8 -*-
"""FINAL frontier table, all four classifier metrics, one consistent denominator rule.

Rule: for every band statistic the denominator is the set of pockets where the model value is
non-null AND the band is well defined (width > 0). Conjunctions use COMPLETE CASES only -- pandas
treats NaN as truthy in .all()/.any(), which silently inflated an earlier version of this table.

Backs Supplementary Table S5 (per-metric frontier, containment bands and median z; the complete-case
conjunctions under it) and the Results' 55-66% frontier and 54-67% within +/- 1 SD.

    python frontier_final.py [frontier_per_pocket.csv [model_metrics.csv [apoholo_stats.csv]]] > frontier_final_table.txt

Inputs: the per-pocket table of frontier_at_scale.py (default P.OUT/frontier_per_pocket.csv), the
model-row harvest with model_pocket_rms (default under cfg.AHOJ_DB_DIR/Summaries/model_metrics/full_run_3)
and the apo/holo metric means (cfg.APOHOLO_METRIC_STATS_CSV).
"""
import ma_paths as P  # roots; puts the classification code on sys.path

import sys
import numpy as np, pandas as pd
MODEL_METRICS = (P.cfg.AHOJ_DB_DIR / "Summaries/model_metrics/full_run_3/run_23082026_180200_do_collect_model_pLDDT"
                 / "query_summaries_do_collect_model_pLDDT.csv")
d = pd.read_csv(P.out(sys.argv[1]) if len(sys.argv) > 1 else P.out("frontier_per_pocket.csv"), low_memory=False)
r = pd.read_csv(sys.argv[2] if len(sys.argv) > 2 else MODEL_METRICS, low_memory=False)
d = d.merge(r[["job_name", "model_pocket_rms"]], on="job_name", how="left")
st = pd.read_csv(sys.argv[3] if len(sys.argv) > 3 else P.cfg.APOHOLO_METRIC_STATS_CSV,
                 usecols=["job_name","pocket_rms_apo_mean","pocket_rms_holo_mean"], low_memory=False)
d = d.merge(st, on="job_name", how="left")

SPEC = [("volume","model_pocket_volume"),("sasa","model_pocket_sasa"),
        ("RoG","model_pocket_RoG"),("pocket_rms","model_pocket_rms")]
rows=[]
for met,mod in SPEC:
    v=d[mod]; lo,hi,mu,sd=d[f"{met}_min"],d[f"{met}_max"],d[f"{met}_mean"],d[f"{met}_std_dev"]
    have=v.notna()
    def pct(mask, denom): return 100*float((mask&denom).sum())/float(denom.sum())
    dR=have&((hi-lo)>0); dS=have&(sd>0)
    rec={"metric":met,"n_model":int(have.sum())}
    rec["inside_minmax"]=pct((v>=lo)&(v<=hi),dR); rec["n_minmax"]=int(dR.sum())
    rec["pm2SD"]=pct((v>=mu-2*sd)&(v<=mu+2*sd),dS)
    rec["pm1SD"]=pct((v>=mu-sd)&(v<=mu+sd),dS); rec["n_SD"]=int(dS.sum())
    z=(v-mu).abs()/sd.where(sd>0); rec["z_lt_1"]=pct(z<1,dS); rec["median_z"]=float(np.nanmedian(z[dS]))
    a,h=d[f"{met}_apo_mean"],d[f"{met}_holo_mean"]
    ds=(h-a).abs(); dm=pd.concat([(v-a).abs(),(v-h).abs()],axis=1).min(axis=1)
    dF=have&ds.notna()&dm.notna()&(ds>0)
    rec["frontier"]=pct(dm<ds,dF); rec["n_frontier"]=int(dF.sum())
    d[f"F_{met}"]=(dm<ds).where(dF)
    rows.append(rec)
t=pd.DataFrame(rows)
pd.set_option("display.width",220,"display.max_columns",30)
print("=== FINAL per-metric frontier table (consistent denominators) ===")
print(t[["metric","frontier","n_frontier","inside_minmax","n_minmax","pm2SD","pm1SD","z_lt_1","median_z","n_SD"]].round(1).to_string(index=False))

for lab,MS in [("volume/SASA/RoG",["volume","sasa","RoG"]),("all four",["volume","sasa","RoG","pocket_rms"])]:
    c=[f"F_{m}" for m in MS]; ok=d[c].notna().all(axis=1)
    print(f"\n{lab}: complete cases n={int(ok.sum()):,} | frontier on ALL {100*d.loc[ok,c].all(axis=1).mean():.1f}% "
          f"| on AT LEAST ONE {100*d.loc[ok,c].any(axis=1).mean():.1f}%")
c=[f"F_{m}" for m in ["volume","sasa","RoG","pocket_rms"]]; ok=d[c].notna().all(axis=1)
k=d.loc[ok,c].sum(axis=1)
print("\nmetrics on the frontier (of 4), % of complete cases:")
print((100*k.value_counts(normalize=True).sort_index()).round(1).to_string())
