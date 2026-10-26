# -*- coding: utf-8 -*-
"""
Does the holo/apo **evidence ratio** predict P2Rank success better than the
apo/holo **class / probability** — as it does for docking?  (Cross-method test.)

For each P2Rank success definition
    DCA      = success_dca_4A   (any predicted pocket <= 4 A; lenient detection)
    TNS      = success_tns      (rank <= n_unp_pockets; primary metric)
    TNM      = success_tnm      (rank <= n_unp_pockets_multichain)
    non-TNS  = success_topn     (any-rank DCA4)    and  success_top1 (top-1)
we correlate success with each apo/holo predictor
    log2_ratio = log2((n_holo_avg+1)/n_apo_avg)     [canonical ratio_holo_to_apo_avg_plus1]
    pocket_p_holo, winning_p, is_holo, pocket_separation_pair
both zero-order (Spearman) and **size-controlled** (partial Spearman controlling
pocket size `rg` + ligand size `ligand_atoms` — the dominant P2Rank confounders,
mirroring the docking union_rg + ligand-heavy-atom controls), and compare nested
cross-validated logistic AUC (size / +ratio / +prob / +class).

Input : cfg.P2RANK_MERGED_PER_POCKET_CSV
Output: <out_dir>/  p2rank_ratio_vs_class_correlations.csv,
        p2rank_ratio_vs_class_models.csv, p2rank_ratio_vs_class.png, REPORT.md
Run in the `plotting` env (pandas/numpy/scipy/sklearn/matplotlib).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import sys as _sys
from pathlib import Path as _Path
for _root in _Path(__file__).resolve().parents:
    if (_root / "common" / "config.py").is_file():
        _sys.path.insert(0, str(_root))
        break
from common import config as cfg

try:
    from scipy import stats as sstats
except Exception:
    sstats = None

SUCCESS = [("success_dca_4A", "DCA (<=4A)"), ("success_tns", "TNS"), ("success_tnm", "TNM"),
           ("success_topn", "non-TNS (any-rank)"), ("success_top1", "top-1")]
PREDICTORS = ["log2_ratio", "pocket_p_holo", "winning_p", "is_holo", "pocket_separation_pair"]
SIZE_CONTROLS = ["rg", "ligand_atoms"]
CLASS_COLORS = {"apo": "#2C7FB8", "holo": "#F28E2B"}
PRED_COLORS = {"log2_ratio": "#7B3294", "pocket_p_holo": "#2C7FB8", "winning_p": "#8DB9DC",
               "is_holo": "#c2a5cf", "pocket_separation_pair": "#a6dba0"}


def _fmt_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return None
    if p == 0.0:
        return 0.0
    return float(f"{p:.1e}") if p < 1e-4 else round(float(p), 4)


def partial_spearman(x, y, ctrl):
    m = pd.concat([x, y] + ctrl, axis=1).notna().all(axis=1)
    n = int(m.sum())
    if n < 20:
        return None, None, n
    xr, yr = x[m].rank().values, y[m].rank().values
    Z = np.column_stack([np.ones(n)] + [c[m].rank().values for c in ctrl])
    if np.linalg.matrix_rank(Z) < Z.shape[1]:
        return None, None, n

    def resid(v):
        beta, *_ = np.linalg.lstsq(Z, v, rcond=None)
        return v - Z @ beta
    rx, ry = resid(xr), resid(yr)
    if rx.std() == 0 or ry.std() == 0:
        return None, None, n
    r = float(np.corrcoef(rx, ry)[0, 1])
    df = n - 2 - len(ctrl)
    p = float(2 * sstats.t.sf(abs(r * np.sqrt(df / (1 - r * r))), df)) if sstats else None
    return r, p, n


def spearman0(x, y):
    m = x.notna() & y.notna()
    n = int(m.sum())
    if n < 20 or x[m].nunique() < 2 or y[m].nunique() < 2:
        return None, None, n
    r, p = sstats.spearmanr(x[m], y[m])
    return (None if np.isnan(r) else float(r)), (None if np.isnan(p) else float(p)), n


def prep(df):
    d = df.copy()
    for c in ["log2_ratio_holo_to_apo_avg_plus1", "pocket_p_holo", "winning_p",
              "pocket_separation_pair", "rg", "ligand_atoms", "n_apo_avg", "n_holo_avg"] + [s for s, _ in SUCCESS]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    # canonical ratio (recompute defensively; identical to the stored column)
    d["log2_ratio"] = np.log2((d["n_holo_avg"] + 1.0) / d["n_apo_avg"].replace(0, np.nan))
    d["is_holo"] = d["pocket_class"].astype(str).str.lower().map({"holo": 1.0, "apo": 0.0})
    return d


def correlations(d):
    rows = []
    for scol, slab in SUCCESS:
        if scol not in d.columns:
            continue
        y = d[scol]
        for pred in PREDICTORS:
            r0, p0, n0 = spearman0(d[pred], y)
            ctrl = [d[c] for c in SIZE_CONTROLS if c in d.columns]
            rp, pp, n = partial_spearman(d[pred], y, ctrl)
            rows.append({"success": scol, "success_label": slab, "predictor": pred,
                         "spearman": None if r0 is None else round(r0, 3), "spearman_p": _fmt_p(p0),
                         "partial_ctrl_size": None if rp is None else round(rp, 3),
                         "partial_p": _fmt_p(pp), "controls": "+".join(SIZE_CONTROLS), "n": n})
    return pd.DataFrame(rows)


def nested_auc(d):
    rows = []
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_predict, StratifiedKFold
        from sklearn.metrics import roc_auc_score
    except Exception as e:
        return pd.DataFrame([{"success": "ERROR", "model": repr(e)}])
    models = {"size": SIZE_CONTROLS, "size+ratio": SIZE_CONTROLS + ["log2_ratio"],
              "size+prob": SIZE_CONTROLS + ["pocket_p_holo"], "size+class": SIZE_CONTROLS + ["is_holo"],
              "size+ratio+prob": SIZE_CONTROLS + ["log2_ratio", "pocket_p_holo"]}
    for scol, slab in SUCCESS:
        if scol not in d.columns:
            continue
        y = d[scol]
        for name, cols in models.items():
            cols = [c for c in cols if c in d.columns]
            m = d[cols].notna().all(axis=1) & y.notna()
            yy = y[m].astype(int).values
            if len(np.unique(yy)) < 2 or m.sum() < 100:
                continue
            X = StandardScaler().fit_transform(d.loc[m, cols].values)
            cv = StratifiedKFold(5, shuffle=True, random_state=0)
            proba = cross_val_predict(LogisticRegression(max_iter=2000), X, yy, cv=cv,
                                      method="predict_proba")[:, 1]
            rows.append({"success": scol, "success_label": slab, "model": name,
                         "cv_auc": round(float(roc_auc_score(yy, proba)), 4), "n": int(m.sum())})
    return pd.DataFrame(rows)


def make_figure(corr, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [lab for _, lab in SUCCESS if _ in corr.success.values]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, (metric_col, title) in zip(axes, [("spearman", "zero-order Spearman"),
                                              ("partial_ctrl_size", "size-controlled partial Spearman")]):
        x = np.arange(len(labels)); w = 0.16
        for k, pred in enumerate(PREDICTORS):
            vals = [corr[(corr.success_label == lab) & (corr.predictor == pred)][metric_col].values
                    for lab in labels]
            vals = [float(v[0]) if len(v) and v[0] is not None and not pd.isna(v[0]) else np.nan for v in vals]
            ax.bar(x + (k - 2) * w, vals, width=w, color=PRED_COLORS.get(pred, "0.5"),
                   label=pred if ax is axes[0] else None)
        ax.axhline(0, color="0.3", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
        ax.set_title(title); ax.set_ylabel("Spearman ρ with P2Rank success"); ax.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8, title="predictor", ncol=1, loc="upper right")
    fig.suptitle("P2Rank success vs holo/apo evidence ratio, class probability, and binary class")
    fig.tight_layout(); fig.savefig(out_png, dpi=140); plt.close(fig)


def write_report(corr, models, path):
    L = ["# P2Rank success — evidence ratio vs class/probability (run_3, per-pocket)\n",
         "Cross-method test of the docking finding that the **holo/apo evidence ratio** "
         "`log2((n_holo+1)/n_apo)` predicts success better than the apo/holo class or its "
         "probability. Size controls = `rg` + `ligand_atoms` (the dominant P2Rank confounders).\n"]
    for scol, slab in SUCCESS:
        sub = corr[corr.success == scol]
        if sub.empty:
            continue
        L.append(f"\n## {slab}  (`{scol}`)\n")
        L.append("| predictor | ρ (zero-order) | p | ρ (size-controlled) | p | n |")
        L.append("|---|---|---|---|---|---|")
        for _, r in sub.iterrows():
            L.append(f"| {r['predictor']} | {r['spearman']} | {r.get('spearman_p')} | "
                     f"{r['partial_ctrl_size']} | {r.get('partial_p')} | {r['n']} |")
    if not models.empty and "model" in models.columns and "cv_auc" in models.columns:
        L.append("\n## Nested cross-validated logistic AUC (does the ratio/class add over size?)\n")
        piv = models.pivot_table(index="success_label", columns="model", values="cv_auc", aggfunc="first")
        order = [c for c in ["size", "size+ratio", "size+prob", "size+class", "size+ratio+prob"] if c in piv.columns]
        piv = piv.reindex(columns=order)
        L.append("| success | " + " | ".join(order) + " |")
        L.append("|" + "---|" * (len(order) + 1))
        for idx, row in piv.iterrows():
            L.append(f"| {idx} | " + " | ".join(f"{row[c]:.4f}" if pd.notna(row[c]) else "-" for c in order) + " |")
    L.append("\n## Reading\n")
    L.append("- If `ρ(ratio)` > `ρ(p_holo)` > `ρ(winning_p)`/`ρ(is_holo)` after size control, the "
             "continuous evidence ratio is the strongest apo/holo predictor of P2Rank detectability — "
             "the same ordering docking shows.")
    L.append("- If `size+ratio` ≈ `size+ratio+prob` in AUC, the class probability adds little beyond "
             "the ratio (the classification is a thresholded summary of the evidence), consistent with "
             "the docking redundancy result.")
    path.write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_pocket_csv", type=Path, default=cfg.P2RANK_MERGED_PER_POCKET_CSV)
    ap.add_argument("--out_dir", type=Path, default=cfg.P2RANK_TABLES_DIR.parent / "ratio_vs_class")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    d = prep(pd.read_csv(args.per_pocket_csv, low_memory=False))
    corr = correlations(d); corr.to_csv(args.out_dir / "p2rank_ratio_vs_class_correlations.csv", index=False)
    models = nested_auc(d); models.to_csv(args.out_dir / "p2rank_ratio_vs_class_models.csv", index=False)
    make_figure(corr, args.out_dir / "p2rank_ratio_vs_class.png")
    write_report(corr, models, args.out_dir / "REPORT.md")

    print("[CORRELATIONS]\n", corr.to_string(index=False))
    print("\n[NESTED AUC]\n", models.to_string(index=False))
    print(f"\n[SAVE] {args.out_dir}")


if __name__ == "__main__":
    main()
