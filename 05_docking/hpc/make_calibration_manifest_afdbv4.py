# -*- coding: utf-8 -*-
"""
Calibration manifest for the AFDB v4 re-run of the docking: a stratified sample of pockets whose
AlphaFold model is the same file in v4 and v6, so that the July (v6) result of every sampled job is
the exact expectation. With seed 42 and the same image, any affinity difference on this set is an
environment change, not a model change.

Sampling: pockets of the v4 manifest on accessions of category `identical`
(`afdb_v4_v6_accessions.csv`, ahoj-af-project), strata = pocket_class (apo, holo) x tercile of union_rg,
equal allocation, fixed seed; every job of a sampled pocket is kept. Rows are copied unchanged.

    python make_calibration_manifest_afdbv4.py --manifest <run3_afdbv4/manifest/docking_manifest.csv> \
        --accessions <afdb_v4_v6_accessions.csv> --out <.../docking_manifest_calib.csv> [--n_pockets 2000]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--accessions", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n_pockets", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    man = pd.read_csv(a.manifest, dtype=str, keep_default_na=False)
    cat = pd.read_csv(a.accessions, usecols=["acc", "category"]).set_index("acc").category
    idn = man[man.site_uniprot.map(cat) == "identical"]
    pk = idn.drop_duplicates("site_uid")[["site_uid", "pocket_class", "union_rg"]].copy()
    pk = pk[pk.pocket_class.isin(["apo", "holo"])]          # the few tied pockets would take a third of the strata
    pk["size_tercile"] = pd.qcut(pk.union_rg.astype(float), 3, labels=["small", "mid", "large"])
    strata = pk.groupby(["pocket_class", "size_tercile"], observed=True)
    per = a.n_pockets // strata.ngroups
    pick = pd.concat([g.sample(n=min(per, len(g)), random_state=a.seed) for _, g in strata])
    out = man[man.site_uid.isin(set(pick.site_uid))]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    meta = {"manifest": str(a.manifest), "accessions": str(a.accessions), "seed": a.seed,
            "identical_model_pockets": int(len(pk)), "pockets": int(out.site_uid.nunique()), "jobs": int(len(out)),
            "receptors": int(out.site_uniprot.nunique()),
            "pockets_by_stratum": {f"{c}/{s}": int(len(g)) for (c, s), g in pick.groupby(["pocket_class", "size_tercile"], observed=True)}}
    a.out.with_name(a.out.stem + "_manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
