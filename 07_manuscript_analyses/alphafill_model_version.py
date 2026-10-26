# -*- coding: utf-8 -*-
"""Which AlphaFold model underlies the AlphaFill transplants of this work (Methods, AlphaFill matching; Data
availability; Supplementary Note 4).

AlphaFill computes its transplants on the model it serves and records that model in the JSON it returns.
The 12,105 cached JSONs (<WORK>/alphafill/jsons, downloaded between 2025-10 and 2026-07) fall into two kinds:

  * archive entries (AlphaFill 2.1.0 / 2.1.1, dated 2023-11 to 2024-12): the JSON names the model file,
    e.g. "/DATA/afdb-v4//A0/AF-A0A009I821-F1-model_v4.cif.gz", or carries no file field but predates the
    version 6 release;
  * on-demand entries ("source": "user", AlphaFill 2.2.0 / 2.3.0, dated 2025-11 to 2026-07): computed when our
    downloader triggered them, on the model the AlphaFold DB served that day, which since 2025-08 is version 6.

The script tallies both kinds over the headline AlphaFill set of the manuscript (ST_100 sites with a
same-UniProt transplant at alignment identity >= 0.30: 17,551 sites, 7,995 AF-Apo / 9,556 AF-Holo), joins the
on-demand accessions to the local v4/v6 model comparison (afdb_v4_v6_diff/v4_v6_atoms.csv: coordinate-identical
or re-predicted), recomputes the Figure 4a medians (average clash score by separation tier and class) with and
without the on-demand sites, and with --fetch N downloads N on-demand entries' model files from alphafill.eu and
compares their CA coordinates with the local v4 and v6 stores (results cached in
<manuscript_analyses_dir>/alphafill_model_version_fetch.json, which is read whenever it exists).

These are the Methods numbers "For 84% of the 17,551 sites the transplants are AlphaFill archive entries ...
The remaining 16% (2,849 sites on 1,325 accessions) were computed on request ...; for 2,540 of these sites the
version 6 model is coordinate-identical to version 4, so at most 309 sites (1.8%) carry transplants on
re-predicted coordinates. Leaving the on-demand sites out changes no tier median of Figure 4a by more than 0.014."

Inputs (read-only)
    <alphafill_unified analysis_v1>/per_ubs_summary_same_uniprot_ge_0p30.csv      (cfg.ALPHAFILL_UNIFIED_ANALYSIS_DIR)
    <alphafill jsons>/<accession>.json                                             (cfg.ALPHAFILL_JSONS_DIR)
    <manuscript_analyses_dir>/afdb_v4_v6_diff/v4_v6_atoms.csv   (afdb_v4_v6_diff.py atoms)
    <afdb_store_dir>/{v4,v6}/AF-<acc>-F1-model_v{4,6}.cif.gz      (--fetch only)

Usage
    python alphafill_model_version.py [--fetch N] [--seed 7]
"""
import argparse
import gzip
import json
import os
import random
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import ma_paths as P

HERE = Path(__file__).resolve().parent
WORK = Path(P.WD)
STORE = Path(P.anchor("afdb_store_dir", "AHOJ_AFDB_STORE_DIR"))     # v4/, v3/ and v6/ model folders
PER_UBS = Path(P.cfg.ALPHAFILL_UNIFIED_ANALYSIS_DIR) / "per_ubs_summary_same_uniprot_ge_0p30.csv"
JSONS = Path(P.cfg.ALPHAFILL_JSONS_DIR)
ATOMS = Path(P.OUT) / "afdb_v4_v6_diff" / "v4_v6_atoms.csv"
FETCH_CACHE = Path(P.out("alphafill_model_version_fetch.json"))
AFF_URL = "https://alphafill.eu/v1/aff/{acc}"
V6_RELEASE = "2025-08-01"      # the version 6 revision date carried in the model files' audit history
TIERS = ["very_low", "low", "moderate", "high", "very_high"]


def json_kind(acc):
    """(kind, alphafill_version, date, model file) for one cached JSON."""
    p = JSONS / f"{acc}.json"
    if not p.exists():
        return "no json", None, None, None
    with open(p, encoding="utf-8") as fh:
        j = json.load(fh)
    if not isinstance(j, dict):
        return "null json", None, None, None
    f = j.get("file") or ""
    v, d, src = j.get("alphafill_version"), (j.get("date") or ""), j.get("source")
    if "afdb-v4" in f or "model_v4" in f:
        kind = "archive, v4 file named"
    elif src == "user" or d >= V6_RELEASE:
        kind = "on demand, after v6 release"
    else:
        kind = "archive, no file field, before v6 release"
    return kind, v, d, f


def ca_coords(path):
    """label_seq_id -> (x, y, z) of CA atoms (polymer ATOM records) from an mmCIF, gzip or plain."""
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", encoding="utf-8", errors="replace") as fh:
        lines = fh.read().split("\n")
    cols, out, inloop = [], {}, False
    for l in lines:
        if l.startswith("_atom_site."):
            cols.append(l.split()[0].split(".")[1]); inloop = True
        elif inloop and l and not l.startswith(("_", "#")):
            p = l.split()
            if len(p) >= len(cols):
                d = dict(zip(cols, p))
                if d.get("group_PDB") == "ATOM" and d.get("label_atom_id") == "CA" and d["label_seq_id"].isdigit():
                    out[int(d["label_seq_id"])] = tuple(float(d[k]) for k in ("Cartn_x", "Cartn_y", "Cartn_z"))
        elif inloop and l.startswith("#") and out:
            break
    return out


def revisions(path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", encoding="utf-8", errors="replace") as fh:
        t = fh.read()
    m = re.search(r"_pdbx_audit_revision_history\.revision_date\s*\n((?:.*\n){0,12})", t)
    return re.findall(r"\d{4}-\d{2}-\d{2}", m.group(1)) if m else []


def fetch_compare(accs, cache):
    for acc in accs:
        if acc in cache and "server_json_date" in cache[acc]:
            continue
        rec = cache.get(acc, {})
        try:
            # the JSON the server holds today: if its date equals the cached JSON's, the served model file is
            # the one the cached transplants were computed on
            req = urllib.request.Request(AFF_URL.format(acc=acc) + "/json", headers={"User-Agent": "ahoj-af-verify/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                sj = json.load(r)
            rec["server_json_date"] = sj.get("date")
            rec["server_json_version"] = sj.get("alphafill_version")
            rec["cached_json_date"] = json_kind(acc)[2]
            if "server_n_ca" in rec:
                cache[acc] = rec
                FETCH_CACHE.write_text(json.dumps(cache, indent=1), encoding="utf-8")
                continue
            req = urllib.request.Request(AFF_URL.format(acc=acc), headers={"User-Agent": "ahoj-af-verify/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                raw = r.read()
            tmp = Path(P.out(f"_aff_{acc}.cif"))
            tmp.write_bytes(raw)
            a = ca_coords(tmp)
            rec["server_revisions"] = revisions(tmp)
            rec["server_n_ca"] = len(a)
            for v in ("v4", "v6"):
                p = STORE / v / f"AF-{acc}-F1-model_{v}.cif.gz"
                if not p.exists():
                    rec[v] = "absent locally"; continue
                b = ca_coords(p)
                common = sorted(set(a) & set(b))
                md = max(max(abs(a[k][i] - b[k][i]) for i in range(3)) for k in common) if common else None
                rec[v] = {"n_ca": len(b), "n_common": len(common), "max_abs_dxyz": md,
                          "identical": bool(md is not None and md == 0.0 and len(common) == len(a) == len(b))}
            tmp.unlink()
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{e.__class__.__name__}: {e}"
        cache[acc] = rec
        FETCH_CACHE.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    return cache


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fetch", type=int, default=0, help="download N on-demand entries from alphafill.eu and compare coordinates")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    d = pd.read_csv(PER_UBS, low_memory=False,
                    usecols=["threshold", "unit_uid", "uniprots_str", "pocket_class", "matched",
                             "separation_level_db", "avg_clash", "avg_clash_count"])
    d = d[(d.threshold == "ST_100") & d.matched.fillna(False).astype(bool)].copy()
    d["cls"] = d.pocket_class.astype(str).str.lower()
    d = d[d.cls.isin(["apo", "holo"])].copy()
    d["acc"] = d.uniprots_str.astype(str).str.split().str[0]
    print("=== the headline AlphaFill set (ST_100, same-UniProt transplant, identity >= 0.30) ===")
    print(f"  sites {len(d):,}: AF-Apo {int((d.cls == 'apo').sum()):,}, AF-Holo {int((d.cls == 'holo').sum()):,}; "
          f"accessions {d.acc.nunique():,}")

    kinds = {acc: json_kind(acc) for acc in d.acc.unique()}
    d["kind"] = d.acc.map(lambda x: kinds[x][0])
    d["af_version"] = d.acc.map(lambda x: kinds[x][1])
    d["af_date"] = d.acc.map(lambda x: kinds[x][2])
    print("\n=== the model AlphaFill used, from the JSON of each site's accession ===")
    tab = d.groupby(["kind", "cls"]).size().unstack("cls").fillna(0).astype(int)
    tab["all"] = tab.sum(axis=1)
    tab["share %"] = 100 * tab["all"] / len(d)
    print(tab.to_string())
    on = d[d.kind.str.startswith("on demand")]
    print(f"\n  on-demand sites: {len(on):,} ({100 * len(on) / len(d):.1f}%; AF-Apo {100 * (on.cls == 'apo').sum() / (d.cls == 'apo').sum():.1f}% "
          f"of AF-Apo sites, AF-Holo {100 * (on.cls == 'holo').sum() / (d.cls == 'holo').sum():.1f}% of AF-Holo sites) "
          f"on {on.acc.nunique():,} accessions")
    print("  AlphaFill version x JSON year, all sites:")
    vy = d.assign(year=d.af_date.fillna("").str[:4]).groupby(["af_version", "year"]).size()
    print("    " + "; ".join(f"{v} / {y}: {n:,}" for (v, y), n in vy.items()))

    # did the v6 model differ from the v4 model for the on-demand accessions?
    at = pd.read_csv(ATOMS, usecols=["acc", "status", "identical", "coords_equal", "category"], low_memory=False)
    at = at.drop_duplicates("acc").set_index("acc")
    on_acc = pd.Index(on.acc.unique())
    j = at.reindex(on_acc)
    print("\n=== for the on-demand accessions: does the version 6 model equal the version 4 model? (v4_v6_atoms.csv) ===")
    print(f"  accessions {len(on_acc):,}; with a local v4/v6 comparison {int(j.status.notna().sum()):,}")
    print("  status: " + "; ".join(f"{k}: {v:,}" for k, v in j.status.value_counts(dropna=False).items()))
    ident = j.identical.eq(True)
    print(f"  coordinate-identical v6 model: {int(ident.sum()):,} accessions; moved: {int((j.status.notna() & ~ident).sum()):,}")
    on_sites_ident = on[on.acc.isin(j.index[ident])]
    print(f"  on-demand sites on a coordinate-identical model: {len(on_sites_ident):,} of {len(on):,}; "
          f"on a re-predicted model: {len(on) - len(on_sites_ident):,} ({100 * (len(on) - len(on_sites_ident)) / len(d):.1f}% of all sites)")
    print("  category of the moved ones: " + "; ".join(f"{k}: {v:,}" for k, v in j[~ident].category.value_counts(dropna=False).items()))

    print("\n=== Figure 4a medians: average clash score by separation tier, AF-Apo / AF-Holo ===")
    d["tier"] = d.separation_level_db.astype(str).str.strip().str.lower()
    d["avg_clash"] = pd.to_numeric(d.avg_clash, errors="coerce")
    keep = d[~d.kind.str.startswith("on demand")]
    print(f"  {'tier':10s} {'all sites (n apo/holo)':>34s} {'without on-demand sites (n)':>36s}")
    for t in TIERS + ["overall"]:
        s = d if t == "overall" else d[d.tier == t]
        k = keep if t == "overall" else keep[keep.tier == t]
        sa, sh = s[s.cls == "apo"].avg_clash, s[s.cls == "holo"].avg_clash
        ka, kh = k[k.cls == "apo"].avg_clash, k[k.cls == "holo"].avg_clash
        print(f"  {t:10s} {sa.median():.3f} / {sh.median():.3f} ({len(sa):,} / {len(sh):,})"
              f"{'':6s}{ka.median():.3f} / {kh.median():.3f} ({len(ka):,} / {len(kh):,})")

    if a.fetch or FETCH_CACHE.exists():
        cache = json.loads(FETCH_CACHE.read_text(encoding="utf-8")) if FETCH_CACHE.exists() else {}
        if a.fetch:
            rng = random.Random(a.seed)
            pool = sorted(set(on_acc) & set(j.index[j.status.notna()]))
            moved = [x for x in pool if not ident.get(x, False)]
            same = [x for x in pool if ident.get(x, False)]
            pick = rng.sample(moved, min(a.fetch // 2, len(moved))) + rng.sample(same, min(a.fetch - a.fetch // 2, len(same)))
            cache = fetch_compare(pick, cache)
        print(f"\n=== alphafill.eu model files of on-demand accessions against the local stores ({FETCH_CACHE.name}) ===")
        c = Counter()
        same_date = 0
        for acc, r in cache.items():
            if "error" in r:
                c["error"] += 1; continue
            v4, v6 = r.get("v4"), r.get("v6")
            has_v6_rev = any(x >= V6_RELEASE for x in r.get("server_revisions", []))
            dated = r.get("server_json_date") == r.get("cached_json_date") and r.get("server_json_date") is not None
            same_date += dated
            lab = lambda x: ("no local file" if not isinstance(x, dict) else ("identical" if x["identical"] else "differs"))  # noqa: E731
            c[("server JSON is the cached one" if dated else "server JSON is newer than the cached one",
               f"v6 revision in header: {has_v6_rev}", f"local v4: {lab(v4)}", f"local v6: {lab(v6)}")] += 1
        print(f"  entries compared: {len(cache):,}; the server still holds the cached JSON (same date) for {same_date:,} of them")
        for k, n in sorted(c.items(), key=str):
            print(f"    {n:4d}  {k}")
        recs = [r for r in cache.values() if "error" not in r]
        eq4 = sum(1 for r in recs if isinstance(r.get("v4"), dict) and r["v4"]["identical"])
        eq4_nov6 = sum(1 for r in recs if isinstance(r.get("v4"), dict) and r["v4"]["identical"] and not isinstance(r.get("v6"), dict))
        ne4_eq6 = sum(1 for r in recs if isinstance(r.get("v4"), dict) and not r["v4"]["identical"]
                      and isinstance(r.get("v6"), dict) and r["v6"]["identical"])
        print(f"  reading: the file AlphaFill computed on has the version 4 coordinates for {eq4} of {len(recs)} sampled accessions\n"
              f"  ({eq4_nov6} of them have no local version 6 file to compare); it differs from version 4 and equals the local\n"
              f"  version 6 file for {ne4_eq6}, i.e. the transplants sit on a re-predicted model. The sample is half 'moved',\n"
              f"  half 'identical' accessions, so the local v4/v6 comparison above bounds the re-predicted share from above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
