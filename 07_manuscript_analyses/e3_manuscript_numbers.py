# -*- coding: utf-8 -*-
"""
Every pooled number the manuscript quotes for detection and docking, printed for one model set.

The detection and docking arms ran on the AFDB v6 files first and were re-run on the v4 models the
classification used (Supplementary Note 4). The manuscript quotes, for the v4 run, the per-tool and
*consensus* pooled rates (Figure 3a's "overall" bars), the "missed by all three" split (Supplementary
Table S2), the docking medians and the denominators (the detection set of 87,434 sites: 35,124 AF-Apo,
52,014 AF-Holo, 296 tied; docking on 61,216 G1 pockets, 61,012 of them with a definite class). This
script recomputes all of them from the same tables Figure 3 is drawn from.

It builds `base` exactly as the Figure 3 composer does -- same three merged per-pocket tables, same
left joins on `site_uid`, same per-pocket consensus mean -- so the printed pooled rates are by
construction the "overall" bars of Figure 3a.

Usage (needs pandas only):

    python e3_manuscript_numbers.py                               # July, v6 paths
    AHOJ_AF_MODEL_SET=afdbv4 python e3_manuscript_numbers.py      # the v4 re-run (the manuscript's)
    python e3_manuscript_numbers.py --both                        # both, side by side

`--both` re-imports nothing: it reads each tree by explicit suffix, so it does not depend on the
environment variable being set correctly.

Validation: with the switch unset every printed value must reproduce the July (v6) values the
draft carried before the v4 re-run (P2Rank 63/47, fpocket 63/50, IF-SitePred 68/46, consensus 65/48,
docking 69/49, 86,946 / 86,651 / 10,419 / 10,395); `--both` prints that gate as a PASS/FAIL line.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

import ma_paths as P

W = P.WD


def load(*parts):
    return pd.read_csv(os.path.join(W, *parts), low_memory=False)


def build(sfx):
    """The Figure 3 table for one model set. `sfx` is '' (July/v6) or '_afdbv4'."""
    tabs = {
        "P2Rank": load("lbs_generality", "p2rank" + sfx,
                       "lbs_p2rank_st100_merged_per_pocket.csv"),
        "fpocket": load("lbs_generality", "fpocket", "full" + sfx,
                        "lbs_fpocket_st100_merged_per_pocket.csv"),
        "IF-SitePred": load("lbs_generality", "ifsitepred", "full" + sfx,
                            "lbs_ifsitepred_st100_merged_per_pocket.csv"),
    }
    pred = ["P2Rank", "fpocket", "IF-SitePred"]
    base = tabs["P2Rank"][["site_uid", "site_uniprot", "pocket_class", "pocket_p_holo",
                           "log2_ratio_holo_to_apo_avg_plus1", "winning_p",
                           "prob_level_db"]].copy()
    base = base.rename(columns={"log2_ratio_holo_to_apo_avg_plus1": "log2_ratio"})
    for k in pred:
        base = base.merge(
            tabs[k][["site_uid", "success_tns"]].rename(columns={"success_tns": k}),
            on="site_uid", how="left")
    base["consensus"] = base[pred].mean(axis=1)
    base["cls"] = base["pocket_class"].str.lower()

    dk = load("docking", "run3" + sfx, "analysis", "pocket_level_table.csv")
    dk1 = dk[dk["ligand_group"] == 1].dropna(subset=["is_holo"]).copy()
    dk1["dockable_bin"] = (dk1["frac_dockable"] > 0).astype(int)
    dk1["cls"] = np.where(dk1["is_holo"].eq(1), "holo", "apo")
    return base, dk, dk1, pred


def rates(frame, col, cls_col="cls"):
    """(apo %, holo %, gap in points) for a 0/1 or fractional success column."""
    a = frame.loc[frame[cls_col] == "apo", col].mean() * 100
    h = frame.loc[frame[cls_col] == "holo", col].mean() * 100
    return a, h, h - a


def report(sfx, out=sys.stdout):
    tag = "AFDB v4 (re-run)" if sfx == "_afdbv4" else "AFDB v6 (July, as published)"
    p = lambda s="": print(s, file=out)
    base, dk, dk1, pred = build(sfx)

    p("=" * 78)
    p(f"  {tag}   [suffix {sfx!r}]")
    p("=" * 78)

    # ---- the four denominators, which must not be mixed.
    # NOTE the site list is the same under both versions (87,434 rows): the 488 sites whose
    # accession has no v6 model are PRESENT in July's table but carry no score. So every count
    # here has to be conditioned on `scored`, or v6 silently reports v4's numbers.
    scored = base["P2Rank"].notna()
    classified = base["cls"].isin(["apo", "holo"])
    s = base[scored]
    p("\n-- denominators --")
    p(f"  rows in the site list             {len(base):>9,}   (same under both versions)")
    p(f"  scored detection sites            {int(scored.sum()):>9,}")
    p(f"  classified (class contrasts)      {int((scored & classified).sum()):>9,}")
    p(f"    AF-Apo                          {int((s.cls == 'apo').sum()):>9,}")
    p(f"    AF-Holo                         {int((s.cls == 'holo').sum()):>9,}")
    p(f"  tied, no class                    {int((s.cls == 'tie').sum()):>9,}")
    p(f"  UniProt accessions (scored sites) {s.site_uniprot.nunique():>9,}")
    p(f"  UniProt clusters (classified)     "
      f"{s.loc[s.cls.isin(['apo', 'holo']), 'site_uniprot'].nunique():>9,}")

    # ---- Figure 3a 'overall' bars: pooled class rates, per method + consensus
    p("\n-- detection, pooled over all pockets (Figure 3a 'overall' group) --")
    p(f"  {'method':<14}{'AF-Apo':>9}{'AF-Holo':>10}{'gap (pp)':>11}")
    det = {}
    for m in pred + ["consensus"]:
        a, h, g = rates(base[scored & classified], m)
        det[m] = (a, h, g)
        p(f"  {m:<14}{a:>8.2f}%{h:>9.2f}%{g:>10.2f}")
    p("  rounded as the text quotes them: " + ", ".join(
        f"{m} {det[m][1]:.0f}/{det[m][0]:.0f} (+{det[m][2]:.0f})" for m in pred + ["consensus"]))

    # ---- missed by all three
    miss = base[scored & classified].copy()
    miss["none"] = (miss[pred].sum(axis=1) == 0).astype(int)
    a, h, g = rates(miss, "none")
    p("\n-- detected by NO predictor (Supplementary Table S2) --")
    p(f"  of {int(len(miss)):,} classified pockets: AF-Apo {a:.1f}%, AF-Holo {h:.1f}%")

    # ---- docking
    p("\n-- docking, cognate (group 1), pocket level --")
    p(f"  docked pockets (group 1)          {int(len(dk[dk.ligand_group == 1])):>9,}")
    p(f"  of which carry a definite class   {int(len(dk1)):>9,}")
    a, h, g = rates(dk1, "dockable_bin")
    dock = (a, h, g)
    p(f"  top-pose success                  AF-Apo {a:.2f}%  AF-Holo {h:.2f}%  gap {g:.2f} pp")
    # The text's "-7.16 versus -6.03 kcal/mol" is the JOB-level group-1 median, not the
    # pocket-level one (which is -7.22 / -6.05). Recomputed here at full precision, because
    # `success_by_group_class.csv` stores it to 3 dp and v4's AF-Apo lands on -6.035.
    jm = load("docking", "run3" + sfx, "analysis", "docking_run3_merged.csv")
    j1 = jm[(jm["ligand_group"] == 1) & jm["affinity_top"].notna()].copy()
    j1["cls"] = np.where(j1["is_holo"].eq(1), "holo", "apo")
    ma = j1.loc[j1.cls == "apo", "affinity_top"].median()
    mh = j1.loc[j1.cls == "holo", "affinity_top"].median()
    p(f"  median affinity, JOB level        AF-Apo {ma:.4f}  AF-Holo {mh:.4f}"
      f"   (text quotes 2 dp: {mh:.2f} versus {ma:.2f})")
    for col in ("best_aff", "median_aff"):
        pa = dk1.loc[dk1.cls == "apo", col].median()
        ph = dk1.loc[dk1.cls == "holo", col].median()
        p(f"  median {col:<11} pocket level  AF-Apo {pa:.2f}   AF-Holo {ph:.2f}"
          f"   (not the text's number)")
    return det, dock, (int(scored.sum()), int((scored & classified).sum()),
                       s.site_uniprot.nunique(),
                       s.loc[s.cls.isin(['apo', 'holo']), 'site_uniprot'].nunique(),
                       int((s.cls == 'apo').sum()), int((s.cls == 'holo').sum()),
                       int((s.cls == 'tie').sum()),
                       int(len(dk[dk.ligand_group == 1])), int(len(dk1)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--both", action="store_true",
                    help="print both model sets and check v6 against the published values")
    a = ap.parse_args()

    if not a.both:
        report(P.MODEL_SFX)
        return

    d6, k6, n6 = report("")
    print()
    d4, k4, n4 = report("_afdbv4")

    # the reproduce-before-you-trust gate: v6 must equal what the shared draft of 2026-09-15 says
    print("\n" + "=" * 78)
    print("  GATE: v6 reproduces the published values")
    print("=" * 78)
    ok = True

    def chk(what, got, want, fmt="{}"):
        nonlocal ok
        good = got == want
        ok &= good
        print(f"  {what:<34} published {fmt.format(want):>9}   recomputed {fmt.format(got):>9}"
              f"   {'ok' if good else 'MISMATCH'}")

    for m, (wa, wh) in {"P2Rank": (47, 63), "fpocket": (50, 63),
                        "IF-SitePred": (46, 68), "consensus": (48, 65)}.items():
        chk(f"{m} AF-Holo/AF-Apo (rounded)",
            f"{round(d6[m][1])}/{round(d6[m][0])}", f"{wh}/{wa}")
    for what, got, want in [
            ("scored detection sites", n6[0], 86946),
            ("classified sites", n6[1], 86651),
            ("UniProt accessions", n6[2], 10419),
            ("UniProt clusters", n6[3], 10395),
            ("AF-Apo sites", n6[4], 34900),
            ("AF-Holo sites", n6[5], 51751),
            ("tied sites", n6[6], 295),
            ("docked pockets (group 1)", n6[7], 60893),
            ("docked with a definite class", n6[8], 60689),
            ("docking AF-Holo (rounded)", round(k6[1]), 69),
            ("docking AF-Apo (rounded)", round(k6[0]), 49)]:
        chk(what, got, want, "{:,}")
    print(f"\n  {'PASS' if ok else 'FAIL'}")

    print("\n" + "=" * 78)
    print("  v4 against v6 -- what the masters have to change")
    print("=" * 78)
    lbl = ["scored detection sites", "classified sites", "UniProt accessions",
           "UniProt clusters", "AF-Apo sites", "AF-Holo sites", "tied sites",
           "docked pockets (group 1)", "docked with a definite class"]
    for i, t in enumerate(lbl):
        print(f"  {t:<32}{n6[i]:>9,}  ->{n4[i]:>9,}")
    print()
    for m in d4:
        print(f"  {m + ' gap (pp)':<32}{d6[m][2]:>9.2f}  ->{d4[m][2]:>9.2f}"
              f"   ({d4[m][2] - d6[m][2]:+.2f})")
    print(f"  {'docking gap (pp)':<32}{k6[2]:>9.2f}  ->{k4[2]:>9.2f}   ({k4[2] - k6[2]:+.2f})")


if __name__ == "__main__":
    main()
