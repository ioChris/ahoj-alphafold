# -*- coding: utf-8 -*-
"""
Regenerate manuscript v4 **Table 1** -- the analysis-set tally, as markdown.

v4 reduces Table 1 to counts. The site-size columns (mean / weighted mean / median / weighted
median) move to the SI with the passage that discussed them, which is MN's C16 and ledger item A4.
What remains is threshold, proteins, distinct accessions and sites.

Three column definitions, because two of them look interchangeable and are not:

  * **Proteins** -- the UniProt accessions in the analysis set. Fixed by FILTERING, so it is the
    same 12,255 at every threshold by construction; clustering never adds or removes a protein.
  * **Distinct accessions (incl. interface partners)** -- every accession appearing on any site at
    that threshold. This GROWS with clustering, because a unified binding site can span an
    interface and so carry more than one accession.
  * **Sites / UBSs** -- rows at that threshold, TIES INCLUDED, which is what the published Table 1
    does and says. The tie-excluded counts are SI Table S1b and differ: 236,861 vs 237,922 at the
    unclustered level.

`uniprots_str` is SPACE-delimited, not comma-delimited -- splitting it the usual way silently
yields one long token per interface site and undercounts the accession column.

Usage (from the repository root; reads cfg.ANALYSIS_UNIFIED_LEVELS_CSV):
    python plotting/manuscript_v4/make_table1_v4.py --out Table1_analysis_set.md
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mv4_paths import cfg  # noqa: E402  (common/config.py of this repository)

ROWS = [("UNCLUSTERED", "entry"), ("ST_100", "ubs"), ("ST_070", "ubs"), ("ST_050", "ubs")]
LABEL = {"UNCLUSTERED": "UNCLUSTERED", "ST_100": "ST_100", "ST_070": "ST_070", "ST_050": "ST_050"}


def accessions(series):
    out = set()
    for v in series.dropna().astype(str):
        out.update(t for t in v.split() if t)      # SPACE-delimited
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", default=None, help="write the markdown here (default: stdout only)")
    a = ap.parse_args()

    d = pd.read_csv(cfg.ANALYSIS_UNIFIED_LEVELS_CSV, low_memory=False,
                    usecols=["threshold", "level", "uniprots_str"])
    # the analysis set's protein count: every accession present before clustering
    proteins = len(accessions(d[(d.threshold == "UNCLUSTERED") & (d.level == "entry")]["uniprots_str"]))

    lines = ["| Threshold | Proteins | Distinct accessions (incl. interface partners) | Sites / UBSs |",
             "|---|---|---|---|"]
    for thr, lv in ROWS:
        s = d[(d.threshold == thr) & (d.level == lv)]
        lines.append(f"| {LABEL[thr]} | {proteins:,} | {len(accessions(s['uniprots_str'])):,} "
                     f"| {len(s):,} |")
    md = "\n".join(lines)
    print(md)
    print("\n(ties included, as the published Table 1 states; the tie-excluded version is Table S1b)")
    if a.out:
        with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(md + "\n")
        print("\nwrote", a.out)


if __name__ == "__main__":
    main()
