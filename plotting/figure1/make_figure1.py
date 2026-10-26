"""Figure 1 - natcomm_v4_rev1 (v3 "chain" layout, promoted 2026-09-10).

    python make_figure1.py --out_dir <dir>                 # the manuscript figure
    python make_figure1.py --out_dir <dir> --pptx          # + the editable deck
    python make_figure1.py --variant grid --out_dir <dir>  # the rejected GA-grid panel b, kept for reference

Writes to --out_dir (default: beside this script):
    figure1_full.{png,svg}          the manuscript figure, 180 x 154 mm, all four panels
    panel_b.{svg,pdf,png}           panel b standalone
    panel_c_flow.*, panel_d_composition.*   data panels standalone
    (--variant grid writes figure1_grid_full.* and panel_b_grid.* instead)
and, with --pptx, <out_dir>/../assembly/figure1_editable.pptx
    (panel a as native shapes; b, c, d placed as images).

Panel b
-------
The five-box chain of the 2026-08-24 figure, with the four metrics named inside
the chain, the SASA KDE of the worked example (2yiw A I46 1354, MAPK14), ONE
per-metric bar (SASA, P(apo) = 0.95) and the averaged verdict bar (0.80 ->
AF-Apo), the confidence note (= the winning probability, reported with its
density and separation supports) and the two one-line definitions.  The
example's identity, n_apo / n_holo, the tie exclusion and the scope sentence
are caption material (plan_v4 FIGURE_MANIFEST sec.20).  "grid" is the
graphical-abstract layout that was considered and rejected as too much visual
load for an opening figure.

Shared fixes over the 2026-08-24 render (author's review, 2026-09-10)
---------------------------------------------------------------------
* Panel a: the three arrows from the query box are symmetric - the card
  centres sit at -16.5 / 0 / +16.5 mm from the box centre and all three
  arrows end on one line.  Since 2026-10-07 they are also identical: each
  drops straight down from the box's lower edge above its own label, so all
  three have one length and one angle (the author's review; from a single
  origin, equal lengths and a common end line cannot both hold - the middle
  arrow was a short vertical stub between two long diagonals).
* Panel a: the strip says "AHoJ-DB" with a small version note, and quotes the
  set this run PROCESSED (516,794 pockets, 121,227 PDB entries, 37,662
  ligands - all recomputed from the job lists), so panel a and the first bar
  of panel c are the same number.  The released database's own statistics
  (515,463 / 121,029 / 37,620 / 29,463 UniProts) are a different denominator
  and belong in Methods, with the +4 / +1 residual footnoted there.
* Panel d: AF-Apo / AF-Holo sit at the two ends of the bar they colour.
* Layout: panel d starts under its letter with a 56 mm axis; panel c's
  reduction labels sit beside their totals; row and column gutters 8-10 mm;
  canvas 180 x 154 mm.

Type: Helvetica Neue; letters 8 pt bold lowercase; titles 7; body 6.5;
ticks/notes 6; floor 6.  No arrow glyphs; every arrow is drawn.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.colors import LinearSegmentedColormap
import json
import os

import sys as _sys
from pathlib import Path as _Path
for _root in _Path(__file__).resolve().parents:
    if (_root / "common" / "config.py").is_file():
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))
        break
from common import config as cfg
from common import figspec

MM = figspec.MM
DPI = 600
W_MM, H_MM = figspec.W2, 154.0

FONT = figspec.FONT_STACK[0]
FS_LETTER = figspec.FS_LETTER
FS_TITLE = figspec.FS_TITLE
FS_BODY = figspec.FS_BODY
FS_TICK = figspec.FS_TICK
FS_NOTE = figspec.FS_NOTE

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": figspec.FONT_STACK,
    "font.size": FS_BODY,
    "axes.linewidth": 0.6,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

APO, HOLO, AF = figspec.APO, figspec.HOLO, figspec.MODEL
GREY, LGREY, EDGE, DARK = "#5a5a5a", "#d9d9d9", "#9aa5b1", "#c6ced6"
FILL_APO, FILL_HOLO, FILL_AF, FILL_N = "#e8f1f9", "#fdf1e3", "#eaf6ea", "#f4f6f8"

CARD = {
    "apo":  dict(border=APO,  fills=["#D6E6F4", "#C2DAEE", "#AECEE9"], img="struct_apo.png"),
    "holo": dict(border=HOLO, fills=["#FBE3CC", "#F9D4B0", "#F7C795"], img="struct_holo.png"),
    "af":   dict(border=AF,   fills=["#DDEFDD"],                       img="struct_af.png"),
}
ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def _cropped_struct(name):
    from PIL import Image
    src = os.path.join(ASSETS, name)
    dst = os.path.join(ASSETS, name.replace(".png", "_crop.png"))
    if not os.path.exists(dst) or os.path.getmtime(dst) < os.path.getmtime(src):
        im = Image.open(src).convert("RGBA")
        bbox = im.split()[-1].getbbox()
        pad = int(0.02 * max(im.size))
        box = (max(bbox[0] - pad, 0), max(bbox[1] - pad, 0),
               min(bbox[2] + pad, im.size[0]), min(bbox[3] + pad, im.size[1]))
        im.crop(box).save(dst)
    w, h = Image.open(dst).size
    return dst, h / w


# panel slots, mm, y from top
SLOT = {"a": (4, 4, 48, 60), "b": (60, 4, 116, 60),
        "c": (4, 72, 76, 82), "d": (90, 72, 86, 82)}
LETTER_POS = {"a": (1.5, 6.5), "b": (57.5, 6.5), "c": (1.5, 74.5), "d": (87.5, 74.5)}
SUBTITLE_Y = {"row1": 7.0, "row2": 75.0}

# --------------------------------------------------------------------- numbers
# Panel a quotes the set this run PROCESSED, i.e. the same 516,794 that opens panel c.
# PDB entries and ligands are parsed from the job names (<pdb>-<chain>-<ligand>-<resnum>).
PROCESSED = dict(pockets=516794, pdbs=121227, ligs=37662, version="v2c")
FLOW = [
    (516794, None, None),
    (464869, 51925, "no usable AHoJ result"),
    (438143, 26726, "no AlphaFold model"),
    (435314, 2829, "no classification"),
    (238738, 196576, "no apo state"),
    (237922, 816, "no usable resolution"),
]
ANALYSIS = dict(pockets=237922, unps=12255)
UBS_SITES = [("ST 100%", 105713), ("ST 70%", 44613), ("ST 50%", 33580), ("ST 30%", 28868)]
LADDER = [
    ("Unclustered", "individual pockets", 90246, 146615, 38.1, 61.9),
    ("UBS  ST 100%", "identical residue sets", 41555, 63791, 39.4, 60.6),
    ("UBS  ST 70%", "Jaccard >= 0.70", 21479, 22982, 48.3, 51.7),
    ("UBS  ST 50%", "Jaccard >= 0.50", 17257, 16202, 51.6, 48.4),
    ("UBS  ST 30%", "Jaccard >= 0.30", 15249, 13513, 53.0, 47.0),
]

# The worked example (panel b).  Per-metric P(apo) and the averaged verdict are
# the pipeline's own numbers for 2yiw A I46 1354 (MAPK14), read from
# classification_kde_pocket.json at render time - nothing here is retyped.
EXAMPLE_DIR = _Path(cfg.EXAMPLE_ENTRIES_DIR) / "093" / "2yiw-A-I46-1354" / "statistics"
EXAMPLE_CSV = EXAMPLE_DIR / "global_results_plots.csv"
EXAMPLE_KDE_JSON = EXAMPLE_DIR / "classification_kde_pocket.json"
EXAMPLE_ID = "2yiw A I46 1354"
EXAMPLE_PROTEIN = "MAPK14"
# (csv column, json key, row label, unit)
METRICS = [
    ("sasa",       "sasa",       "pocket SASA",        "Å²"),
    ("pocket_rms", "pocket_rms", "pocket RMSD",        "Å"),
    ("volume",     "volume",     "pocket volume",      "Å³"),
    ("RoG",        "RoG",        "radius of gyration", "Å"),
]


def _example():
    """Real run_3 values for the 2yiw example: per-metric apo/holo arrays, the
    model value, and the pipeline's per-metric and overall P(apo)."""
    d = pd.read_csv(EXAMPLE_CSV, low_memory=False)
    exp = d[d.role.isin(["C", "Q"])]
    kde = json.load(open(EXAMPLE_KDE_JSON))
    out = {}
    for col, key, label, unit in METRICS:
        apo = exp.loc[exp.apoholo_assignment == "A", col].dropna().to_numpy(float)
        holo = exp.loc[exp.apoholo_assignment != "A", col].dropna().to_numpy(float)
        model = float(d.loc[d.role == "M", col].iloc[0])
        out[key] = dict(apo=apo, holo=holo, model=model, label=label, unit=unit,
                        p_apo=float(kde[key]["P(apo)"]), p_holo=float(kde[key]["P(holo)"]),
                        json_model=float(kde[key]["model_value"]))
    n_apo = int((exp.apoholo_assignment == "A").sum())
    n_holo = int((exp.apoholo_assignment != "A").sum())
    ov = kde["overall"]
    return out, n_apo, n_holo, float(ov["avg_P(apo)"]), str(ov["overall_classification"])


# ------------------------------------------------------------- verify, do not retype
def verify_numbers(strict=True):
    out, bad = [], []

    def chk(label, got, want, fmt="{:>9,}"):
        ok = (got == want)
        out.append(f"    {'ok ' if ok else 'BAD'}  {label:34s} source {fmt.format(got)}  "
                   f"figure {fmt.format(want)}")
        if not ok:
            bad.append(f"{label}: source {got} != figure {want}")

    merged = _Path(cfg.CLASSIFICATION_MERGED_DIR)
    jobs = []
    for f in ("Success_jobs.txt", "Discarded_jobs.txt"):
        for l in open(merged / f, encoding="utf-8", errors="replace"):
            l = l.strip()
            if l:
                jobs.append(l.split(":")[0].strip())
    n_no = sum(1 for l in open(merged / "Discarded_jobs.txt", encoding="utf-8",
                               errors="replace") if l.strip())
    chk("processed (success+discarded)", len(jobs), FLOW[0][0])
    chk("panel a pockets = panel c start", PROCESSED["pockets"], FLOW[0][0])
    pdbs, ligs = set(), set()
    for j in jobs:
        p = j.split("-")
        pdbs.add(p[0].lower()); ligs.add("-".join(p[2:-1]).upper())
    chk("processed PDB entries", len(pdbs), PROCESSED["pdbs"])
    chk("processed ligands", len(ligs), PROCESSED["ligs"])
    chk("no usable AHoJ result", n_no, FLOW[1][1])

    rep = pd.read_csv(_Path(cfg.FILTERED_SURVIVORS_DIR) / "filter_report.csv")
    by = {r.step: r for r in rep.itertuples()}
    for step, idx in [("00_no_model", 2), ("01_missing_pocket_probs", 3),
                      ("03_min_n_apo", 4), ("05_max_query_resolution", 5)]:
        chk(FLOW[idx][2], int(by[step].sequential_removed_rows), FLOW[idx][1])
        chk(FLOW[idx][2] + " -> kept", int(by[step].sequential_kept_rows), FLOW[idx][0])
    chk("analysis UniProts", int(by["05_max_query_resolution"].sequential_kept_unique_unps),
        ANALYSIS["unps"])

    lv = pd.read_csv(cfg.ANALYSIS_UNIFIED_LEVELS_CSV, low_memory=False,
                     usecols=["threshold", "level", "pocket_class"])
    lv["cls"] = lv["pocket_class"].astype(str).str.lower()
    for name, want in UBS_SITES:
        thr = "ST_" + name.split()[1].rstrip("%").rjust(3, "0")
        chk(f"sites {name}", int(((lv.threshold == thr) & (lv.level == "ubs")).sum()), want)
    for lab, _crit, a_want, h_want, _ap, _hp in LADDER:
        thr = "UNCLUSTERED" if lab == "Unclustered" else "ST_" + lab.split()[-1].rstrip("%").rjust(3, "0")
        lev = "entry" if thr == "UNCLUSTERED" else "ubs"
        sub = lv[(lv.threshold == thr) & (lv.level == lev)]
        chk(f"{lab} apo", int((sub.cls == "apo").sum()), a_want)
        chk(f"{lab} holo", int((sub.cls == "holo").sum()), h_want)

    # panel b: the JSON's model value must equal the per-structure table's, so the
    # curve (CSV) and the probability (JSON) describe the same model
    ex, n_apo, n_holo, p_ov, cls = _example()
    for key, e in ex.items():
        chk(f"2yiw model {key}", round(e["model"], 2), round(e["json_model"], 2), "{:>9.2f}")
        chk(f"2yiw P(apo)+P(holo) {key}", round(e["p_apo"] + e["p_holo"], 2), 1.0, "{:>9.2f}")
    chk("2yiw overall = mean of four", round(float(np.mean([e["p_apo"] for e in ex.values()])), 2),
        round(p_ov, 2), "{:>9.2f}")
    chk("2yiw n_apo", n_apo, 148)
    chk("2yiw n_holo", n_holo, 117)
    chk("2yiw class is Apo", int(cls.lower() == "apo"), 1)

    print("  number check:")
    print("\n".join(out))
    if bad:
        msg = ("figure numbers disagree with run_3:\n  " + "\n  ".join(bad))
        if strict:
            raise SystemExit("ABORT -- " + msg)
        print("  ! " + msg)
    else:
        print(f"  all {len(out)} numbers reproduce from source")
    return not bad


# ===================================================================== panel a
def _a_layout():
    """Panel a as shapes (mm, y from top); consumed by matplotlib and pptx alike."""
    ax0, ay0, aw, _ = SLOT["a"]
    boxes, arrows, labels, images, lines = [], [], [], [], []

    labels.append(dict(x=ax0, y=SUBTITLE_Y["row1"],
                       t="one AHoJ-DB entry  ·  one UniProt accession", s=FS_NOTE,
                       c=GREY, align="left"))
    qx, qy, qw, qh = ax0 + 4, ay0 + 6.0, 40, 10.0
    qc = qx + qw / 2
    boxes.append(dict(x=qx, y=qy, w=qw, h=qh,
                      t="query holo pocket\n(biologically relevant ligand)",
                      fill=FILL_HOLO, edge=HOLO, bold=True, s=FS_NOTE))
    trio = [("apo", "apo states\n" + figspec.N_APO, -16.5),
            ("holo", "holo states\n" + figspec.N_HOLO, 0.0),
            ("af", "AlphaFold\nmodel", 16.5)]
    cw, chh, off = 14.0, 10.0, 1.2
    cy = ay0 + 32.0
    nmax = max(len(CARD[k]["fills"]) for k, *_ in trio)
    y_arrow_end = cy - (nmax - 1) * off - 9.0        # one line for all three arrows
    fronts = []
    for key, label, dx in trio:
        spec = CARD[key]
        n = len(spec["fills"])
        # the VISUAL centre of a stack (front card + offset back cards) sits at qc + dx
        x = qc + dx - cw / 2 + (n - 1) * off / 2
        for k in range(n - 1, -1, -1):
            boxes.append(dict(x=x - k * off, y=cy - k * off, w=cw, h=chh, t="",
                              fill=spec["fills"][k], edge=spec["border"], bold=False,
                              s=FS_NOTE))
        fronts.append((x, cy, cw, chh))
        ipath, aspect = _cropped_struct(spec["img"])
        iw = cw - 1.6
        ih = iw * aspect
        if ih > chh - 1.6:
            ih = chh - 1.6
            iw = ih / aspect
        images.append(dict(path=ipath, x=x + (cw - iw) / 2,
                           y=cy + (chh - ih) / 2, w=iw, h=ih))
        labels.append(dict(x=qc + dx, y=cy - (nmax - 1) * off - 3.8, t=label,
                           s=FS_NOTE, c=spec["border"], align="center", bold=True))
        # straight down from the box edge above the label (|dx| = 16.5 < half the box, 20 mm)
        arrows.append((qc + dx, qy + qh, qc + dx, y_arrow_end))
    # bracket under the two experimental stacks, and the provenance tags
    bl = fronts[0][0] - (len(CARD["apo"]["fills"]) - 1) * off
    br = fronts[1][0] + cw
    by = cy + chh + 2.2
    lines += [(bl, by, br, by), (bl, by, bl, by - 1.4), (br, by, br, by - 1.4)]
    labels.append(dict(x=(bl + br) / 2, y=by + 3.0,
                       t="experimental range  ·  PDB", s=FS_NOTE, c=GREY,
                       align="center"))
    labels.append(dict(x=qc + trio[2][2], y=by + 3.0, t="AlphaFold DB", s=FS_NOTE, c=AF,
                       align="center"))
    # the processed set, three cells; version as a small note beside the name
    sx, sy, sw, sh = ax0 + 1, ay0 + 49.5, aw - 2, 11.0
    boxes.append(dict(x=sx, y=sy, w=sw, h=sh, t="", fill="#f0f2f4", edge=EDGE,
                      bold=False, s=FS_NOTE))
    labels.append(dict(x=sx + sw / 2 - 2.2, y=sy + 2.7, t="AHoJ-DB", s=FS_NOTE,
                       c="#000000", align="center", bold=True))
    labels.append(dict(x=sx + sw / 2 + 4.6, y=sy + 2.7, t=PROCESSED["version"], s=FS_NOTE,
                       c=GREY, align="center"))
    cells = [(f"{PROCESSED['pockets']:,}", "query pockets"),
             (f"{PROCESSED['pdbs']:,}", "PDB entries"),
             (f"{PROCESSED['ligs']:,}", "ligands")]
    for i, (v, k) in enumerate(cells):
        cx = sx + sw * (i + 0.5) / 3.0
        labels.append(dict(x=cx, y=sy + 6.4, t=v, s=FS_BODY, c="#000000", align="center",
                           bold=True))
        labels.append(dict(x=cx, y=sy + 9.5, t=k, s=FS_NOTE, c=GREY, align="center"))
    return boxes, arrows, labels, images, lines


A_BOXES = A_ARROWS = A_LABELS = A_IMAGES = A_LINES = None


# ====================================================== panel b, shared pieces
GRAD = LinearSegmentedColormap.from_list("apoholo", [APO, "#ffffff", HOLO])
B_BAR_H = 3.0


def _gradient_bar(ax, x0, x1, yc, p_holo, value="", bold=False, h=B_BAR_H):
    """Blue-white-orange bar with the model marked at P(holo); P(apo) printed right."""
    g = np.linspace(0, 1, 256)[None, :]
    ax.imshow(g, cmap=GRAD, extent=(x0, x1, yc + h / 2, yc - h / 2),
              zorder=2, interpolation="bilinear", vmin=0, vmax=1)
    ax.add_patch(Rectangle((x0, yc - h / 2), x1 - x0, h, fill=False, ec=EDGE, lw=0.5, zorder=3))
    xm = x0 + (x1 - x0) * p_holo
    ax.add_patch(Rectangle((xm - 0.45, yc - h / 2 - 0.5), 0.9, h + 1.0,
                           fc="#000000", ec="none", zorder=4))
    if value:
        ax.text(x1 + 1.8, yc, value, fontsize=FS_BODY, va="center", ha="left", zorder=5,
                fontweight="bold" if bold else "normal")


def _kde_curves(e):
    from scipy.stats import gaussian_kde
    apo, holo, model = e["apo"], e["holo"], e["model"]
    allv = np.concatenate([apo, holo])
    # crop to the 0.5-99.5 % range of the pooled states (plus the model), so a
    # handful of outliers does not push the curves into a corner of the panel
    lo = min(np.quantile(allv, 0.005), model)
    hi = max(np.quantile(allv, 0.995), model)
    pad = 0.06 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    xs = np.linspace(lo, hi, 400)
    ya, yh = gaussian_kde(apo)(xs), gaussian_kde(holo)(xs)
    top = max(ya.max(), yh.max())
    return xs, ya / top, yh / top, model, lo, hi


def _draw_kde(ax, e, ymax, labelled, xlabel=None):
    xs, ya, yh, model, lo, hi = _kde_curves(e)
    ax.fill_between(xs, ya, color=APO, alpha=0.33, lw=0)
    ax.fill_between(xs, yh, color=HOLO, alpha=0.33, lw=0)
    ax.plot(xs, ya, color=APO, lw=0.9)
    ax.plot(xs, yh, color=HOLO, lw=0.9)
    ax.axvline(model, color=AF, lw=1.1, ls=(0, (2.6, 1.6)))
    if labelled:
        # each class label sits beside its own peak, on the side AWAY from the model
        # line, so neither label crosses the dashed line however close the model sits
        rng = hi - lo
        for arr, name, col in ((ya, "apo", APO), (yh, "holo", HOLO)):
            i = int(np.argmax(arr))
            px, py = xs[i], arr[i]
            if abs(px - model) < 0.12 * rng:
                side = -1 if px <= model else 1
                ax.text(px + side * 0.03 * rng, py + 0.02, name, color=col, fontsize=FS_BODY,
                        fontweight="bold", ha="right" if side < 0 else "left", va="center")
            else:
                ax.text(px, py + 0.08, name, color=col, fontsize=FS_BODY, fontweight="bold",
                        ha="center", va="bottom")
        # the model label goes to whichever side of the line has more room
        room_right = hi - model
        ha = "left" if room_right >= 0.45 * rng else "right"
        dx = 0.025 * rng * (1 if ha == "left" else -1)
        ax.text(model + dx, ymax - 0.02, "AlphaFold model", fontsize=FS_NOTE, color=AF,
                ha=ha, va="top", fontweight="bold")
    ax.set_xlim(lo, hi); ax.set_ylim(0, ymax)
    ax.set_xticks([]); ax.set_yticks([])
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FS_NOTE, labelpad=1.5)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["bottom"].set_color(GREY)


def _b_subtitles(ax, X, Y, n_apo, n_holo, example=False):
    """The example's identity and n_apo / n_holo belong in the caption; the grid
    variant keeps them on the panel (it prints four curves), the chain does not."""
    bx0, by0, bw, bh = SLOT["b"]
    ax.text(X(bx0), Y(SUBTITLE_Y["row1"]), "pocket conformational classification",
            fontsize=FS_NOTE, color=GREY, va="center")
    if example:
        ax.text(X(bx0 + bw), Y(SUBTITLE_Y["row1"]),
                f"worked example: {EXAMPLE_ID} ({EXAMPLE_PROTEIN})  ·  "
                f"{figspec.N_APO} = {n_apo}  ·  {figspec.N_HOLO} = {n_holo}",
                fontsize=FS_NOTE, color=GREY, va="center", ha="right")


def _box(ax, x, y, w, h, text, fill=FILL_N, edge=EDGE, bold=False, s=FS_NOTE, color="#000000"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.9",
                                fc=fill, ec=edge, lw=0.6, zorder=2))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=s, zorder=3,
                linespacing=1.3, fontweight="bold" if bold else "normal", color=color)


# =============================================================== panel b: grid
B_COL = dict(metric=(58.0, 80.0), kde=(80.0, 116.0), bar=(121.0, 147.0), verdict=(160.0, 176.0))
B_HEADER_Y = 13.6
B_ROW0, B_ROWH = 23.0, 9.6
B_KDE_H = 7.6


def draw_panel_b_grid(fig, ax, ox=0.0, oy=0.0, W=W_MM, H=H_MM):
    ex, n_apo, n_holo, p_ov, cls = _example()
    bx0, by0, bw, bh = SLOT["b"]
    X = lambda v: v - ox
    Y = lambda v: v - oy
    mx0, mx1 = B_COL["metric"]; kx0, kx1 = B_COL["kde"]
    px0, px1 = B_COL["bar"]; vx0, vx1 = B_COL["verdict"]
    _b_subtitles(ax, X, Y, n_apo, n_holo, example=True)

    hy = Y(B_HEADER_Y)
    ax.text(X(mx0), hy, "pocket metric", fontsize=FS_NOTE, va="center", fontweight="bold")
    ax.text(X((kx0 + kx1) / 2), hy, "apo / holo density of the experimental states",
            fontsize=FS_NOTE, va="center", ha="center", fontweight="bold")
    ax.text(X((px0 + px1) / 2), hy, "P(apo) per metric", fontsize=FS_NOTE, va="center",
            ha="center", fontweight="bold")
    ax.text(X((vx0 + vx1) / 2), hy, "pocket verdict", fontsize=FS_NOTE, va="center",
            ha="center", fontweight="bold")
    ax.plot([X(mx0), X(vx1)], [hy + 2.3, hy + 2.3], color=EDGE, lw=0.5)
    # legend strip between the header rule and row 1, drawn once
    ly = hy + 4.8
    lx = X(mx0)
    for col, name in ((APO, "apo"), (HOLO, "holo")):
        ax.add_patch(Rectangle((lx, ly - 0.9), 3.0, 1.8, fc=col, alpha=0.4, ec=col, lw=0.6))
        ax.text(lx + 4.0, ly, name, fontsize=FS_NOTE, color=col, va="center", fontweight="bold")
        lx += 4.0 + len(name) * 1.25 + 3.0
    ax.plot([lx + 0.5, lx + 0.5], [ly - 1.4, ly + 1.4], color=AF, lw=1.1, ls=(0, (2.6, 1.6)))
    ax.text(lx + 2.5, ly, "AlphaFold model", fontsize=FS_NOTE, color=AF, va="center",
            fontweight="bold")
    ax.text(X(px0), ly, "apo", fontsize=FS_NOTE, color=APO, va="center", ha="left",
            fontweight="bold")
    ax.text(X(px1), ly, "holo", fontsize=FS_NOTE, color=HOLO, va="center", ha="right",
            fontweight="bold")

    for i, (col, key, label, unit) in enumerate(METRICS):
        e = ex[key]
        yc = B_ROW0 + i * B_ROWH
        ax.text(X(mx0), Y(yc) - 1.4, label, fontsize=FS_BODY, va="center", fontweight="bold")
        ax.text(X(mx0), Y(yc) + 1.9, unit, fontsize=FS_NOTE, va="center", color=GREY)
        kax = fig.add_axes([X(kx0) / W, 1 - Y(yc + 3.6) / H, (kx1 - kx0) / W, B_KDE_H / H])
        _draw_kde(kax, e, ymax=1.18, labelled=False)
        _gradient_bar(ax, X(px0), X(px1), Y(yc), e["p_holo"], f"{e['p_apo']:.2f}")

    y_top, y_bot = Y(B_ROW0), Y(B_ROW0 + (len(METRICS) - 1) * B_ROWH)
    y_mid = (y_top + y_bot) / 2
    bxx = X(px1) + 9.2
    ax.plot([bxx - 1.2, bxx, bxx, bxx - 1.2], [y_top, y_top, y_bot, y_bot], color=GREY,
            lw=0.7, solid_capstyle="butt")
    _arrow_mm(ax, bxx, y_mid, X(vx0) - 0.8, y_mid)
    vc = X((vx0 + vx1) / 2)
    ax.text(vc, y_mid - 6.4, "average of the\nfour metrics", fontsize=FS_NOTE,
            color=GREY, ha="center", va="center", linespacing=1.3)
    _gradient_bar(ax, X(vx0), X(vx1), y_mid, 1.0 - p_ov)
    ax.text(vc, y_mid + 4.8, f"P(apo) = {p_ov:.2f}", fontsize=FS_BODY,
            ha="center", va="center", fontweight="bold")
    ax.text(vc, y_mid + 8.8, f"AF-{cls.capitalize()}", fontsize=FS_TITLE,
            ha="center", va="center", fontweight="bold", color=APO if cls.lower() == "apo" else HOLO)

    ny = Y(by0 + bh - 8.2)
    ax.text(X(bx0), ny, "confidence:  density  =  experimental support where the model sits",
            fontsize=FS_NOTE, color=GREY, va="center")
    ax.text(X(bx0) + 16.2, ny + 3.2,
            "separation  =  how far apart the apo and holo distributions lie",
            fontsize=FS_NOTE, color=GREY, va="center")
    ax.text(X(bx0), ny + 6.4, "applied to every entry with at least one experimental "
            "counterpart", fontsize=FS_NOTE, color=GREY, va="center")


# ============================================================== panel b: chain
def draw_panel_b_chain(fig, ax, ox=0.0, oy=0.0, W=W_MM, H=H_MM):
    """v1's chain of boxes, plus: the metric names inside the chain, one
    per-metric bar (SASA) and the averaged verdict beside v1's SASA KDE."""
    ex, n_apo, n_holo, p_ov, cls = _example()
    bx0, by0, bw, bh = SLOT["b"]
    X = lambda v: v - ox
    Y = lambda v: v - oy
    _b_subtitles(ax, X, Y, n_apo, n_holo)

    # ---- the chain (v1), 5 boxes, 20 x 13 mm, 4 mm gaps -> exactly 116 mm
    chain = ["pocket residues\nof every state\nand the model",
             "4 pocket metrics\nvolume · SASA\nRoG · pocket RMSD",
             "apo / holo\nKDE densities",
             "P(apo), P(holo)\nper metric",
             "averaged:\nclass +\nconfidence"]
    cw, cgap, cy0, ch = 20.0, 4.0, by0 + 8.0, 13.0
    for i, t in enumerate(chain):
        x = bx0 + i * (cw + cgap)
        _box(ax, X(x), Y(cy0), cw, ch, t)
        if i:
            _arrow_mm(ax, X(x - cgap), Y(cy0 + ch / 2), X(x), Y(cy0 + ch / 2))

    # ---- below the chain: v1's SASA KDE (left) | the two bars (middle) | notes (right)
    ky0, kh = cy0 + ch + 5.0, 23.5           # KDE block top and height, mm
    kx0, kw = bx0 + 2.0, 36.0
    kax = fig.add_axes([X(kx0) / W, 1 - Y(ky0 + kh - 3.5) / H, kw / W, (kh - 3.5) / H])
    _draw_kde(kax, ex["sasa"], ymax=1.30, labelled=True,
              xlabel="pocket SASA  ($\\mathregular{\\AA^2}$)")

    # bars block: SASA's own P(apo), then the average of the four
    bx = bx0 + 44.0
    lab_x, bar0, bar1 = bx, bx + 12.0, bx + 35.0
    ty = ky0 + 1.5
    ax.text(X(lab_x), Y(ty), "P(apo)", fontsize=FS_NOTE, va="center", fontweight="bold")
    ax.text(X(bar0), Y(ty), "apo", fontsize=FS_NOTE, color=APO, va="center", ha="left",
            fontweight="bold")
    ax.text(X(bar1), Y(ty), "holo", fontsize=FS_NOTE, color=HOLO, va="center", ha="right",
            fontweight="bold")
    e = ex["sasa"]
    y1 = ty + 6.0
    ax.text(X(lab_x), Y(y1), "SASA", fontsize=FS_NOTE, va="center")
    _gradient_bar(ax, X(bar0), X(bar1), Y(y1), e["p_holo"], f"{e['p_apo']:.2f}")
    ax.text(X(lab_x), Y(y1 + 3.2), "(one of four)", fontsize=FS_NOTE, va="center", color=GREY)
    y2 = y1 + 9.5
    ax.text(X(lab_x), Y(y2 - 1.6), "average of", fontsize=FS_NOTE, va="center")
    ax.text(X(lab_x), Y(y2 + 1.6), "four metrics", fontsize=FS_NOTE, va="center")
    _gradient_bar(ax, X(bar0), X(bar1), Y(y2), 1.0 - p_ov, f"{p_ov:.2f}", bold=True)
    ax.text(X((bar0 + bar1) / 2), Y(y2 + 6.2), f"AF-{cls.capitalize()}", fontsize=FS_TITLE,
            ha="center", va="center", fontweight="bold",
            color=APO if cls.lower() == "apo" else HOLO)

    # notes column at the right, as in v1.  "Confidence" is the winning probability -
    # the quantity the Results later threshold - and density / separation are the
    # two supports reported beside it, not its definition.
    nx = bx0 + 91.0
    ax.text(X(nx), Y(ty), "confidence", fontsize=FS_NOTE, va="center", fontweight="bold")
    ax.text(X(nx), Y(ty + 4.0),
            f"the winning probability,\nhere P(apo) = {p_ov:.2f}",
            fontsize=FS_NOTE, color=GREY, va="top", linespacing=1.35)
    ax.text(X(nx), Y(ty + 12.0),
            "reported together with\nits density and\nseparation supports",
            fontsize=FS_NOTE, color=GREY, va="top", linespacing=1.35)
    # the two definitions, one line each, on a band under the inset and the bars
    dy = by0 + bh - 5.6
    ax.text(X(bx0), Y(dy), "density  =  experimental support where the model sits",
            fontsize=FS_NOTE, color=GREY, va="center")
    ax.text(X(bx0), Y(dy + 3.4),
            "separation  =  how far apart the apo and holo distributions lie",
            fontsize=FS_NOTE, color=GREY, va="center")


# variant -> (full-figure stem, panel-b drawer, panel-b stem)
VARIANTS = {"chain": ("figure1_full", draw_panel_b_chain, "panel_b"),
            "grid": ("figure1_grid_full", draw_panel_b_grid, "panel_b_grid")}


# ===================================================================== panel c
def draw_panel_c(ax, w=None, h=None):
    w = w or SLOT["c"][2]; h = h or SLOT["c"][3]
    ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.axis("off")
    COL_BAR, COL_TOT = 1.0, 40.0
    COL_LAB = COL_TOT + 3.0
    BAR_MAX = 25.0

    ax.text(COL_BAR, SUBTITLE_Y["row2"] - SLOT["c"][1], "pockets processed in this run",
            fontsize=FS_NOTE, color=GREY, va="center")
    top, rowh = 7.0, 6.4
    maxc = FLOW[0][0]
    for i, (kept, removed, label) in enumerate(FLOW):
        y = top + i * rowh
        if removed is not None:
            ax.text(COL_LAB, y - rowh / 2, f"\u2212{removed:,}   {label}",
                    fontsize=FS_NOTE, color=GREY, va="center", ha="left")
        bw = BAR_MAX * kept / maxc + 4.0
        endpoint = i in (0, len(FLOW) - 1)
        ax.add_patch(Rectangle((COL_BAR, y - 1.9), bw, 3.8,
                               fc=DARK if endpoint else LGREY, ec=EDGE, lw=0.5))
        ax.text(COL_TOT, y, f"{kept:,}", fontsize=FS_BODY, va="center", ha="right",
                fontweight="bold" if endpoint else "normal")
    yend = top + (len(FLOW) - 1) * rowh
    ax.text(COL_BAR, yend + 5.8,
            f"analysis set   {ANALYSIS['pockets']:,} pockets  ·  "
            f"{ANALYSIS['unps']:,} UniProts", fontsize=FS_BODY, va="center",
            fontweight="bold")

    ydiv = yend + 9.6
    ax.plot([COL_BAR, w - 1], [ydiv, ydiv], color=EDGE, lw=0.6, ls=(0, (3, 2)))
    ax.text(COL_BAR, ydiv + 3.6, "redundancy reduction — unified binding sites",
            fontsize=FS_BODY, va="center", fontweight="bold")
    gy = ydiv + 9.0
    for k, dx in enumerate((0.0, 1.9, 3.8)):
        ax.add_patch(Circle((COL_BAR + 2.0 + dx, gy), 1.9, fc=HOLO if k % 2 else APO,
                            alpha=0.45, ec="none"))
    _arrow_mm(ax, COL_BAR + 8.2, gy, COL_BAR + 12.4, gy)
    ax.add_patch(Circle((COL_BAR + 15.2, gy), 2.2, fc="#8e9aa6", alpha=0.75, ec="none"))
    ax.text(COL_BAR + 19.5, gy - 1.6, "pockets of one UniProt with identical residue sets",
            fontsize=FS_NOTE, color=GREY, va="center")
    ax.text(COL_BAR + 19.5, gy + 1.6,
            f"merge into one site  ·  from {ANALYSIS['pockets']:,}",
            fontsize=FS_NOTE, color=GREY, va="center")

    ty, bh = gy + 6.6, 9.4
    colw = (w - 2 * COL_BAR) / 4.0
    ax.add_patch(Rectangle((COL_BAR, ty), w - 2 * COL_BAR, bh, fc="#f0f2f4",
                           ec=EDGE, lw=0.5, zorder=1))
    for i, (name, n) in enumerate(UBS_SITES):
        cx = COL_BAR + colw * (i + 0.5)
        main = i == 0
        if main:
            ax.add_patch(Rectangle((COL_BAR + 0.5, ty + 0.5), colw - 1.0, bh - 1.0,
                                   fc="#ffffff", ec="black", lw=0.9, zorder=2))
        elif i < 4:
            ax.plot([COL_BAR + colw * i] * 2, [ty + 1.3, ty + bh - 1.3],
                    color=EDGE, lw=0.4, zorder=2)
        ax.text(cx, ty + 3.1, name, fontsize=FS_NOTE, ha="center", va="center",
                color="#000000" if main else GREY, zorder=3,
                fontweight="bold" if main else "normal")
        ax.text(cx, ty + 6.6, f"{n:,}", fontsize=FS_BODY, ha="center", va="center",
                zorder=3, fontweight="bold" if main else "normal")
    ax.text(COL_BAR, ty + bh + 3.0, "unified binding sites per clustering threshold  ·  "
            "boxed = used in the main text", fontsize=FS_NOTE, color=GREY, va="center")


def _arrow_mm(ax, x1, y1, x2, y2, lw=0.8, color=GREY):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=4.5, lw=lw, color=color,
                                 shrinkA=0, shrinkB=0))


# ===================================================================== panel d
D_AXES = (25.5, 12.0, 48.0, 56.0)      # inset_left, inset_bottom, w, h (mm inside the slot)


def draw_panel_d(ax):
    y = np.arange(len(LADDER))[::-1]
    for i, (lab, crit, a, h, ap, hp) in enumerate(LADDER):
        yy = y[i]
        ax.barh(yy, ap, color=APO, height=0.62, zorder=3)
        ax.barh(yy, hp, left=ap, color=HOLO, height=0.62, zorder=3)
        ax.text(ap / 2, yy, f"{ap:.1f}", ha="center", va="center", fontsize=FS_BODY,
                color="white", fontweight="bold", zorder=4)
        ax.text(ap + hp / 2, yy, f"{hp:.1f}", ha="center", va="center", fontsize=FS_BODY,
                color="white", fontweight="bold", zorder=4)
        ax.text(103, yy, f"{a + h:,}", fontsize=FS_TICK, va="center", color=GREY)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{l}\n{c}" for l, c, *_ in LADDER], fontsize=FS_TICK,
                       linespacing=1.4)
    ax.set_xlim(0, 100); ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("class composition (%)", fontsize=FS_TITLE, labelpad=2)
    ax.tick_params(labelsize=FS_TICK, length=2, width=0.5, pad=2)
    ax.axvline(50, color="white", lw=0.6, zorder=5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    hi = y[1]
    ax.add_patch(Rectangle((-0.5, hi - 0.46), 101, 0.92, fill=False, ec="black",
                           lw=0.9, zorder=6, clip_on=False))
    ax.get_yticklabels()[list(y).index(hi)].set_fontweight("bold")
    # class labels at the two ends of the bar they colour, one row above the top bar
    ytop = len(LADDER) - 0.36
    ax.text(0, ytop, "AF-Apo", color=APO, fontsize=FS_BODY, fontweight="bold",
            ha="left", va="center")
    ax.text(98, ytop, "AF-Holo", color=HOLO, fontsize=FS_BODY, fontweight="bold",
            ha="right", va="center")
    ax.text(103, ytop, "sites,\nexcl. ties", fontsize=FS_TICK, color=GREY, va="center",
            linespacing=1.25)


# ================================================================ full figure
def _mm_ax(fig, W=W_MM, H=H_MM):
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    return ax


def render_a(ax):
    for b in A_BOXES:
        _box(ax, b["x"], b["y"], b["w"], b["h"], b["t"], fill=b["fill"], edge=b["edge"],
             bold=b["bold"], s=b["s"])
    for im in A_IMAGES:
        arr = plt.imread(im["path"])
        ax.imshow(arr, extent=(im["x"], im["x"] + im["w"],
                               im["y"] + im["h"], im["y"]),
                  zorder=4, interpolation="antialiased")
    for (x1, y1, x2, y2) in A_LINES:
        ax.plot([x1, x2], [y1, y2], color=GREY, lw=0.7, zorder=3, solid_capstyle="butt")
    for (x1, y1, x2, y2) in A_ARROWS:
        _arrow_mm(ax, x1, y1, x2, y2)
    for l in A_LABELS:
        ax.text(l["x"], l["y"], l["t"], fontsize=l["s"], color=l["c"],
                ha={"center": "center", "left": "left"}[l.get("align", "left")],
                va="center", linespacing=1.35, zorder=5,
                fontweight="bold" if l.get("bold") else "normal")


def export_full(path, draw_b):
    fig = plt.figure(figsize=(W_MM * MM, H_MM * MM))
    fig.patch.set_facecolor("white")
    ax = _mm_ax(fig)
    render_a(ax)
    draw_b(fig, ax)
    cx, cy, cw, ch = SLOT["c"]
    draw_panel_c(fig.add_axes([cx / W_MM, 1 - (cy + ch) / H_MM, cw / W_MM, ch / H_MM]))
    dx, dy, dw, dh = SLOT["d"]
    il, ib, iw, ih = D_AXES
    draw_panel_d(fig.add_axes([(dx + il) / W_MM, 1 - (dy + dh - ib) / H_MM,
                               iw / W_MM, ih / H_MM]))
    fig.text(dx / W_MM, 1 - SUBTITLE_Y["row2"] / H_MM,
             "classification outcome, and the effect of redundancy reduction",
             fontsize=FS_NOTE, color=GREY, va="center")
    for k, (lx, ly) in LETTER_POS.items():
        fig.text(lx / W_MM, 1 - ly / H_MM, k, fontsize=FS_LETTER, fontweight="bold",
                 va="baseline")
    for ext in ("png", "svg"):
        fig.savefig(f"{path}.{ext}", dpi=DPI, facecolor="white")
    plt.close(fig)


def export_panel_b(path, draw_b):
    x, y, w, h = SLOT["b"]
    fig = plt.figure(figsize=(w * MM, h * MM)); fig.patch.set_facecolor("white")
    ax = _mm_ax(fig, w, h)
    draw_b(fig, ax, ox=x, oy=y, W=w, H=h)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(f"{path}.{ext}", dpi=DPI, facecolor="white")
    plt.close(fig)


def export_panel_c(path):
    w, h = SLOT["c"][2], SLOT["c"][3]
    fig = plt.figure(figsize=(w * MM, h * MM)); fig.patch.set_facecolor("white")
    draw_panel_c(fig.add_axes([0, 0, 1, 1]))
    for ext in ("svg", "pdf", "png"):
        fig.savefig(f"{path}.{ext}", dpi=DPI, facecolor="white")
    plt.close(fig)


def export_panel_d(path):
    w, h = SLOT["d"][2], SLOT["d"][3]
    il, ib, iw, ih = D_AXES
    fig = plt.figure(figsize=(w * MM, h * MM)); fig.patch.set_facecolor("white")
    draw_panel_d(fig.add_axes([il / w, ib / h, iw / w, ih / h]))
    for ext in ("svg", "pdf", "png"):
        fig.savefig(f"{path}.{ext}", dpi=DPI, facecolor="white")
    plt.close(fig)


# ======================================================================== pptx
def build_pptx(out_path, here, panel_b_png):
    """Panel a as native shapes; b, c, d as placed images."""
    from pptx import Presentation
    from pptx.util import Mm, Pt
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
    from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
    from lxml import etree

    def rgb(h):
        return RGBColor.from_string(h.lstrip("#").upper())

    prs = Presentation()
    prs.slide_width, prs.slide_height = Mm(W_MM), Mm(H_MM)
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    def textbox(x, y, w, h, text, size, color="#000000", bold=False, align=PP_ALIGN.CENTER):
        tb = slide.shapes.add_textbox(Mm(x), Mm(y), Mm(w), Mm(h))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        for i, line in enumerate(text.split("\n")):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = align
            r = p.add_run(); r.text = line
            r.font.name, r.font.size = FONT, Pt(size)
            r.font.bold, r.font.color.rgb = bold, rgb(color)
        return tb

    for b in A_BOXES:
        sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Mm(b["x"]), Mm(b["y"]),
                                    Mm(b["w"]), Mm(b["h"]))
        sh.adjustments[0] = 0.12
        sh.fill.solid(); sh.fill.fore_color.rgb = rgb(b["fill"])
        sh.line.color.rgb = rgb(b["edge"]); sh.line.width = Pt(0.6)
        tf = sh.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        for i, line in enumerate(b["t"].split("\n")):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = PP_ALIGN.CENTER
            r = p.add_run(); r.text = line
            r.font.name, r.font.size = FONT, Pt(b["s"])
            r.font.bold, r.font.color.rgb = b["bold"], rgb("#000000")
    for im in A_IMAGES:
        slide.shapes.add_picture(im["path"], Mm(im["x"]), Mm(im["y"]), Mm(im["w"]), Mm(im["h"]))
    for (x1, y1, x2, y2) in A_LINES:
        cn = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Mm(x1), Mm(y1), Mm(x2), Mm(y2))
        cn.line.color.rgb = rgb(GREY); cn.line.width = Pt(0.7)
    for (x1, y1, x2, y2) in A_ARROWS:
        cn = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Mm(x1), Mm(y1), Mm(x2), Mm(y2))
        cn.line.color.rgb = rgb(GREY); cn.line.width = Pt(0.75)
        ln = cn.line._get_or_add_ln()
        ns = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
        tail = etree.SubElement(ln, f"{ns}tailEnd")
        tail.set("type", "triangle"); tail.set("w", "sm"); tail.set("len", "sm")
    for l in A_LABELS:
        align = {"center": PP_ALIGN.CENTER, "left": PP_ALIGN.LEFT}[l.get("align", "left")]
        nlines = l["t"].count("\n") + 1
        hh = 1.5 * nlines + 1.0
        ww = 46.0
        x = l["x"] - (ww / 2 if align == PP_ALIGN.CENTER else 0)
        textbox(x, l["y"] - hh / 2, ww, hh, l["t"], l["s"], l["c"],
                bold=l.get("bold", False), align=align)
    for key, name in (("b", panel_b_png), ("c", "panel_c_flow.png"),
                      ("d", "panel_d_composition.png")):
        x, y, w, h = SLOT[key]
        slide.shapes.add_picture(os.path.join(here, name), Mm(x), Mm(y), Mm(w), Mm(h))
    textbox(SLOT["d"][0], SUBTITLE_Y["row2"] - 2.0, 80, 4.0,
            "classification outcome, and the effect of redundancy reduction",
            FS_NOTE, GREY, align=PP_ALIGN.LEFT)
    for k, (lx, ly) in LETTER_POS.items():
        textbox(lx, ly - 3.4, 6, 4.2, k, FS_LETTER, "#000000", bold=True, align=PP_ALIGN.LEFT)
    prs.save(out_path)


if __name__ == "__main__":
    import argparse
    A_BOXES, A_ARROWS, A_LABELS, A_IMAGES, A_LINES = _a_layout()
    ap = argparse.ArgumentParser(description="Build manuscript v4 Figure 1.")
    ap.add_argument("--variant", choices=["chain", "grid", "both"], default="chain",
                    help="chain = the manuscript figure (default); grid = the rejected GA-grid panel b")
    ap.add_argument("--out_dir", default=None,
                    help="output folder (default: beside this script)")
    ap.add_argument("--no_verify", action="store_true",
                    help="skip the check that every printed number reproduces from run_3")
    ap.add_argument("--pptx", action="store_true", help="also build the editable deck(s)")
    args = ap.parse_args()
    verify_numbers(strict=not args.no_verify)
    here = args.out_dir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(here, exist_ok=True)
    export_panel_c(os.path.join(here, "panel_c_flow"))
    export_panel_d(os.path.join(here, "panel_d_composition"))
    wanted = ["chain", "grid"] if args.variant == "both" else [args.variant]
    for v in wanted:
        stem, draw_b, bname = VARIANTS[v]
        export_panel_b(os.path.join(here, bname), draw_b)
        export_full(os.path.join(here, stem), draw_b)
        print(f"{v:5s} -> {stem}.* in {here}")
        if args.pptx:
            # the deck is rebuilt from scratch, so it is only written when asked for;
            # CF's hand-edited copy is figure1_editable_working.pptx, never touched here
            assembly = os.path.join(os.path.dirname(here), "assembly")
            os.makedirs(assembly, exist_ok=True)
            p = os.path.join(assembly, "figure1_editable.pptx" if v == "chain"
                             else "figure1_grid_editable.pptx")
            build_pptx(p, here, bname + ".png")
            print("pptx ->", p)

# ---------------------------------------------------------- NUMBER SOURCES
# 516,794 / 51,925 / 464,869 : CLASSIFICATION_MERGED_DIR/{Success,Discarded}_jobs.txt
#     (+ merged_manifest.json).  121,227 PDB entries and 37,662 ligands are parsed from
#     those 516,794 job names, <pdb>-<chain>-<ligand>-<resnum>.
#     The RELEASED database publishes 515,463 / 121,029 / 37,620 / 29,463 UniProts
#     (dev.apoholo.cz/api/db/archive/v2c/db_stats.json): a different denominator
#     (1,327 "Empty dataframe" discards drop at release; +4 entries / +1 PDB unexplained).
#     Quote those in Methods / Data availability, not on the figure.
# 26,726 / 2,829 / 196,576 / 816 / 237,922 / 12,255 : FILTERED_SURVIVORS_DIR/filter_report.csv,
#     sequential order.  Order-dependent; do not reorder without recomputing.
# UBS_SITES : ANALYSIS_UNIFIED_LEVELS_CSV, level=='ubs', INCLUDING ties.
# LADDER : same table, ties EXCLUDED.
# Panel b (2yiw A I46 1354, MAPK14, run_3):
#   curves      statistics/global_results_plots.csv (role C/Q experimental, role M model)
#   P(apo)      statistics/classification_kde_pocket.json per metric; overall = avg_P(apo)
#               (0.799 -> printed 0.80), overall_classification = Apo.  The JSON's
#               model_value is checked against the CSV's role-M row per metric.
#   n_apo/n_holo 148 / 117 from the same CSV.
