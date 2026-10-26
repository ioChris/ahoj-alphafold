# -*- coding: utf-8 -*-
"""
Figure 3B (REV1) — Unified-table-only Sankey (threshold-aware)

Inputs:
  - analysis_unified_levels.csv  OR  analysis_unified_levels.zip (containing the CSV)

Goal:
  For each requested threshold:
    - UNCLUSTERED  -> use level == "entry"
    - clustered    -> use level == "ubs"
  Build a multi-stage Sankey showing progression through:
    Start(class) -> Density(bin) -> Separation(bin) -> Probability(bin) -> Final class

Mapping to unified schema (pocket-mode; default):
  - Final class:              pocket_class
  - KDE probabilities:        pocket_p_apo, pocket_p_holo   (probability = max of the two)
  - Density evidence metric:  pocket_density_pair           (fallbacks: pocket_density_combined, pocket_density_strongest)
  - Overlap metric:           pocket_overlap_pair           (fallbacks: pocket_overlap_overall, pocket_overlap_mode)
                              separation = 1 - overlap

If you want a chain-mode Sankey instead, set MODE = "chain" and the script will use:
  chain_class, chain_p_apo, chain_p_holo, chain_density_combined, chain_overlap_pair, ...

Exports:
  PDF + PNG by default (SVG optional). Requires plotly + kaleido.

Spyder-friendly: user settings are globals below (no CLI).
"""
import os
import zipfile
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.colors import to_rgb, to_hex

# ============================================================
# USER SETTINGS (Spyder)
# ============================================================
# --- locate repo root so `common` is importable (works standalone / in Spyder) ---
import sys as _sys
from pathlib import Path as _Path
for _root in _Path(__file__).resolve().parents:
    if (_root / "common" / "config.py").is_file():
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))
        break
from common import config as cfg
from common import figspec


INPUT_TABLE  = str(cfg.ANALYSIS_UNIFIED_LEVELS_CSV)
OUTPUT_DIR   = str(cfg.PLOTS_DIR / "3B_sankey")
OUT_BASENAME = "Figure3B_sankey_REV1"


# Output
WRITE_PDF   = True
WRITE_PNG   = True
WRITE_SVG   = True
WRITE_HTML  = True
TRANSPARENT_BG = False


# Thresholds to plot (order matters)
#THRESHOLDS_TO_PLOT = ["ST_050"]
THRESHOLDS_TO_PLOT = ["UNCLUSTERED", "ST_100", "ST_070", "ST_050", "ST_030"]

# Pocket vs chain mode
MODE = "pocket"   # "pocket" or "chain"

# Make two separate files
PLOT_CLASSES = ["holo", "apo"]
SPLIT_BY_CLASS = False   # True: one figure per class, False: combined apo+holo

# Filtering
INCLUDE_TIES = False        # include pocket_class=="tie"
INCLUDE_UNDEFINED = False   # include pocket_class=="undefined" (rare; usually drop)
REQUIRE_VALID_CLASS_INPUTS = True   # require p_apo/p_holo + density not NaN

# Plot labels
SHOW_NODE_PCT_LABELS = False
NODE_PCT_LABEL_MODE = "class"  # "class" = % of apo/holo lane, "all" = % of all plotted
LABEL_ONLY_INTERNAL = True     # if True, hide Start/End node labels
LABEL_DECIMALS = 0             # 0 or 1

# Optional: drop tiny streams after aggregation
MIN_COUNT = 5  # set to 1 to keep everything

# Stage switches (set False to simplify the Sankey)
INCLUDE_STAGE_DENSITY     = True
INCLUDE_STAGE_SEPARATION  = True
INCLUDE_STAGE_PROBABILITY = True

# Missing overlap handling:
#  - 0    -> treat missing overlap as 0.0 (=> High separation)
#  - 1    -> treat missing overlap as 1.0 (=> Low separation)
#  None -> keep as NaN and create "No data" bucket (recommended if you want transparency)
TREAT_MISSING_OVERLAP_AS = None
DROP_NA_METRICS = True   # if True, remove rows with any "No data" bin in enabled stages

# Binning rules (legacy-compatible)
# Density / separation bins share thresholds
#DENSITY_BINS = dict(low=0.30, moderate=0.70)   # <=low => Low, <=moderate => Moderate, else High
DENSITY_BINS = dict(low=0.30, moderate=0.60)

# Probability bins (legacy)
#PROB_BINS = dict(low=0.60, moderate=0.70)      # <=low => Low, <=moderate => Moderate, else High
PROB_BINS = dict(low=0.5933, moderate=0.6996)


# Aesthetics (align broadly with Fig2)
FONT_FAMILY = "Helvetica Neue, Arial Nova, Arial, Liberation Sans, DejaVu Sans"
FONT_SIZE   = 13
NODE_LABEL_FONT_SIZE = 16
NODE_PAD    = 28 # Smaller = tighter gap between nodes (default=28)
NODE_THICK  = 38
FIG_WIDTH   = 2400
FIG_HEIGHT  = 1200
MARGIN_PX   = 20


# Color scheme:
#   "class"   -> apo blue / holo orange (streams inherit final class hue)
#   "neutral" -> same base hue for both apo/holo (for side-by-side non-biased comparison)
COLOR_MODE = "hybrid"  # "class", "neutral", or "hybrid"
FLAT_LINK_COLOR = True  # True = one single shade for all flows (no impact shading)
FLAT_LINK_LIGHTEN = 0.25  # 0..1 (higher = paler)

APO_COLOR  = figspec.APO   # single source: common/figspec.py
HOLO_COLOR = figspec.HOLO
#APO_COLOR = "#4F6D8A" # (slate blue)
#HOLO_COLOR = "#B06A4A" #(terracotta)
#APO_COLOR = "#4E7A7A" # blue-green (very neutral)
#HOLO_COLOR = "#C07A5A" # warm clay (most “pastel”)

MUTE_CLASS_NODES = True
NODE_MUTE_FACTOR_START = 0.25   # 0=no change, 1=white
NODE_MUTE_FACTOR_END   = 0.15   # end nodes slightly less muted


# Lighten flows
LINK_ALPHA_SCALE = 0.70 #0.70   # 0.0–1.0 ; smaller = more muted

#NEUTRAL_BASE = "#6b6b6b"
#NEUTRAL_BASE = "#E6D7C3"  # warm neutral/cream (stands out from gray nodes)
#NEUTRAL_BASE = "#555555"  # Neutral dark gray with strong alpha + lightening (most “unbiased”)
#NEUTRAL_BASE = "#4F6D7A"  # blue-gray / steel teal (stronger than cream)
#NEUTRAL_BASE = "#8A6F5A"  # warm taupe (much stronger than cream)
#NEUTRAL_BASE = "#4E6E5D"   # (sage / eucalyptus green-gray)
NEUTRAL_BASE = "#6B7D4B"   # (muted olive; slightly earthier)
#NEUTRAL_BASE = "#2F5E4E"   # (deep forest green, still muted)

# Stream color progression: "darker_with_impact" (legacy-like)
#MAX_LIGHTEN = 0.55
#MIN_LIGHTEN = 0.06
#GAMMA       = 0.40
#START_LIGHTEN_BOOST = 0.12

#MAX_LIGHTEN=0.75
#MIN_LIGHTEN=0.05
#GAMMA=0.85
# Stream color progression: higher contrast
#MAX_LIGHTEN = 0.70          # paler starts
#MIN_LIGHTEN = 0.00          # allow fully saturated base at max impact
#GAMMA       = 0.85          # stronger separation between Low/Moderate/High
#START_LIGHTEN_BOOST = 0.18  # extra pale "Start" links

#ALPHA_BY_STAGE = {0: 0.90, 1: 0.92, 2: 0.96, 3: 1.00}
#ALPHA_BY_STAGE = {0: 0.70, 1: 0.82, 2: 0.92, 3: 1.00}


# warm taupe 1
#MAX_LIGHTEN = 0.88
#MIN_LIGHTEN = 0.08
#GAMMA = 0.95
#TART_LIGHTEN_BOOST = 0.00  # was ~0.22 → don’t bleach the first link
#ALPHA_BY_STAGE = {0: 0.60, 1: 0.78, 2: 0.90, 3: 1.00}

# 2
MAX_LIGHTEN = 0.80          # was ~0.88 → less whitening overall
MIN_LIGHTEN = 0.10          # keep highs strong
GAMMA = 0.95
START_LIGHTEN_BOOST = 0.05  # was ~0.22 → don’t bleach the first link
ALPHA_BY_STAGE = {0: 0.80, 1: 0.86, 2: 0.93, 3: 1.00}  # stage 0 more visible



# Internal node fill (grayscale bins)
INTERNAL_NODE_COLORS = {
    "High":     "#7F7F7F",
    "Moderate": "#A6A6A6",
    "Low":      "#E2E2E2",
    "No data":  "#D9D9D9",
}

# Ordering controls (stable, interpretable)
CLASSES = ["holo", "apo"]   # which class row is drawn "above" in dual-start labeling
CLASS_ON_TOP = "apo"        # within a band, which class is painted last (on top)

# ============================================================
# Helpers
# ============================================================
def load_unified(path: str) -> pd.DataFrame:
    """Accept .csv or .zip (containing analysis_unified_levels.csv)."""
    p = str(path)
    if p.lower().endswith(".zip"):
        with zipfile.ZipFile(p) as z:
            name = "analysis_unified_levels.csv"
            if name not in z.namelist():
                csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
                if not csvs:
                    raise SystemExit("Zip contains no CSV.")
                name = csvs[0]
            with z.open(name) as f:
                df = pd.read_csv(f, low_memory=False)
    else:
        df = pd.read_csv(p, low_memory=False)
    return df

def require_cols(df: pd.DataFrame, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print("Missing required columns:", missing)
        print("Available columns:", list(df.columns))
        raise SystemExit("Stopping due to missing columns.")

def _col(mode: str, base: str) -> str:
    # unified schema uses "pocket_*" and "chain_*" prefixes
    if mode not in ("pocket", "chain"):
        raise ValueError("MODE must be 'pocket' or 'chain'")
    return f"{mode}_{base}"

def pick_first_existing(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def subset_threshold(df: pd.DataFrame, thr: str) -> pd.DataFrame:
    lvl = "entry" if thr == "UNCLUSTERED" else "ubs"
    sub = df[(df["threshold"].astype(str) == str(thr)) &
             (df["level"].astype(str).str.lower() == lvl)].copy()
    return sub

def classify_3bin(x, *, low, moderate):
    """Return High/Moderate/Low or No data."""
    try:
        x = float(x)
    except Exception:
        return "No data"
    if not np.isfinite(x):
        return "No data"
    if x <= low:
        return "Low"
    if x <= moderate:
        return "Moderate"
    return "High"

def blend_towards_white(rgb, amount):
    return tuple((1 - amount) * c + amount for c in rgb)

def hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r = int(h[0:2], 16); g = int(h[2:4], 16); b = int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

def cumulative_color(base_rgb, cum_weight, max_total):
    # impact increases -> amount decreases -> darker
    t = 0.0 if max_total == 0 else (cum_weight / max_total) ** GAMMA
    amount = MAX_LIGHTEN - (MAX_LIGHTEN - MIN_LIGHTEN) * t
    return to_hex(blend_towards_white(base_rgb, amount))

def cumulative_color_start(base_rgb):
    amount = min(1.0, MAX_LIGHTEN + START_LIGHTEN_BOOST)
    return to_hex(blend_towards_white(base_rgb, amount))

def stable_threshold_label(thr: str) -> str:
    if thr == "UNCLUSTERED":
        return "UNCLUSTERED (entry)"
    if thr.startswith("ST_") and thr[3:].isdigit():
        return f"{thr} (ubs)"
    return f"{thr} (ubs)"

def lighten_hex(hex_color: str, t: float) -> str:
    """Mix hex_color toward white by fraction t (0..1). Returns hex."""
    t = max(0.0, min(1.0, float(t)))
    r, g, b = to_rgb(hex_color)          # floats 0..1
    r = int(round(r * 255))
    g = int(round(g * 255))
    b = int(round(b * 255))
    r2 = int(round(r + (255 - r) * t))
    g2 = int(round(g + (255 - g) * t))
    b2 = int(round(b + (255 - b) * t))
    return f"#{r2:02X}{g2:02X}{b2:02X}"

# ============================================================
# Core Sankey builder
# ============================================================
def build_sankey_for_threshold(df_all: pd.DataFrame, thr: str, class_filter: str | None = None):
#def build_sankey_for_threshold(df_all: pd.DataFrame, thr: str):
    # --- Select rows for this threshold
    df = subset_threshold(df_all, thr)
    if df.empty:
        print(f"[warn] No rows for threshold={thr}")
        return None

    # --- Column mapping (explicit, with fallbacks)
    col_class = "pocket_class" if MODE == "pocket" else "chain_class"
    col_p_apo  = _col(MODE, "p_apo")
    col_p_holo = _col(MODE, "p_holo")

    col_density = pick_first_existing(df, [_col(MODE, "density_pair"),
                                           _col(MODE, "density_combined"),
                                          _col(MODE, "density_strongest")])

    col_overlap = pick_first_existing(df, [_col(MODE, "overlap_pair"),
                                          _col(MODE, "overlap_overall"),
                                          _col(MODE, "overlap_mode")])

    # Required essentials
    needed = ["threshold", "level", col_class, col_p_apo, col_p_holo]
    if REQUIRE_VALID_CLASS_INPUTS:
        if col_density:
            needed.append(col_density)
        if col_overlap:
            needed.append(col_overlap)
    require_cols(df, needed)

    # --- Normalize class labels
    df[col_class] = df[col_class].astype(str).str.strip().str.lower()

    keep = ["apo", "holo"]
    if INCLUDE_TIES:
        keep.append("tie")
    if INCLUDE_UNDEFINED:
        keep.append("undefined")
    df = df[df[col_class].isin(keep)].copy()

    # --- Numeric coercion + input validity
    df[col_p_apo]  = pd.to_numeric(df[col_p_apo], errors="coerce")
    df[col_p_holo] = pd.to_numeric(df[col_p_holo], errors="coerce")
    if col_density:
        df[col_density] = pd.to_numeric(df[col_density], errors="coerce")
    if col_overlap:
        df[col_overlap] = pd.to_numeric(df[col_overlap], errors="coerce")

    if REQUIRE_VALID_CLASS_INPUTS:
        need_non_na = [col_p_apo, col_p_holo]
        if INCLUDE_STAGE_DENSITY and col_density:
            need_non_na.append(col_density)
        df = df.dropna(subset=need_non_na).copy()

    # --- Derive stage metrics
    df["final_class"] = df[col_class].astype(str)
    
    if class_filter is not None:
        df = df[df["final_class"] == class_filter].copy()


    df["probability"] = df[[col_p_apo, col_p_holo]].max(axis=1)

    if INCLUDE_STAGE_DENSITY and col_density:
        df["density_metric"] = df[col_density]
    if INCLUDE_STAGE_SEPARATION and col_overlap:
        if TREAT_MISSING_OVERLAP_AS in (0, 1):
            overlap_src = df[col_overlap].fillna(float(TREAT_MISSING_OVERLAP_AS))
        else:
            overlap_src = df[col_overlap]
        df["separation_metric"] = 1.0 - overlap_src

    # --- Bin stages (legacy thresholds)
    BINS3 = ["High", "Moderate", "Low", "No data"]
    BIN_WEIGHT = {"Low": 0, "Moderate": 1, "High": 2, "No data": 0}
    enabled_bins = []

    if INCLUDE_STAGE_DENSITY and col_density:
        df["density_bin"] = df["density_metric"].apply(lambda x: classify_3bin(x, **DENSITY_BINS))
        enabled_bins.append("density_bin")
    if INCLUDE_STAGE_SEPARATION and col_overlap:
        df["separation_bin"] = df["separation_metric"].apply(lambda x: classify_3bin(x, **DENSITY_BINS))
        enabled_bins.append("separation_bin")
    if INCLUDE_STAGE_PROBABILITY:
        df["probability_bin"] = df["probability"].apply(lambda x: classify_3bin(x, **PROB_BINS))
        enabled_bins.append("probability_bin")

    if DROP_NA_METRICS and enabled_bins:
        for b in enabled_bins:
            df = df[df[b] != "No data"].copy()

    # Keep only apo/holo for final class coloring (unless you explicitly include others)
    df = df[df["final_class"].isin(["apo", "holo"])].copy()

    if df.empty:
        print(f"[warn] Empty after filters/bins for threshold={thr}")
        return None

    # --- Cohort totals for hover percentages (post-filter = plotted sites)
    plotted_total_all = int(len(df))
    plotted_total_by_class = df["final_class"].value_counts().to_dict()

    # --- Node definition (stage order)
    stages = []
    if INCLUDE_STAGE_DENSITY and col_density:
        stages.append(("Density", "density_bin"))
    if INCLUDE_STAGE_SEPARATION and col_overlap:
        stages.append(("Separation", "separation_bin"))
    if INCLUDE_STAGE_PROBABILITY:
        stages.append(("Probability", "probability_bin"))

    # --- Colors: base by class or neutral
    # --- Colors:
    #   class  -> links + class nodes use apo/holo colors
    #   neutral-> links + class nodes use one neutral color
    #   hybrid -> links use neutral; Start/End nodes keep apo/holo colors
    if COLOR_MODE == "neutral":
        class_base_colors = {"apo": to_rgb(NEUTRAL_BASE), "holo": to_rgb(NEUTRAL_BASE)}
        class_node_colors = {"apo": NEUTRAL_BASE, "holo": NEUTRAL_BASE}
        if MUTE_CLASS_NODES:
            class_node_colors = {
                "apo":  lighten_hex(class_node_colors["apo"],  NODE_MUTE_FACTOR_START),
                "holo": lighten_hex(class_node_colors["holo"], NODE_MUTE_FACTOR_START),}
    
    elif COLOR_MODE == "hybrid":
        class_base_colors = {"apo": to_rgb(NEUTRAL_BASE), "holo": to_rgb(NEUTRAL_BASE)}  # links
        class_node_colors = {"apo": APO_COLOR, "holo": HOLO_COLOR}  # Start/End nodes
        if MUTE_CLASS_NODES:
            class_node_colors = {
                "apo":  lighten_hex(class_node_colors["apo"],  NODE_MUTE_FACTOR_START),
                "holo": lighten_hex(class_node_colors["holo"], NODE_MUTE_FACTOR_START),}    
    else:  # "class"
        class_base_colors = {"apo": to_rgb(APO_COLOR), "holo": to_rgb(HOLO_COLOR)}
        class_node_colors = {"apo": APO_COLOR, "holo": HOLO_COLOR}
        if MUTE_CLASS_NODES:
            class_node_colors = {
                "apo":  lighten_hex(class_node_colors["apo"],  NODE_MUTE_FACTOR_START),
                "holo": lighten_hex(class_node_colors["holo"], NODE_MUTE_FACTOR_START),}
    # --- Nodes
    base_labels, display_labels, node_colors = [], [], []

    # Start nodes
    for cls in CLASSES:
        base_labels.append(f"Start: {cls}")
        display_labels.append(f"Start: {cls}")
        node_colors.append(class_node_colors.get(cls, "#999999"))

    # Internal nodes: optionally separated by class to produce two "lanes"
    SEPARATE_INTERNAL_BY_CLASS = True  # keeps the legacy lane separation (readability)
    if SEPARATE_INTERNAL_BY_CLASS:
        for cls in CLASSES:
            for stage_name, _bin_col in stages:
                for bin_name in BINS3:
                    base_labels.append(f"{stage_name}: {bin_name}|{cls}")
                    display_labels.append(f"{stage_name}: {bin_name}")
                    node_colors.append(INTERNAL_NODE_COLORS[bin_name])
    else:
        for stage_name, _bin_col in stages:
            for bin_name in BINS3:
                base_labels.append(f"{stage_name}: {bin_name}")
                display_labels.append(f"{stage_name}: {bin_name}")
                node_colors.append(INTERNAL_NODE_COLORS[bin_name])

    # End nodes
    for cls in CLASSES:
        base_labels.append(cls)
        display_labels.append(cls)
        node_colors.append(class_node_colors.get(cls, "#999999"))

    label_to_index = {lab: i for i, lab in enumerate(base_labels)}

    # --- Aggregate links
    raw_links = []

    # --- Node membership counts (count sites, not link volume)
    node_site_counts = Counter()            # base_label -> number of plotted sites visiting node

    def node_key(stage_name, bin_name, cls):
        if SEPARATE_INTERNAL_BY_CLASS:
            return f"{stage_name}: {bin_name}|{cls}"
        return f"{stage_name}: {bin_name}"

    max_total = 0
    # precompute maximum possible cumulative weight for enabled stages
    # (High=2) * number_of_enabled_stages
    max_total = 2 * max(1, len(stages))

    for _, row in df.iterrows():
        cls = row["final_class"]

        # stage bins in order
        bins = []
        for stage_name, bcol in stages:
            bins.append((stage_name, str(row[bcol])))

        # count node membership for this site (each node once per site)
        if not bins:
            # no stages: Start -> class
            node_site_counts[f"Start: {cls}"] += 1
            node_site_counts[cls] += 1
        else:
            node_site_counts[f"Start: {cls}"] += 1
            for stage_name, bin_name in bins:
                node_site_counts[node_key(stage_name, bin_name, cls)] += 1
            node_site_counts[cls] += 1

        # Start -> first stage (or start -> end if no stages)
        if not bins:
            src = f"Start: {cls}"
            tgt = cls

            #col = hex_to_rgba(cumulative_color_start(class_base_colors[cls]), ALPHA_BY_STAGE[0] * LINK_ALPHA_SCALE)
            if FLAT_LINK_COLOR:
                flat_hex = to_hex(blend_towards_white(class_base_colors[cls], FLAT_LINK_LIGHTEN))
                col = hex_to_rgba(flat_hex, ALPHA_BY_STAGE[0] * LINK_ALPHA_SCALE)
            else:
                col = hex_to_rgba(cumulative_color_start(class_base_colors[cls]), ALPHA_BY_STAGE[0] * LINK_ALPHA_SCALE)

            raw_links.append((src, tgt, col, f"Start ({cls}) → {cls}", cls))
            continue

        # Start link
        src0 = f"Start: {cls}"
        tgt0 = node_key(bins[0][0], bins[0][1], cls)
        #col0 = hex_to_rgba(cumulative_color_start(class_base_colors[cls]), ALPHA_BY_STAGE[0] * LINK_ALPHA_SCALE)
        if FLAT_LINK_COLOR:
            flat_hex = to_hex(blend_towards_white(class_base_colors[cls], FLAT_LINK_LIGHTEN))
            col0 = hex_to_rgba(flat_hex, ALPHA_BY_STAGE[0] * LINK_ALPHA_SCALE)
        else:
            col0 = hex_to_rgba(cumulative_color_start(class_base_colors[cls]), ALPHA_BY_STAGE[0] * LINK_ALPHA_SCALE)

        raw_links.append((src0, tgt0, col0, f"Start ({cls}) → {bins[0][0]}: {bins[0][1]}", cls))

        # Stage-to-stage links
        cum = 0
        for i in range(len(bins) - 1):
            stage_i, bin_i = bins[i]
            stage_j, bin_j = bins[i + 1]
            cum += BIN_WEIGHT.get(bin_i, 0)
            src = node_key(stage_i, bin_i, cls)
            tgt = node_key(stage_j, bin_j, cls)
            #col = hex_to_rgba(cumulative_color(class_base_colors[cls], cum, max_total), ALPHA_BY_STAGE[min(i+1, 3)] * LINK_ALPHA_SCALE)
            if FLAT_LINK_COLOR:
                flat_hex = to_hex(blend_towards_white(class_base_colors[cls], FLAT_LINK_LIGHTEN))
                col = hex_to_rgba(flat_hex, ALPHA_BY_STAGE[min(i+1, 3)] * LINK_ALPHA_SCALE)
            else:
                col = hex_to_rgba(cumulative_color(class_base_colors[cls], cum, max_total),
                                  ALPHA_BY_STAGE[min(i+1, 3)] * LINK_ALPHA_SCALE)

            raw_links.append((src, tgt, col, f"{stage_i}:{bin_i} → {stage_j}:{bin_j}", cls))

        # Last stage -> end
        cum += BIN_WEIGHT.get(bins[-1][1], 0)
        srcL = node_key(bins[-1][0], bins[-1][1], cls)
        tgtL = cls
        #colL = hex_to_rgba(cumulative_color(class_base_colors[cls], cum, max_total), ALPHA_BY_STAGE[3] * LINK_ALPHA_SCALE)
        if FLAT_LINK_COLOR:
            flat_hex = to_hex(blend_towards_white(class_base_colors[cls], FLAT_LINK_LIGHTEN))
            colL = hex_to_rgba(flat_hex, ALPHA_BY_STAGE[3] * LINK_ALPHA_SCALE)
        else:
            colL = hex_to_rgba(cumulative_color(class_base_colors[cls], cum, max_total), ALPHA_BY_STAGE[3] * LINK_ALPHA_SCALE)

        raw_links.append((srcL, tgtL, colL, f"{bins[-1][0]}:{bins[-1][1]} → {cls}", cls))

    # Aggregate identical links
    agg = defaultdict(int)
    for src, tgt, col, hover, cls in raw_links:
        agg[(src, tgt, col, hover, cls)] += 1

    # Drop tiny streams
    agg2 = {k: v for k, v in agg.items() if v >= int(MIN_COUNT)}
    if not agg2:
        print(f"[warn] All links removed by MIN_COUNT for threshold={thr} (MIN_COUNT={MIN_COUNT}).")
        return None

    # Sorting / z-order (stable): by (src_idx, tgt_idx) then class on top
    def class_z(cls: str) -> int:
        return 1 if cls == CLASS_ON_TOP else 0

    compiled = []
    for (src, tgt, col, hover, cls), count in agg2.items():
        compiled.append(dict(
            src_label=src,
            tgt_label=tgt,
            src_idx=label_to_index[src],
            tgt_idx=label_to_index[tgt],
            color=col,
            hover=hover,
            cls=cls,
            count=count,
            cls_z=class_z(cls),
        ))

    compiled.sort(key=lambda d: (d["src_idx"], d["tgt_idx"], d["cls_z"], d["src_label"], d["tgt_label"]))

    # Totals per final class (for % annotations like legacy)
    class_totals = Counter()
    for d in compiled:
        class_totals[d["cls"]] += d["count"]
    total_all = sum(class_totals.values())

    sources = [d["src_idx"] for d in compiled]
    targets = [d["tgt_idx"] for d in compiled]
    values  = [d["count"] for d in compiled]
    colors  = [d["color"] for d in compiled]

    hover_texts = []
    for d in compiled:
        denom = plotted_total_by_class.get(d["cls"], 0)
        #denom = class_totals.get(d["cls"], 0)
        pct = (100.0 * d["count"] / denom) if denom else 0.0
        hover_texts.append(f"{d['hover']}: n={d['count']:,}  ({pct:.1f}% of {d['cls']})")

    def extract_cls_from_base(base: str):
        # base labels look like:
        # "Start: apo"
        # "Density: High|apo"
        # "Separation: Moderate|holo"
        # "Probability: Low|apo"
        # "apo" / "holo"
        if base in CLASSES:
            return base
        if base.startswith("Start:"):
            return base.split("Start:")[-1].strip()
        if "|" in base:
            return base.split("|")[-1].strip()
        return None
    
    node_hover = []
    for base, disp in zip(base_labels, display_labels):
        n = node_site_counts.get(base, 0)
        cls = extract_cls_from_base(base)

        total_all = plotted_total_all
        class_totals = plotted_total_by_class
        
        if cls in class_totals and class_totals[cls] > 0:
            pct_class = 100.0 * n / class_totals[cls]
            pct_all = 100.0 * n / total_all if total_all else 0.0
            node_hover.append(
                f"{disp}<br>n={n:,} / {class_totals[cls]:,}  ({pct_class:.1f}% of {cls})"
                f"<br>{pct_all:.1f}% of All"
            )
        else:
            pct_all = 100.0 * n / total_all if total_all else 0.0
            node_hover.append(f"{disp}<br>n={n:,}<br>{pct_all:.1f}% of All")

    node_labels = []
    for base, disp in zip(base_labels, display_labels):
        # default: blank
        txt = ""

        is_start = base.startswith("Start:")
        is_end = base in CLASSES
        is_internal = (not is_start) and (not is_end)

        if SHOW_NODE_PCT_LABELS and (is_internal or not LABEL_ONLY_INTERNAL):
            cls = extract_cls_from_base(base)

            if NODE_PCT_LABEL_MODE == "all":
                denom = plotted_total_all
            else:  # "class"
                denom = plotted_total_by_class.get(cls, 0)

            n = node_site_counts.get(base, 0)
            if denom:
                pct = 100.0 * n / denom
                fmt = f"{{:.{int(LABEL_DECIMALS)}f}}%"
                txt = fmt.format(pct)

        node_labels.append(txt)

    #title = f"{OUT_BASENAME} — {stable_threshold_label(thr)} — {MODE}"

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=NODE_PAD,
            thickness=NODE_THICK,
            line=dict(color="rgba(0,0,0,0)", width=0),
            label=node_labels,
            #label=[""] * len(display_labels),  # hide labels (hover only) for clean NAR look
            color=node_colors,
            customdata=node_hover,
            hovertemplate="%{customdata}<extra></extra>",
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=colors,
            customdata=hover_texts,
            hovertemplate="%{customdata}<extra></extra>",
            #line=dict(color="rgba(0,0,0,0.12)", width=0.4) # link outline

        ),
        textfont=dict(color="#222222", size=NODE_LABEL_FONT_SIZE , family=FONT_FAMILY),
    )])

    if TRANSPARENT_BG:
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    else:
        fig.update_layout(paper_bgcolor="white", plot_bgcolor="white")

    fig.update_layout(
        #title=dict(text=title, x=0.5, xanchor="center"),
        margin=dict(l=MARGIN_PX, r=MARGIN_PX, t=MARGIN_PX, b=MARGIN_PX),
        width=FIG_WIDTH,
        height=FIG_HEIGHT,
        font=dict(family=FONT_FAMILY, size=FONT_SIZE, color="#222222"),
    )
    return fig

# ============================================================
# Main driver
# ============================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_unified(INPUT_TABLE)
    require_cols(df, ["threshold", "level"])

    # Quick diagnostics: show columns relevant to the mapping
    print("[info] Loaded unified table:", df.shape)
    print("[info] MODE =", MODE)
    print("[info] Available columns:", len(df.columns))
    
    for thr in THRESHOLDS_TO_PLOT:

        if SPLIT_BY_CLASS:
            class_list = PLOT_CLASSES  # e.g. ["holo", "apo"]
        else:
            class_list = [None]        # single combined plot

        for cls in class_list:
            fig = build_sankey_for_threshold(df, str(thr), class_filter=cls)
            if fig is None:
                continue

            safe_thr = str(thr).replace(":", "_")
            cls_tag = f"_cls-{cls}" if cls is not None else ""
            out_base = os.path.join(
                OUTPUT_DIR,
                f"{OUT_BASENAME}_{MODE}_{safe_thr}_cmode-{COLOR_MODE}{cls_tag}"
            )

            out_height = FIG_HEIGHT // 2 if (SPLIT_BY_CLASS and cls is not None) else FIG_HEIGHT

            if WRITE_PDF:
                fig.write_image(out_base + ".pdf", width=FIG_WIDTH, height=out_height)
                print("[OK] Saved:", out_base + ".pdf")
            if WRITE_PNG:
                fig.write_image(out_base + ".png", width=FIG_WIDTH, height=out_height, scale=2)
                print("[OK] Saved:", out_base + ".png")
            if WRITE_SVG:
                fig.write_image(out_base + ".svg", width=FIG_WIDTH, height=out_height)
                print("[OK] Saved:", out_base + ".svg")
            if WRITE_HTML:
                fig.write_html(out_base + ".html")
                print("[OK] Saved:", out_base + ".html")


if __name__ == "__main__":
    main()
