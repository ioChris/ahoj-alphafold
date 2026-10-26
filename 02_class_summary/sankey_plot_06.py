""" Version 06: Multi-threshold & multi-mode driver (pocket/chain), LJ normalization, per-output naming """
""" Multi-threshold/multi-mode Sankey driver (adapted from an earlier Ions-project sankey script). """


import os
import pandas as pd
import plotly.graph_objects as go
from collections import Counter, defaultdict
from matplotlib.colors import to_rgb, to_hex

# =========================
# User-tunable parameters
# =========================
SEPARATE_SANKEY_BY_CLASS = True   # keep your previous toggle

# Which modes to run: any subset of ["pocket", "chain"]
RUN_MODES = ["pocket"]#, "chain"]   # e.g. ["pocket"], ["chain"], or ["pocket","chain"]

# Threshold selection:
#   - "auto": discover from the CSV; all numeric like "ST_050","ST_070", etc., plus one logical "ST_LJ" bucket for any "ST_LJ*".
#   - or provide an explicit list, e.g. ["ST_050", "ST_LJ"]
#THRESHOLDS_TO_RUN = "auto"
THRESHOLDS_TO_RUN = ["ST_050"]

# Output image toggles (keep your HTML export on)
WRITE_HTML = False
WRITE_SVG  = True
WRITE_PDF  = False
WRITE_PNG  = False

# Optional transparent background for exports
TRANSPARENT_BG = False

# Pick colore palette (out of 1-5)
COLOR_PALETTE = 2
ALPHA_PALETTE = 1

# Treat NaN overlap as zero (so separation=1.0). If False, keep "No data".
TREAT_MISSING_OVERLAP_AS = 0    # 0 (=High Separation) | 1 (=Low Separation) | None (="No Data" category), old default was 0
DROP_NA_METRICS = True          # Remove metrics with "No Data" before plotting (old = always on)


# -------- Candidate-count filtering (set any to None to disable) --------
FILTER_COUNTS = dict(
    min_sum_n_apo  = None,  # e.g., 1 or None
    min_sum_n_holo = None, # e.g., 1
    min_avg_n_apo  = 1,     # e.g., 2
    min_avg_n_holo = None,    # e.g., 2
)


# =========================

def get_2level_cwd():
    npath = os.path.normpath(os.getcwd())   # Normalize the path string for the OS
    path0 = os.path.join(npath.split(os.sep)[0], os.sep, npath.split(os.sep)[1], os.path.sep.join(npath.split(os.sep)[2:3]))
    # the above keeps the behavior of your original function (first three path segments)
    if os.path.exists(path0):
        print("Root path found >> " + path0)
    else:
        print("Error finding root path in working dir:" + npath)
    return path0


# === Load data ===
# --- locate repo root so `common` is importable (works standalone / in Spyder) ---
import sys as _sys
from pathlib import Path as _Path
for _root in _Path(__file__).resolve().parents:
    if (_root / "common" / "config.py").is_file():
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))
        break
from common import config as cfg

# Path to input CSV (pick the aggregation run via parsed_ubs_dir)
parsed_ubs_dir = '21012026_151032_v1.7_Noneunplimit'  # current
#parsed_ubs_dir = '07102025_120602_v1.6_Noneunplimit'
#parsed_ubs_dir = '05092025_114851_v1.4_Noneunplimit'

ubs_aggr_class_dir = os.path.join(str(cfg.UBS_AGG_DIR), parsed_ubs_dir)
csv_path = os.path.join(ubs_aggr_class_dir, 'ubs_allthresholds_aggregated_classifications.csv')

# Output dir
output_dir = os.path.join(ubs_aggr_class_dir, "sankey_results")
os.makedirs(output_dir, exist_ok=True)

# Load CSV
df_init = pd.read_csv(csv_path)

# ---------------------------------------------------------------------
# Helpers to normalize thresholds and discover which to run
# ---------------------------------------------------------------------
def normalize_threshold_label(th: str) -> str:
    """Map any ST_LJ* string to 'ST_LJ', otherwise return the threshold as-is."""
    if isinstance(th, str) and th.startswith("ST_LJ"):
        return "ST_LJ"
    return th

def discover_thresholds(dataframe: pd.DataFrame):
    """Return a list of logical thresholds to run, sorting numeric ST_XX ascending, with ST_LJ last (if present)."""
    uniq = dataframe["threshold"].dropna().astype(str).unique().tolist()
    logical = [normalize_threshold_label(x) for x in uniq]
    logical = sorted(set(logical))  # de-dup

    numeric = []
    has_lj = False
    for t in logical:
        if t == "ST_LJ":
            has_lj = True
        elif t.startswith("ST_"):
            # try to parse number after ST_
            rest = t.replace("ST_", "")
            try:
                # tolerate "05" or "0.5" etc. (your data uses "ST_050")
                val = float(rest.replace("_", "").replace("-", "").replace("LJ", ""))
            except:
                # if parsing fails, push to the end (rare)
                val = float("inf")
            numeric.append((val, t))
        else:
            # unknown naming → shove to the end but keep
            numeric.append((float("inf"), t))
    numeric_sorted = [t for _, t in sorted(numeric, key=lambda z: z[0])]
    if has_lj and "ST_LJ" not in numeric_sorted:
        numeric_sorted.append("ST_LJ")
    return numeric_sorted

def slice_df_for_threshold(dataframe: pd.DataFrame, thr_key: str) -> pd.DataFrame:
    """Select rows for one logical threshold key."""
    if thr_key == "ST_LJ":
        return dataframe[dataframe["threshold"].astype(str).str.startswith("ST_LJ")].copy()
    return dataframe[dataframe["threshold"].astype(str) == thr_key].copy()

def filter_by_candidate_thresholds(
    df: pd.DataFrame,
    spec: dict,
    *,
    classification_col: str | None = None,
    mode: str = "global",   # "global" or "per_class"
    verbose: bool = True,
    nan_as_zero: bool = True,  # True: NaN→0 so it fails thresholds; False: NaN rows fail explicitly
) -> pd.DataFrame:
    """
    Filter rows by minimum candidate-count thresholds, with diagnostics.

    Parameters
    ----------
    df : DataFrame
        Input table with columns like sum_n_apo, sum_n_holo, avg_n_apo, avg_n_holo.
    spec : dict
        Thresholds; set any to None to disable:
            {
              "min_sum_n_apo": 1,
              "min_sum_n_holo": 1,
              "min_avg_n_apo": 2,
              "min_avg_n_holo": 2,
            }
    classification_col : str or None
        Column holding "apo"/"holo" labels (e.g., your `col_class` remapped).
    mode : {"global","per_class"}
        - "global": apply all enabled thresholds to all rows.
        - "per_class": apply apo thresholds only to apo rows and holo thresholds only to holo rows.
    verbose : bool
        Print class counts and apo/holo prevalence before/after.
    nan_as_zero : bool
        If True, treat NaN counts as 0. If False, NaN fails thresholds explicitly.

    Returns
    -------
    DataFrame
        Filtered copy of df.
    """
    # Map spec keys to the expected column names in df
    columns_map = {
        "min_sum_n_apo":  "sum_n_apo",
        "min_sum_n_holo": "sum_n_holo",
        "min_avg_n_apo":  "avg_n_apo",
        "min_avg_n_holo": "avg_n_holo",
    }

    # --- Utilities ---------------------------------------------------------
    def _class_counts(dfx):
        if not classification_col or classification_col not in dfx.columns:
            return {}, 0, 0, 0
        vc = dfx[classification_col].astype(str).value_counts(dropna=False)
        a = int(vc.get("apo", 0))
        h = int(vc.get("holo", 0))
        return vc.to_dict(), a, h, (a + h)

    def _prev(a, h, total):
        if total <= 0:
            return "apo=0.0%, holo=0.0%"
        return f"apo={(100.0*a/total):.1f}%, holo={(100.0*h/total):.1f}%"

    def _series_for(col):
        s = pd.to_numeric(df[col], errors="coerce")
        if nan_as_zero:
            s = s.fillna(0.0)
        return s

    # --- Report BEFORE -----------------------------------------------------
    if verbose:
        counts_pre, a_pre, h_pre, tot_pre = _class_counts(df)
        print(f"[counts-pre] n={len(df)} class_counts={counts_pre}  prevalence={_prev(a_pre, h_pre, tot_pre)}")

    # --- Build mask --------------------------------------------------------
    mask = pd.Series(True, index=df.index)
    missing = []

    if mode == "global":
        for key, col in columns_map.items():
            thr = spec.get(key, None)
            if thr is None:
                continue
            if col not in df.columns:
                missing.append(col); continue

            s = _series_for(col)
            if not nan_as_zero:
                mask &= s.notna()
            mask &= (s >= float(thr))

    elif mode == "per_class":
        # Start with True, then apply class-appropriate thresholds only within that class
        # If classification column is missing, fall back to global behavior
        if not classification_col or classification_col not in df.columns:
            for key, col in columns_map.items():
                thr = spec.get(key, None)
                if thr is None:
                    continue
                if col not in df.columns:
                    missing.append(col); continue
                s = _series_for(col)
                if not nan_as_zero:
                    mask &= s.notna()
                mask &= (s >= float(thr))
        else:
            is_apo  = df[classification_col].astype(str).eq("apo")
            is_holo = df[classification_col].astype(str).eq("holo")

            # Apo thresholds on apo rows
            for key, col in (("min_sum_n_apo","sum_n_apo"), ("min_avg_n_apo","avg_n_apo")):
                thr = spec.get(key, None)
                if thr is None:
                    continue
                if col not in df.columns:
                    missing.append(col); continue
                s = _series_for(col)
                if not nan_as_zero:
                    mask &= (~is_apo) | s.notna()
                mask &= (~is_apo) | (s >= float(thr))

            # Holo thresholds on holo rows
            for key, col in (("min_sum_n_holo","sum_n_holo"), ("min_avg_n_holo","avg_n_holo")):
                thr = spec.get(key, None)
                if thr is None:
                    continue
                if col not in df.columns:
                    missing.append(col); continue
                s = _series_for(col)
                if not nan_as_zero:
                    mask &= (~is_holo) | s.notna()
                mask &= (~is_holo) | (s >= float(thr))
    else:
        raise ValueError(f"mode must be 'global' or 'per_class', got: {mode}")

    if missing and verbose:
        print(f"[filter_by_candidate_thresholds] Missing columns skipped: {sorted(set(missing))}")

    out = df[mask].copy()

    # --- Report AFTER ------------------------------------------------------
    if verbose:
        counts_post, a_post, h_post, tot_post = _class_counts(out)
        print(f"[counts-post] n={len(out)} class_counts={counts_post}  prevalence={_prev(a_post, h_post, tot_post)}")

    return out


# ---------------------------------------------------------------------
# Sankey builder (essentially your existing plotting section, parameterized)
# ---------------------------------------------------------------------
def build_and_save_sankey(df: pd.DataFrame, mode: str, threshold_key: str):
    """
    mode: "pocket" or "chain"
    threshold_key: e.g., "ST_050" or logical "ST_LJ"
    """
    # Columns for the selected mode
    suffix = "_pocket" if mode == "pocket" else "_chain"
    col_p_apo   = f"avg_P(apo)_mean{suffix}"
    col_p_holo  = f"avg_P(holo)_mean{suffix}"
    col_density = f"avg_normalized_density_combined_mean{suffix}"
    col_overlap = f"pair_overlap_mean{suffix}"
    col_class   = f"highest_p_mean{suffix}"

    needed = [col_p_apo, col_p_holo, col_density, col_overlap, col_class, "sum_n_apo", "sum_n_holo"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(f"[SKIP] Mode '{mode}' for threshold '{threshold_key}': missing columns {missing}")
        return

    # --- Class label (from your aggregated CSV)
    df["classification"] = df[col_class].astype(str)

    # Only keep apo/holo rows
    df = df[df["classification"].isin(["apo", "holo"])].copy()

    # >>> ADD HERE: candidate thresholds filter <<<
    df = filter_by_candidate_thresholds(
        df,
        FILTER_COUNTS,
        classification_col="classification",
        verbose=True,)
    if df.empty:
        print(f"[SKIP] Mode '{mode}' / '{threshold_key}': no rows after candidate-count filters.")
        return

    # Low sample flag (unchanged logic)
    #df["low_sample_flag"] = (df["sum_n_apo"] < 2) | (df["sum_n_holo"] < 2)

    # === Keep only rows with the columns we need and valid classes ===
    df = df.dropna(subset=[col_p_apo, col_p_holo, col_density, col_class]).copy() # keep overlap NaN
    #df = df.dropna(subset=[col_p_apo, col_p_holo, col_density, col_overlap, col_class]).copy()

    df["probability"]    = df[[col_p_apo, col_p_holo]].max(axis=1)
    if TREAT_MISSING_OVERLAP_AS == 0:
        sep_src = df[col_overlap].fillna(0.0)   # NaN overlap → 0.0 (=> High separation)
    elif TREAT_MISSING_OVERLAP_AS == 1:
        sep_src = df[col_overlap].fillna(1.0)   # NaN overlap → 1.0 (=> Low separation)
    else:
        sep_src = df[col_overlap]               # NaN stays NaN → "No data" bucket
    df["separation"]     = 1.0 - sep_src # df[col_overlap]
    df["density"]        = df[col_density]
    df["classification"] = df[col_class].astype(str)
    df = df[df["classification"].isin(["apo", "holo"])].copy()
    if df.empty:
        print(f"[SKIP] Mode '{mode}' / '{threshold_key}': no rows after filtering.")
        return

    # === Binning rules ===
    def classify_density(x):
        try: x = float(x)
        except: return "No data"
        if pd.isna(x): return "No data"
        if x <= 0.3:      return "Low"
        elif x <= 0.7:    return "Moderate"
        else:             return "High"

    def classify_probability(p):
        try: p = float(p)
        except: return "No data"
        if pd.isna(p): return "No data"
        if p <= 0.60:     return "Low"
        elif p <= 0.70:   return "Moderate"
        else:             return "High"

    df["density_bin"]     = df["density"].apply(classify_density)
    df["separation_bin"]  = df["separation"].apply(classify_density)  # same thresholds as density
    df["probability_bin"] = df["probability"].apply(classify_probability)

    if DROP_NA_METRICS:
        df = df[
            (df["density_bin"] != "No data") &
            #(df["separation_bin"] != "No data") &
            (df["probability_bin"] != "No data")
        ].copy()

    if df.empty:
        print(f"[SKIP] Mode '{mode}' / '{threshold_key}': no rows after valid binning.")
        return

    class_node_palette = {
        1: {
        "apo": "#16305B",       # Dark blue (nat PDF) ✓
        "holo": "#731D17",      # Dark red (nat PDF) ✓
        },
        2: {
        "apo": "#365C85",       # muted navy blue ✓
        "holo": "#A34747",      # muted brick red ✓
        },
        3: {
        "apo": "#0072b2",       # Blue (nat)
        "holo": "#d55e00",      # Vermillion (nat, too bright)
        },
        4: {
        "apo":  "#4682B4",      # natpub1
        "holo": "#DB143C",      # natpub1 (too pink)
        },
        5: {
        "apo":  "#1b3666",      # deep blue
        "holo": "#9b1c1c",      # deep red (too vivid)
        },
        6: {
        "apo": "#4682b4",       # natpub1
        "holo": "#dc143c",
        },
        7: {
        "apo": "#0073B3",       # okabeito
        "holo": "#D9541A",      # okabeito (orange)
        }
    }
    internal_palette = {
        #"High":     "#8A8A8A",  # + ~5% darker
        #"Moderate": "#B0B0B0",
        #"Low":      "#E6E6E6"
        "High": "#7F7F7F",
        "Moderate": "#A6A6A6",
        "Low": "#E2E2E2",
        "No data":  "#D9D9D9",   # new
    }
    class_base_colors = {
        "apo":  to_rgb(class_node_palette[COLOR_PALETTE].get("apo")),
        "holo": to_rgb(class_node_palette[COLOR_PALETTE].get("holo")),
    }

    MAX_LIGHTEN = 0.55  # was 0.60
    MIN_LIGHTEN = 0.06  # was 0.05
    GAMMA       = 0.40  # was 0.65 (slightly more contrast progression)
    #GAMMA       = 0.70  # was 0.65 (slightly more contrast progression)
    START_LIGHTEN_BOOST = 0.12  # was 0.08
    alpha_by_stage_palette = {
        1: {0: 1.00, 1: 1.00, 2: 1.00, 3: 1.00},
        2: {0: 0.85, 1: 0.90, 2: 0.95, 3: 1.00},
        3: {0: 0.75, 1: 0.85, 2: 0.95, 3: 1.00}
        }
    #ALPHA_BY_STAGE = {0: 0.75, 1: 0.85, 2: 0.95, 3: 1.00}  # was all 1.0
    #ALPHA_BY_STAGE = {0: 1.00, 1: 1.00, 2: 1.00, 3: 1.00}

    ALPHA_BY_STAGE = alpha_by_stage_palette[ALPHA_PALETTE]
    BIN_WEIGHT = {"Low": 0, "Moderate": 1, "High": 2, "No data": 0}
    #BIN_WEIGHT = {"Low": 0, "Moderate": 1, "High": 2}
    MAX_TOTAL  = BIN_WEIGHT["High"] * 3  # 0..6

    def blend_towards_white(rgb, amount):
        return tuple((1 - amount) * c + amount for c in rgb)

    def hex_to_rgba(hex_color: str, alpha: float) -> str:
        hex_color = hex_color.lstrip("#")
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    def cumulative_color_start(classification):
        base = class_base_colors.get(classification, (0.5, 0.5, 0.5))
        amount = min(1.0, MAX_LIGHTEN + START_LIGHTEN_BOOST)
        return to_hex(blend_towards_white(base, amount))

    def cumulative_color(classification, cum_weight):
        base = class_base_colors.get(classification, (0.5, 0.5, 0.5))
        t = 0.0 if MAX_TOTAL == 0 else (cum_weight / MAX_TOTAL) ** GAMMA
        amount = MAX_LIGHTEN - (MAX_LIGHTEN - MIN_LIGHTEN) * t
        return to_hex(blend_towards_white(base, amount))

    INTERNAL_STAGES = ["Density", "Separation", "Probability"]
    CLASSES = ["holo", "apo"]            # holo stack above apo; flip if preferred
    BINS3 = ["High", "Moderate", "Low", "No data"]
    #BINS3   = ["High", "Moderate", "Low"]

    def internal_key(stage_name, bin_name, cls=None):
        return f"{stage_name}: {bin_name}"
        # (note: preserving your original function, the first return makes the second unreachable)

    def public_label(stage_name, bin_name):
        return f"{stage_name}: {bin_name}"

    # Node setup
    base_labels, display_labels, node_colors = [], [], []

    # Start nodes (stage 0)
    for cls in CLASSES:
        base_labels.append(f"Start: {cls}")
        display_labels.append(f"Start: {cls}")
        node_colors.append(class_node_palette[COLOR_PALETTE][cls])

    # Node counts
    node_counts = Counter()
    class_node_counts = defaultdict(lambda: defaultdict(int))  # node_label -> class -> count

    for _, row in df.iterrows():
        d = row["density_bin"]; s = row["separation_bin"]; p = row["probability_bin"]; c = row["classification"]
        node_counts[f"Start: {c}"] += 1
        node_counts[c] += 1

        if SEPARATE_SANKEY_BY_CLASS:
            def key(stage, bin): return f"{stage}: {bin}|{c}"
        else:
            def key(stage, bin): return f"{stage}: {bin}"

        dkey = key("Density", d); skey = key("Separation", s); pkey = key("Probability", p)
        for node in [dkey, skey, pkey]:
            node_counts[node] += 1
            class_node_counts[node][c] += 1

    # Internal nodes (split or merged)
    if SEPARATE_SANKEY_BY_CLASS:
        for cls in CLASSES:
            for stage_name in INTERNAL_STAGES:
                for bin_name in BINS3:
                    key = internal_key(stage_name, bin_name)
                    base_labels.append(f"{key}|{cls}")
                    display_labels.append(public_label(stage_name, bin_name))
                    node_colors.append(internal_palette[bin_name])
    else:
        for stage_name in INTERNAL_STAGES:
            for bin_name in BINS3:
                key = internal_key(stage_name, bin_name)
                base_labels.append(key)
                display_labels.append(public_label(stage_name, bin_name))
                node_colors.append(internal_palette[bin_name])

    # End nodes
    for cls in CLASSES:
        base_labels.append(cls)
        display_labels.append(cls)
        node_colors.append(class_node_palette[COLOR_PALETTE][cls])

    labels = base_labels
    label_to_index = {lab: i for i, lab in enumerate(labels)}
    class_totals = df["classification"].value_counts().to_dict()

    # Links
    raw_links = []
    for _, row in df.iterrows():
        d = row["density_bin"]; s = row["separation_bin"]; p = row["probability_bin"]; c = row["classification"]
        wd = BIN_WEIGHT[d]; ws = BIN_WEIGHT[s]; wp = BIN_WEIGHT[p]

        if SEPARATE_SANKEY_BY_CLASS:
            def key(stage, bin): return f"{stage}: {bin}|{c}"
        else:
            def key(stage, bin): return f"{stage}: {bin}"

        src0 = f"Start: {c}"; tgt0 = key("Density", d)
        col0 = hex_to_rgba(cumulative_color_start(c), ALPHA_BY_STAGE[0])
        txt0 = f"Start ({c}) → Density: {d}"
        raw_links.append((src0, tgt0, col0, txt0, c))

        src1 = key("Density", d); tgt1 = key("Separation", s)
        col1 = hex_to_rgba(cumulative_color(c, wd), ALPHA_BY_STAGE[1])
        txt1 = f"{d} → {s}"
        raw_links.append((src1, tgt1, col1, txt1, c))

        src2 = key("Separation", s); tgt2 = key("Probability", p)
        col2 = hex_to_rgba(cumulative_color(c, wd + ws), ALPHA_BY_STAGE[2])
        txt2 = f"{d} | {s} → {p}"
        raw_links.append((src2, tgt2, col2, txt2, c))

        src3 = key("Probability", p); tgt3 = c
        col3 = hex_to_rgba(cumulative_color(c, wd + ws + wp), ALPHA_BY_STAGE[3])
        txt3 = f"{d} | {s} | {p} → {c}"
        raw_links.append((src3, tgt3, col3, txt3, c))

    # ---------------------------
    # Stable, node-first z-order
    # ---------------------------
    def impact_score(txt):
        if "→" not in txt: 
            return 0
        return sum(BIN_WEIGHT.get(b.strip(), 0) for b in txt.split("→")[0].split("|"))
    
    def source_impact_score(source_label):
        try:
            core = source_label.split(":")[1].split("|")[0].strip()
            return BIN_WEIGHT.get(core, 0)
        except IndexError:
            return 0
    
    def target_impact_score(target_label):
        try:
            core = target_label.split(":")[1].split("|")[0].strip()
            return BIN_WEIGHT.get(core, 0)
        except IndexError:
            return 0
    
    # Aggregate unit links
    agg = defaultdict(int)
    for src, tgt, col, txt, cls in raw_links:
        agg[(src, tgt, col, txt, cls)] += 1
    
    # Choose which class should be on top *within the same (src,tgt) band*
    CLASSES_ORDER = ["holo", "apo"]
    CLASS_ON_TOP = "apo"  # flip to "holo" if you prefer
    def class_z(cls: str) -> int:
        # bottom drawn first (0), top drawn last (1)
        return 1 if cls == CLASS_ON_TOP else 0
    
    # Flatten to a sortable list with precomputed keys
    compiled = []
    for (src, tgt, col, txt, cls), count in agg.items():
        compiled.append(dict(
            src_label=src,
            tgt_label=tgt,
            src_idx=label_to_index[src],
            tgt_idx=label_to_index[tgt],
            color=col,
            hover=txt,
            cls=cls,
            count=count,
            s_imp=source_impact_score(src),
            t_imp=target_impact_score(tgt),
            path_imp=impact_score(txt),
            cls_z=class_z(cls),
        ))
    
    if SEPARATE_SANKEY_BY_CLASS:
        # keep your existing behavior for dual Sankeys
        compiled.sort(key=lambda d: (
            CLASSES.index(d["cls"]),   # class grouping (["holo","apo"])
            -d["s_imp"],               # source impact high→low
            -d["t_imp"],               # target impact high→low
            -d["path_imp"],            # cumulative impact high→low
            d["src_label"]             # tiebreaker
        ))
    else:
        # SINGLE Sankey: enforce node priority
        # 1) group by (src_idx, tgt_idx) so each band is contiguous,
        # 2) within the band draw bottom class first, top class last,
        # 3) impacts are gentle tiebreakers.
        compiled.sort(key=lambda d: (
            d["src_idx"],              # node priority: all links leaving a node together
            d["tgt_idx"],              # keep bands from that node contiguous
            d["cls_z"],                # 0 (bottom) → 1 (top) inside band
            -d["s_imp"], -d["t_imp"],  # readability
            -d["path_imp"]
        ))
    
    # Build arrays in the *sorted* order (later = painted on top)
    sources, targets, values, colors, hover_texts = [], [], [], [], []
    for d in compiled:
        sources.append(d["src_idx"])
        targets.append(d["tgt_idx"])
        values.append(d["count"])
        colors.append(d["color"])
        denom = class_totals.get(d["cls"], 0)
        pct = (100.0 * d["count"] / denom) if denom else 0.0
        hover_texts.append(f"{d['hover']}: n={d['count']}  ({pct:.1f}% of {d['cls']})")


    def extract_class_from_base(base):
        if base in CLASSES_ORDER:
            return base
        elif base.startswith("Start:"):
            return base.split("Start:")[-1].strip()
        elif "|" in base:
            return base.split("|")[-1].strip()
        else:
            return None

    total_all = sum(class_totals.values())
    customdata = []
    for base, label in zip(base_labels, display_labels):
        count = node_counts.get(base, 0)
        cls = extract_class_from_base(base)
        if SEPARATE_SANKEY_BY_CLASS and "|" in base:
            node_cls_counts = class_node_counts.get(base, {})
            for c in CLASSES_ORDER:
                if base.endswith(f"|{c}"):
                    class_count = node_cls_counts.get(c, 0)
                    pct_class = 100.0 * class_count / class_totals[c] if class_totals.get(c) else 0.0
                    pct_all = 100.0 * class_count / total_all if total_all else 0.0
                    hover = f"{label}<br>n={class_count} / {class_totals.get(c,0)}  ({pct_class:.1f}% of {c})<br>{pct_all:.1f}% of All"
                    break
            else:
                hover = f"{label}<br>n={count}<br>Unknown class"
        elif cls and cls in class_totals:
            pct_class = 100.0 * count / class_totals[cls] if class_totals.get(cls) else 0.0
            pct_all   = 100.0 * count / total_all if total_all else 0
            hover = f"{label}<br>n={count} / {class_totals.get(cls,0)}  ({pct_class:.1f}% of {cls})<br>{pct_all:.1f}% of All"
        else:
            pct_all = 100.0 * count / total_all if total_all else 0
            hover = f"{label}<br>n={count}<br>{pct_all:.1f}% of All"
        customdata.append(hover)

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=30,          # was 25, then 30
            thickness=40,    # was 40, then 34
            line=dict(color="rgba(0,0,0,0)", width=0),
            label=[""] * len(display_labels),  # hide labels; hover shows info
            color=node_colors,
            customdata=customdata,
            hovertemplate='%{customdata}<extra></extra>',
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=colors,
            customdata=hover_texts,
            hovertemplate="%{customdata}<extra></extra>"
        ),
        textfont=dict(color="#222222", size=13, family="Helvetica Neue, Helvetica, Arial, sans-serif")
    )])

    # Apply transparent background if requested
    if TRANSPARENT_BG:
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)')
        trans_flag = 1
    else:
        fig.update_layout(paper_bgcolor='white',
                          plot_bgcolor='white')
        trans_flag = 0

    # Tight layout: remove excess whitespace between diagram and figure border
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))  # shrink these values to taste

    # --- Save ---
    VERSION = f'06_multi_dual{str(SEPARATE_SANKEY_BY_CLASS)}'
    out_base = os.path.join(output_dir, f"sankey_v{VERSION}_{mode}_{threshold_key}_clr{str(COLOR_PALETTE)}_alpha{str(ALPHA_PALETTE)}_trans{str(trans_flag)}")

    if WRITE_SVG:
        fig.write_image(out_base + ".svg", width=2400, height=1200)
        print(f"[OK] Saved: {out_base}.svg")
    if WRITE_PDF:
        fig.write_image(out_base + ".pdf", width=2400, height=1200)
        print(f"[OK] Saved: {out_base}.pdf")
    if WRITE_PNG:
        fig.write_image(out_base + ".png", width=2400, height=1200, scale=2)
        print(f"[OK] Saved: {out_base}.png")
    if WRITE_HTML:
        fig.write_html(out_base + ".html")
        print(f"[OK] Saved: {out_base}.html")
    return fig

# ---------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------
def main():
    # Determine which thresholds to run
    if THRESHOLDS_TO_RUN == "auto":
        thresholds = discover_thresholds(df_init)
    else:
        thresholds = list(THRESHOLDS_TO_RUN)

    # Loop thresholds × modes
    for thr_key in thresholds:
        df_thr = slice_df_for_threshold(df_init, thr_key)
        if df_thr.empty:
            print(f"[SKIP] Threshold '{thr_key}': no rows.")
            continue

        for mode in RUN_MODES:
            build_and_save_sankey(df_thr.copy(), mode, thr_key)

if __name__ == "__main__":
    main()
