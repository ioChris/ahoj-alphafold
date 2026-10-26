# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 11:11:22 2025

@author: Christos P. Feidakis
"""

import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import zipfile
import ast
from collections import defaultdict
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Patch
import gzip, shutil, json, datetime



# Muted styles for threshold markers (LJ keeps its own style)
ST_LINESTYLE = "--"
ST_LINEWIDTH = 1.8

LJ_COLOR = "#2E7D32"    # dark green
LJ_LINESTYLE = "--"
LJ_LINEWIDTH = 1.8      # previously 2.2

EDGE_EPS = 1e-4 #1e-3         # small nudge to avoid drawing exactly on the axis (e.g., 100%)

def _grey_hex(frac: float) -> str:
    """
    Convert a [0..1] lightness fraction to a hex grey.
    frac=1 -> white, frac=0 -> black.
    """
    frac = max(0.0, min(1.0, float(frac)))
    val = int(round(frac * 255))
    comp = f"{val:02X}"
    return f"#{comp}{comp}{comp}"

def _make_grey_ramp(n: int, start: float = 0.78, end: float = 0.38) -> list:
    """
    Create n grey shades linearly spaced from 'start' (lighter) to 'end' (darker).
    Defaults chosen to be pleasantly muted and distinct.
    """
    if n <= 0:
        return []
    if n == 1:
        return [_grey_hex((start + end) / 2.0)]
    import numpy as _np
    levels = _np.linspace(start, end, num=n)
    return [_grey_hex(x) for x in levels]




# Placeholder: implement your own!
def collapse_identical_entries(df):
    """
    Collapse entries with identical sets of uniprot residues.
    Returns a new DataFrame with collapsed entries.
    """
    df['residues_tuple'] = df['uniprot_residues'].apply(lambda x: tuple(sorted(x)))
    collapsed = df.groupby('residues_tuple').agg({
        'job_name': lambda x: list(x),
        # add other aggregations as needed
    }).reset_index()
    return collapsed


def split_interface_entries(df):
    split_rows = []
    for idx, row in df.iterrows():
        uniprot_val = row['UNPs']
        if pd.isna(uniprot_val) or str(uniprot_val).strip() == '':
            continue

        uniprot_ids = str(uniprot_val).split()

        residues = row['uniprot_residues']
        if isinstance(residues, str):
            residues = ast.literal_eval(residues)

        for unp in uniprot_ids:
            unp_residues = [res for res in residues if res.startswith(f'{unp}_')]
            new_row = row.copy()
            new_row['UNPs'] = unp
            new_row['uniprot_residues'] = unp_residues
            split_rows.append(new_row)

    return pd.DataFrame(split_rows)


def hierarchical_cluster_jaccard(residue_sets, threshold):
    """
    Cluster residue sets using Jaccard similarity and hierarchical clustering.
    Returns cluster labels for each entry.
    """
    n = len(residue_sets)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            a = set(residue_sets[i])
            b = set(residue_sets[j])
            jacc = len(a & b) / len(a | b) if (a | b) else 1.0
            dist[i, j] = dist[j, i] = 1 - jacc
    dist_condensed = dist[np.triu_indices(n, 1)]
    Z = linkage(dist_condensed, method='average')
    labels = fcluster(Z, t=1-threshold, criterion='distance')
    return labels, Z


def get_cluster_labels_from_linkage(Z, distance_threshold):
    return fcluster(Z, t=distance_threshold, criterion='distance')


def _similarity_percent_formatter(x, pos=None):
    # x is Jaccard distance on the dendrogram axis -> convert to similarity %
    sim = (1.0 - float(x)) * 100.0
    return f"{int(round(sim))}"


def save_annotated_ubs_dendrogram(
    linkage_matrix,
    labels,
    residue_sets,
    save_path,
    show_threshold="none",           # kept for backward compatibility: "none", "LJ", "030", ...
    thresholds_to_mark=None,         # NEW: "auto" | list[str] like ["030","050","070","100","LJ"]
    large_jump_similarity=None,      # float in [0,1]
    orientation='right',
    legend_loc: str = "upper right",
    legend_outside: bool = False,
    transparency: float = 0.0,   # NEW: 0=opaque outside, 1=fully transparent outside

):
    """
    Save an annotated dendrogram as SVG.

    Changes:
      • Axis shows Jaccard SIMILARITY (%) instead of distance.
      • No default threshold line is shown.
      • Optional threshold marker: show_threshold in {"none","LJ","030","050","070","100"}.
      • Merge annotations remain "shared/total".
      Changes vs single-marker version:
      • thresholds_to_mark lets you draw multiple vertical/horizontal lines.
      • Accepts "auto" to draw all default STs you pass in from the runner + LJ (if present).
      • 'show_threshold' is still supported for single marker (legacy).
    """

    if len(labels) < 2:
        print("Not enough entries to plot a dendrogram.")
        return

    plt.figure(figsize=(8, max(4, len(labels) // 8))) # 9
    pretty_labels = [f"{job} ({len(res)})" for job, res in zip(labels, residue_sets)]
    ddata = dendrogram(
        linkage_matrix,
        labels=pretty_labels,
        orientation=orientation,
        leaf_font_size=8
    )

    ax = plt.gca()
    
    # --- Selective transparency: inside axes solid white, outside follows `transparency`
    # accept 0..1 or 0..100 (%)
    try:
        _t = float(transparency)
    except Exception:
        _t = 0.0
    if _t > 1.0:
        _t = min(1.0, _t / 100.0)
    if _t < 0.0:
        _t = 0.0
    _outer_alpha = 1.0 - _t
    
    # Inside plotting area (axes): keep solid white
    ax.set_facecolor('white')
    ax.patch.set_alpha(1.0)
    
    # Outside (figure background): use same color, but with alpha from `transparency`
    fig = ax.figure
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(_outer_alpha)

    
    # Thicken non-zero branches a bit, highlight zero-height arms slightly more
    for collection in ax.collections:
        segments = collection.get_segments()
        linewidths = []
        for seg in segments:
            if np.allclose(seg[:, 0], 0, atol=1e-8):
                linewidths.append(3.5)
            else:
                linewidths.append(2.0)
        collection.set_linewidth(linewidths)

    # Grid / ticks — keep geometric spacing in distance but *label* as similarity (%)
    if orientation == "right":
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))   # every 10% similarity
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.02))  # every 2%
        ax.xaxis.set_major_formatter(FuncFormatter(_similarity_percent_formatter))
        ax.grid(True, which='major', axis='x', linestyle='-', linewidth=1, alpha=0.3)
        ax.grid(True, which='minor', axis='x', linestyle=':', linewidth=0.5, alpha=0.2)
        ax.set_xlabel('Jaccard Similarity (%)', fontweight='bold')
    else:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.02))
        ax.yaxis.set_major_formatter(FuncFormatter(_similarity_percent_formatter))
        ax.grid(True, which='major', axis='y', linestyle='-', linewidth=1, alpha=0.3)
        ax.grid(True, which='minor', axis='y', linestyle=':', linewidth=0.5, alpha=0.2)
        ax.set_ylabel('Jaccard Similarity (%)', fontweight='bold')

    # --- Threshold markers (multi or single) ---------------------------------
    def _draw_marker(sim_float: float, label_txt: str, *, color: str, linestyle: str, lw: float):
        # convert similarity->distance for placement on dendrogram axis
        xdist = 1.0 - float(sim_float)
        # Optional nudge if exactly 100% (avoids overlapping the axis line)
        if np.isclose(sim_float, 1.0, atol=1e-12):
            xdist = xdist + EDGE_EPS

        if orientation == "right":
            ax.axvline(x=xdist, color=color, linestyle=linestyle, linewidth=lw, label=label_txt)
        else:
            ax.axhline(y=xdist, color=color, linestyle=linestyle, linewidth=lw, label=label_txt)

    added_any_marker = False

    def _compute_st_color_map(code_list):
        """Build a mapping {int_code: grey_hex}, lighter->darker as similarity increases."""
        # Keep only numeric ST codes (exclude "LJ"), sorted ascending (e.g., 30, 50, 70, 100)
        st_ints = []
        for c in code_list:
            try:
                st_ints.append(int(c))
            except Exception:
                pass
        st_ints = sorted(set(st_ints))
        greys = _make_grey_ramp(len(st_ints), start=0.78, end=0.38)  # tweak if you want lighter/darker
        return {k: v for k, v in zip(st_ints, greys)}

    # Prefer the new 'thresholds_to_mark' if provided
    if thresholds_to_mark is not None:
        if isinstance(thresholds_to_mark, str) and thresholds_to_mark.lower() == "auto":
            codes = getattr(linkage_matrix, "_st_codes", ["030","050","070","100"])
            if (large_jump_similarity is not None) and ("LJ" not in codes):
                codes = list(codes) + ["LJ"]
        elif isinstance(thresholds_to_mark, (list, tuple)):
            codes = list(thresholds_to_mark)
        else:
            codes = []

        st_color_map = _compute_st_color_map(codes)
        seen_labels = set()
        for code in codes:
            if isinstance(code, str) and code.upper() == "LJ":
                if large_jump_similarity is not None:
                    lbl = f"LJ {int(round(large_jump_similarity*100)):03d}"
                    if lbl not in seen_labels:
                        _draw_marker(
                            large_jump_similarity, lbl,
                            color=LJ_COLOR, linestyle=LJ_LINESTYLE, lw=LJ_LINEWIDTH
                        )
                        seen_labels.add(lbl)
                        added_any_marker = True
                continue
            # Expect 3-digit codes like "030","050","070","100"
            try:
                sim_pct = int(code)
                sim = sim_pct / 100.0
                lbl = f"ST {sim_pct:03d}"
                if lbl not in seen_labels:
                    color = st_color_map.get(sim_pct, _grey_hex(0.55))  # default mid-grey if unmatched
                    _draw_marker(
                        sim, lbl,
                        color=color, linestyle=ST_LINESTYLE, lw=ST_LINEWIDTH
                    )
                    seen_labels.add(lbl)
                    added_any_marker = True
            except Exception:
                pass

    # Legacy single-marker support (if thresholds_to_mark wasn't used)
    elif isinstance(show_threshold, str) and show_threshold.lower() != "none":
        if show_threshold.upper() == "LJ" and (large_jump_similarity is not None):
            lbl = f"LJ {int(round(large_jump_similarity*100)):03d}"
            _draw_marker(
                large_jump_similarity, lbl,
                color=LJ_COLOR, linestyle=LJ_LINESTYLE, lw=LJ_LINEWIDTH
            )
            added_any_marker = True
        else:
            try:
                sim_pct = int(show_threshold)
                sim = sim_pct / 100.0
                color = _make_grey_ramp(1)[0]
                _draw_marker(
                    sim, f"ST {sim_pct:03d}",
                    color=color, linestyle=ST_LINESTYLE, lw=ST_LINEWIDTH
                )
                added_any_marker = True
            except Exception:
                pass

    if added_any_marker:
        # Tunable legend appearance (defaults mimic your old semi-opaque box)
        _legend_kwargs = dict(
            fontsize=9,
            frameon=True,          # <— turn the frame back on
            fancybox=True,         # rounded corners
            framealpha=0.65,       # transparency of the legend box
            facecolor="white",     # box fill
            edgecolor="#666",      # no border line; set to '#666' if you want a hairline
        )

        if legend_outside:
            ax.legend(
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0,
                **_legend_kwargs
            )
            # leave space on the right for the outside legend
            plt.tight_layout(rect=[0.0, 0.0, 0.88, 1.0])
        else:
            ax.legend(loc=legend_loc, **_legend_kwargs)


    # --------------------------------------------------------------------------




    # Merge annotations (shared/total), unchanged logic
    annotate_merge_sizes(ax, linkage_matrix, residue_sets, ddata, orientation=orientation)

    plt.tight_layout()
    plt.savefig(save_path, format='svg')
    plt.close()


def annotate_merge_sizes(ax, linkage_matrix, residue_sets, ddata, orientation="right"):

    n_leaves = len(residue_sets)
    cluster_to_residues = {i: set(residue_sets[i]) for i in range(n_leaves)}
    for i, (left, right, _, _) in enumerate(linkage_matrix):
        left = int(left)
        right = int(right)
        new_cluster = n_leaves + i
        cluster_to_residues[new_cluster] = cluster_to_residues[left] | cluster_to_residues[right]

    heights = linkage_matrix[:, 2]
    max_height = np.max(heights) if len(heights) else 0

    height_to_idxs = defaultdict(list)
    for idx, h in enumerate(heights):
        height_to_idxs[round(h, 8)].append(idx)

    for icoord, dcoord in zip(ddata['icoord'], ddata['dcoord']):
        node_height = dcoord[1] if orientation == "right" else icoord[1]
        if np.isclose(node_height, 0, atol=1e-8) or np.isclose(node_height, max_height, atol=1e-8):
            continue
        hkey = round(node_height, 8)
        idxs = height_to_idxs[hkey]
        if not idxs:
            continue
        merge_idx = idxs.pop(0)
        left = int(linkage_matrix[merge_idx, 0])
        right = int(linkage_matrix[merge_idx, 1])
        union = cluster_to_residues[n_leaves + merge_idx]
        inter = cluster_to_residues[left] & cluster_to_residues[right]

        x = dcoord[1] if orientation == "right" else (icoord[1] + icoord[2]) / 2
        y = (icoord[1] + icoord[2]) / 2 if orientation == "right" else dcoord[1]
        ax.text(
            x, y, f"{len(inter)}/{len(union)}",
            color="black", fontsize=8.5, fontweight='bold', va='bottom', ha='center',
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=0.5)
        )


def save_pickle(obj, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(obj, f)


def save_ubs_table(clustered_df, save_path):
    clustered_df.to_csv(save_path, sep='\t', index=False)


def compress_file(filepath: str, method: str = "zip", remove_original: bool = False) -> str:
    """
    Compress a single file as .zip or .gz.
    Returns the path of the compressed file.
    """
    method = (method or "zip").lower()
    if method not in {"zip", "gz"}:
        method = "zip"

    if method == "zip":
        zip_path = filepath + ".zip"
        with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(filepath, arcname=os.path.basename(filepath))
        if remove_original:
            os.remove(filepath)
        return zip_path

    # gzip (single-file stream)
    gz_path = filepath + ".gz"
    with open(filepath, "rb") as f_in, gzip.open(gz_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    if remove_original:
        os.remove(filepath)
    return gz_path


def write_uniprot_manifest(
    out_dir: str,
    uniprot_id: str,
    thresholds: list[dict],
    figures: dict | None = None,
    input_data_file: str | None = None,
    archive_format: str = "zip",
    extra: dict | None = None,
) -> str:
    """
    Write a JSON manifest describing generated files for a UniProt.

    thresholds: list of items like:
      {
        "JaccardSimilarityThreshold": 0.5,
        "label": "ST_050",
        "type": "fixed",         # or "optimal" (for LJ)
        "file": "ST_050.tsv",
        "shortDescription": "...",
        "longDescription": "..."
      }
    figures: {"dendrogram": "dendrogram.svg.zip", "histogram": "binding_positions.svg.gz"}
    """
    manifest = {
        "uniprot": uniprot_id,
        "generatedAt": datetime.datetime.now().isoformat(timespec="seconds"),
        "archiveFormat": archive_format,
        "thresholds": thresholds or [],
        "figures": figures or {},
        "inputData": input_data_file,
    }
    if extra:
        manifest.update(extra)

    out_path = os.path.join(out_dir, "ubs_manifest.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=False)
    return out_path


def _extract_unp_index(token: str):
    """
    Parse UniProt index from strings like 'P02185_40' -> 40.
    Returns None if it can't parse an integer.
    """
    if not isinstance(token, str) or "_" not in token:
        return None
    try:
        idx_str = token.rsplit("_", 1)[1]
        return int(idx_str)
    except Exception:
        return None


def compute_uniprot_binding_counts(df_uniprot):
    """
    Compute per-position counts for a UniProt-split dataframe.

    Returns:
      positions: sorted list of observed UniProt indices
      total_counts: dict {pos -> #jobs where pos binds}
      interface_counts: dict {pos -> #jobs where pos binds in interface (prot_chains_num >= 2)}
      n_jobs: number of unique jobs considered
    """
    total_counts = {}
    interface_counts = {}

    # Denominator for percentages: number of unique jobs for this UniProt
    n_jobs = int(df_uniprot['job_name'].nunique())
    n_pdb = int(df_uniprot['pdb'].nunique()) if 'pdb' in df_uniprot.columns else None

    # Group by job to count a position at most once per job
    for job, sub in df_uniprot.groupby('job_name'):
        # union of positions in this job (dedup within job)
        pos_set = set()
        is_interface = False
        for _, row in sub.iterrows():
            # interface if ANY row of this job has ≥2 protein chains
            try:
                if int(row.get('prot_chains_num', 1)) >= 2:
                    is_interface = True
            except Exception:
                pass
            residues = row.get('uniprot_residues', [])
            if isinstance(residues, str):
                # safety: parse stringified list
                try:
                    import ast
                    residues = ast.literal_eval(residues)
                except Exception:
                    residues = []
            for tok in residues:
                pos = _extract_unp_index(tok)
                if pos is not None:
                    pos_set.add(pos)

        # Update counts
        for pos in pos_set:
            total_counts[pos] = total_counts.get(pos, 0) + 1
            if is_interface:
                interface_counts[pos] = interface_counts.get(pos, 0) + 1

    positions = sorted(total_counts.keys())
    return positions, total_counts, interface_counts, n_jobs, n_pdb

# COLORS
# Pastel, publication-friendly 
col_mono = "#CBD5E1"  # pastel slate grey/blue
col_iface = "#5B8FD1" # NEW: more-contrasty pastel blue
#col_iface = "#dfc7a7" # pastel brown 1
#col_iface = "#cebaa0" # "#dfc7a7" # pastel brown cebaa0
#col_mono = "#a8e6cf" # pastel rainbow green
#col_iface = "#9FD3C7" # pastel teal
# Pastel green
#col_mono = "#e8f4ea" # pastel green 1
#col_iface = "#b8d8be" # pastel green 5
# Pastel Palette Color Palette
#col_mono = "#b8dbd3" # 1
#col_iface = "#f7e7b4" # 2
edge = "#4A4A4A"       # softer outline


def _decade_step(xmin: int, xmax: int, target_ticks: int = 8, min_step: int = 10) -> int:
    """
    Choose a 'nice' tick step aligned to decades (…10, 20, 50, 100, 200, …)
    so we end up with ~target_ticks major ticks, never using odd steps like 21.
    """
    span = max(1, int(xmax - xmin))
    # start from a power of 10 close to span/target
    base = 10 ** int(np.floor(np.log10(max(1.0, span / float(target_ticks)))))
    for m in (1, 2, 5, 10):
        step = int(base * m)
        if step < min_step:
            continue
        if span / step <= target_ticks:
            return step
    return int(max(min_step, base * 10))


def save_uniprot_binding_histogram(
    df_uniprot,
    uniprot_id: str,
    save_path: str,
    as_percent: bool = True,
    show_interface_split: bool = True,
    figsize=(10, 3.2),
    facecolor='white',
    #transparency=0,
    annotate_counts: bool = True,
    histogram_style: str = "stepfill",   # "stepfill" (joined bars) or "bars"
    legend_loc: str = "best",
    annotate_counts_loc: str = "auto",
    x_axis_start: int = 0,               # force UniProt numbering start (default = 1)
    x_axis_full_length: bool = False,    # span to full sequence length if available
    uniprot_length_col: str | None = None,
    x_edge_pad: float = 0, # prev 0.6,             # symmetric half-bin pad left/right (prevents clipping)
    tick_mode: str = "decade10",     # "decade10" (default) or "auto"
    show_zero_tick: bool = True,     # include 0 at left if True
    minor_tick_mode: str = "auto",   # "auto" | "half" | "five" | "none"
    # NEW:
    top_outline: bool = True,
    outline_color: str = "#D1D5DB",   # light neutral grey
    outline_lw: float = 0.8,
    transparency: float = 0.0,

):
    positions, total_counts, interface_counts, n_jobs, n_pdb = compute_uniprot_binding_counts(df_uniprot)
    if len(positions) == 0 or n_jobs == 0:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, f"No binding residues for {uniprot_id}", ha='center', va='center')
        ax.axis('off')
        fig.savefig(save_path, format='svg', facecolor=facecolor)
        plt.close(fig)
        return

    # --- Decide x-axis coverage: always start at 1; end at seq length or max observed
    positions_arr = np.array(positions, dtype=int)
    seq_len = None
    if x_axis_full_length:
        candidates = [uniprot_length_col] if uniprot_length_col else []
        candidates += ['UNP_len', 'uniprot_length', 'seq_len', 'unp_len']
        for c in candidates:
            if c and c in df_uniprot.columns:
                vals = df_uniprot[c].dropna()
                if len(vals):
                    try:
                        seq_len = int(vals.iloc[0]); break
                    except Exception:
                        pass
    x_start = max(1, int(x_axis_start))
    x_end = int(seq_len) if (seq_len is not None and seq_len >= x_start) else int(positions_arr.max())
    x_left, x_right = (x_start - x_edge_pad), (x_end + x_edge_pad)  # half-bin pad each side

    # Data split
    total_vals = np.array([total_counts[p] for p in positions], dtype=float)
    interface_vals = np.array([interface_counts.get(p, 0) for p in positions], dtype=float)
    monomer_vals = total_vals - interface_vals

    if as_percent:
        scale = 100.0 / float(n_jobs)
        total_plot = total_vals * scale
        interface_plot = interface_vals * scale
        monomer_plot = monomer_vals * scale
        ylab = "Binding prevalence (%)"
    else:
        total_plot = total_vals
        interface_plot = interface_vals
        monomer_plot = monomer_vals
        ylab = "Binding counts"

    # --- OUTER background transparency (outside axes) -----------------------
    # transparency: 0 = opaque outer background, 1 = fully transparent outer background
    # also accept 0..100 as percent
    try:
        _t = float(transparency)
    except Exception:
        _t = 0.0
    if _t > 1.0:
        _t = min(1.0, _t / 100.0)
    if _t < 0.0:
        _t = 0.0
    _outer_alpha = 1.0 - _t


    # Pastel, publication-friendly palette
    col_mono = "#CBD5E1"   # pastel slate/blue-grey
    col_iface = "#5B8FD1"  # contrasty pastel blue
    edge = "#4A4A4A"       # soft outline

    fig, ax = plt.subplots(figsize=figsize)

    # Inside axes should always be solid white
    ax.set_facecolor('white')
    ax.patch.set_alpha(1.0)

    # Outside (figure background) color is usually white; alpha comes from `transparency`
    # If you still want to let callers change the outer color, keep using the existing `facecolor` arg.
    fig.patch.set_facecolor(facecolor if facecolor else 'white')
    fig.patch.set_alpha(_outer_alpha)


    if histogram_style.lower() == "bars":
        # Stacked bars, but keep same x padding
        npos = len(positions_arr)
        barw = 0.85 if npos < 50 else max(0.30, 40.0 / npos)
        if show_interface_split:
            ax.bar(positions_arr, monomer_plot, width=barw, color=col_mono, edgecolor=edge, linewidth=0.5,
                   label="Monomer (1 chain)")
            ax.bar(positions_arr, interface_plot, bottom=monomer_plot, width=barw, color=col_iface,
                   edgecolor=edge, linewidth=0.5, label="Interface (≥2 chains)")
            plotted_total = monomer_plot + interface_plot
        else:
            ax.bar(positions_arr, total_plot, width=barw, color=col_mono, edgecolor=edge, linewidth=0.6,
                   label="All binding residues")
            plotted_total = total_plot
        ax.set_xlim(x_left, x_right)
    else:
        # Stepfill silhouette from required start to end
        x_all = np.arange(x_start, x_end + 1, dtype=int)
        mono_dense  = np.zeros_like(x_all, dtype=float)
        iface_dense = np.zeros_like(x_all, dtype=float)
        total_dense = np.zeros_like(x_all, dtype=float)
        pos_to_idx = {p: i for i, p in enumerate(x_all)}
        for p, m, i in zip(positions_arr, monomer_plot, interface_plot):
            idx = pos_to_idx[int(p)]
            mono_dense[idx] = m
            iface_dense[idx] = i
            total_dense[idx] = m + i

        # Fill layers and total outline
        ax.fill_between(x_all, 0.0, mono_dense, step="mid",
                        facecolor=col_mono, edgecolor=edge, linewidth=0.6, alpha=1.0,
                        label="Monomer (1 chain)" if show_interface_split else "All binding residues")
        if show_interface_split:
            ax.fill_between(x_all, mono_dense, total_dense, step="mid",
                            facecolor=col_iface, edgecolor=edge, linewidth=0.6, alpha=1.0,
                            label="Interface (≥2 chains)")
        ax.step(x_all, total_dense, where="mid", color=edge, linewidth=0.7) # prev linewidth=0.9
        plotted_total = total_dense
        ax.set_xlim(x_left, x_right)

    # Y baseline & headroom
    ymax = float(np.nanmax(plotted_total)) if np.size(plotted_total) else 0.0
    ax.set_ylim(0.0, (1.05 * ymax) if ymax > 0 else 1.0)  # +5% headroom
    # IMPORTANT: remove hard x-margins clamp so we keep gaps at ends
    # ax.margins(x=0.0)   # <-- removed on purpose
    ax.margins(y=0.0)     # keep baseline on x-axis

    # Secondary Y (counts vs %), synced to left
    right_ax = ax.twinx()
    if as_percent:
        right_ax.set_ylim(0.0, (ax.get_ylim()[1] * n_jobs) / 100.0)
        right_ax.set_ylabel("Binding counts")
        right_ax.yaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
    else:
        left_top = ax.get_ylim()[1]
        right_ax.set_ylim(0.0, (left_top * 100.0 / n_jobs) if n_jobs else 100.0)
        right_ax.set_ylabel("Binding prevalence (%)")
        right_ax.yaxis.set_major_locator(MaxNLocator(nbins=7, integer=False))
    right_ax.spines['top'].set_visible(False)
    right_ax.grid(False)

    # Axis & grid polish
    ax.set_axisbelow(True)
    ax.set_xlabel("UniProt position")
    ax.set_ylabel(ylab)
    ax.set_title(f"{uniprot_id}: Binding-residue prevalence", fontsize=11, pad=8)

    # Decade-aligned ticks: multiples of 10/20/50/100… anchored at 0
    step = _decade_step(0, x_end, target_ticks=10, min_step=10)
    first_tick = 0  # anchor to 0 so labels are 0, 10, 20, 30, …
    last_tick  = int(np.ceil(x_end / step) * step)
    ticks = np.arange(first_tick, last_tick + 1, step, dtype=int)

    ax.set_xticks(ticks)
    ax.set_xlim(x_left, x_right)

    # Add labeled midpoints when major step is big (e.g., 20, 50, 100…)
    if step >= 20:
        mids = ticks[:-1] + (step // 2)          # e.g., 0..50 -> 25
        all_ticks = np.sort(np.unique(np.r_[ticks, mids]))
        if all_ticks.size <= 14:                  # avoid clutter
            ax.set_xticks(all_ticks)

    # ensure integer labels
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{int(v):d}"))
    # ensure no minor ticks/gridlines
    ax.xaxis.set_minor_locator(ticker.NullLocator())

    ax.grid(True, axis='y', linestyle=':', linewidth=0.7, alpha=0.55)
    ax.spines[['top']].set_visible(True) # Top outline ON
    #ax.spines[['top', 'right']].set_visible(False)

    # No top ticks/labels; bottom ticks unchanged
    ax.tick_params(axis='x', which='both', top=False, labeltop=False)

    # Spines: keep right hidden; top is an optional outline
    #ax.spines['right'].set_visible(False)
    #if top_outline:
    #    ax.spines['top'].set_visible(True)
    #    ax.spines['top'].set_color(outline_color)  # e.g., "#D1D5DB"
    #    ax.spines['top'].set_linewidth(outline_lw) # e.g., 0.8
    #else:
    #    ax.spines['top'].set_visible(False)


    # Legend
    legend_handles = None
    if show_interface_split:
        if histogram_style.lower() == "stepfill":
            legend_handles = [
                Patch(facecolor=col_mono, edgecolor=edge, label="Monomer (1 chain)"),
                Patch(facecolor=col_iface, edgecolor=edge, label="Interface (≥2 chains)")
            ]
        ax.legend(handles=legend_handles, frameon=False, ncol=1, fontsize=9, loc=legend_loc)

    # Counts text: outside, above axes (never overlaps)
    if annotate_counts:
        bits = [f"Entries: {n_jobs}"]
        if n_pdb is not None:
            bits.append(f"PDBs: {n_pdb}")
        counts_str = " • ".join(bits)

        # Make the counts box match the outer background transparency.
        # If transparency > 0, use RGBA facecolor and NO separate alpha (to avoid double-alpha).
        # If transparency == 0, keep a subtle, opaque-ish box for readability.
        if _outer_alpha < 1.0:  # some transparency requested
            box_face = (1.0, 1.0, 1.0, _outer_alpha)   # white with same alpha as the figure patch
            box_kwargs = dict(facecolor=box_face, edgecolor='none', pad=2)
        else:
            box_kwargs = dict(facecolor='white', edgecolor='none', pad=2)
            # If you still want semi-opaque when no transparency:
            # box_kwargs = dict(facecolor=(1,1,1,0.80), edgecolor='none', pad=2)

        ax.text(
            0.995, 1.015, counts_str,
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=9, color='#333',
            bbox=box_kwargs,   # <— no 'alpha' key; alpha lives inside the RGBA facecolor
            clip_on=False, zorder=99)


    # Reserve top room for counts text
    #fig.tight_layout(rect=[0.0, 0.0, 1.0, 1.0])
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
    fig.savefig(save_path, format='svg') #, facecolor=facecolor)
    plt.close(fig)

