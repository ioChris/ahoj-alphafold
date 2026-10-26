# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 15:32:33 2025

@author: Christos P. Feidakis
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"

import os


# Use hex codes for publication quality (no transparency artifacts)
LEVELS = ['Density', 'Separation', 'Prob']

APO_COLORS = {
    'Density':   {'Low': '#dbeafe', 'Moderate': '#93c5fd', 'High': '#2563eb'},
    'Separation':{'Low': '#c7d2fe', 'Moderate': '#818cf8', 'High': '#3730a3'},
    'Prob':      {'Ambiguous': '#bae6fd', 'Moderate': '#38bdf8', 'High': '#0ea5e9'}
}
HOLO_COLORS = {
    'Density':   {'Low': '#fee2e2', 'Moderate': '#fca5a5', 'High': '#b91c1c'},
    'Separation':{'Low': '#fecaca', 'Moderate': '#f87171', 'High': '#7f1d1d'},
    'Prob':      {'Ambiguous': '#fca5a5', 'Moderate': '#fb7185', 'High': '#be123c'}
}
TIE_COLOR = '#cccccc'

GRAY_DENSITY = {"Low": "#e5e7eb", "Moderate": "#d1d5db", "High": "#a3a3a3"}
GRAY_SEPARATION = {"Low": "#d1d5db", "Moderate": "#a3a3a3", "High": "#525252"}

APO_CLASS = {"Ambiguous": "#cfe2f3", "Moderate": "#6fa8dc", "High": "#073763"}
HOLO_CLASS = {"Ambiguous": "#f4cccc", "Moderate": "#ea9999", "High": "#990000"}
TIE_CLASS = "#bcbcbc"


# --- Classification functions ---
def classify_probability0(p):
    try:
        p = float(p)
    except (TypeError, ValueError):
        return "No data"
    if pd.isna(p):
        return "No data"
    if p <= 0.60:
        return "Ambiguous"
    elif p <= 0.65:
        return "Low"
    elif p <= 0.75:
        return "Moderate"
    else:
        return "High"

def classify_density0(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "No data"
    if pd.isna(x):
        return "No data"
    if x <= 0.60:
        return "Very low"
    elif x <= 0.65:
        return "Low"
    elif x <= 0.75:
        return "Moderate"
    else:
        return "High"

def classify_separation0(overlap):
    try:
        overlap = float(overlap)
    except (TypeError, ValueError):
        return "No data"
    if pd.isna(overlap):
        return "No data"
    sep = 1 - overlap
    return classify_density(sep)

def classify_density(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "No data"
    if pd.isna(x):
        return "No data"
    if x <= 0.3:
        return "Low"
    elif x <= 0.7:
        return "Moderate"
    else:
        return "High"

def classify_separation(overlap):
    try:
        overlap = float(overlap)
    except (TypeError, ValueError):
        return "No data"
    if pd.isna(overlap):
        return "No data"
    sep = 1 - overlap
    return classify_density(sep)

def classify_probability(p):
    try:
        p = float(p)
    except (TypeError, ValueError):
        return "No data"
    if pd.isna(p):
        return "No data"
    if p <= 0.60:
        return "Ambiguous"
    elif p <= 0.70:
        return "Moderate"
    else:
        return "High"

def has_prob_data(row):
    return not (pd.isna(row['highest_p_mean_pocket']) or row['highest_p_mean_pocket'] == 0)

def is_low_n(row):
    try:
        apo = float(row['sum_n_apo'])
        holo = float(row['sum_n_holo'])
    except:
        return True  # treat as low if missing/invalid
    return (apo < 2 or holo < 2)

def normalize_threshold(thresh):
    if isinstance(thresh, str) and "UBS_LJ" in thresh:
        return "UBS_LJ"
    return thresh

def get_max_prob(row):
    try:
        pa = float(row.get('avg_P(apo)_mean_pocket', np.nan))
        ph = float(row.get('avg_P(holo)_mean_pocket', np.nan))
        if pd.isna(pa) and pd.isna(ph):
            return np.nan
        elif pd.isna(pa):
            return ph
        elif pd.isna(ph):
            return pa
        else:
            return max(pa, ph)
    except Exception:
        return np.nan

def get_link_ubs_mask(sub, from_label, to_label):
    # Build a mask for UBSs that traverse this link
    if from_label.startswith("Density: ") and to_label.startswith("Separation: "):
        dens = from_label.replace("Density: ", "")
        sep = to_label.replace("Separation: ", "")
        return (sub['dens_class'] == dens) & (sub['sep_class'] == sep)
    elif from_label.startswith("Separation: ") and to_label.startswith("Prob: "):
        sep = from_label.replace("Separation: ", "")
        prob = to_label.replace("Prob: ", "")
        return (sub['sep_class'] == sep) & (sub['prob_class'] == prob)
    elif from_label.startswith("Prob: ") and to_label.startswith("Class: "):
        prob = from_label.replace("Prob: ", "")
        cls = to_label.replace("Class: ", "")
        return (sub['prob_class'] == prob) & (sub['final_class'] == cls)
    elif from_label == root_label and to_label.startswith("Density: "):
        dens = to_label.replace("Density: ", "")
        return (sub['dens_class'] == dens)
    else:
        # Fallback: all
        return pd.Series([True] * len(sub))


###############################################################################










# --- Load data ---
# --- locate repo root so `common` is importable (works standalone / in Spyder) ---
import sys as _sys
from pathlib import Path as _Path
for _root in _Path(__file__).resolve().parents:
    if (_root / "common" / "config.py").is_file():
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))
        break
from common import config as cfg

# Legacy v1.2 UBS05+LJ subset (adjust as needed)
_legacy_dir = cfg.UBS_AGG_DIR / "06082025_122505_v1.2_Noneunplimit"
csv_path = str(_legacy_dir / "ubs_UBS05_and_LJ_subset.csv")

# Output
output_dir = str(_legacy_dir / "sankey_results")
os.makedirs(output_dir, exist_ok=True)


# --- Load the data ---
print('Loading dataframe')
df = pd.read_csv(csv_path, low_memory=False)
df['threshold'] = df['threshold'].apply(normalize_threshold)


# --- Prepare the flow dataframe ---

print('Preparing the flow dataframe')
flow_rows = []
for idx, row in df.iterrows():
    threshold = row['threshold']
    prob_data = has_prob_data(row)
    if not prob_data:
        prob_class = dens_class = sep_class = final_class = "No data"
    else:
        max_prob = get_max_prob(row)
        prob_class = classify_probability(max_prob)
        #prob_class = classify_probability(row['highest_p_mean_pocket'])
        dens_class = classify_density(row['avg_normalized_density_combined_mean_pocket'])
        sep_class = classify_separation(row['pair_overlap_mean_pocket']) # Use the average instead
        #sep_class = classify_separation(row['overall_overlap_mean_pocket'])

        final_class = row['highest_p_mean_pocket'] if isinstance(row['highest_p_mean_pocket'], str) else "No data"
        #final_class = row['majority_vote_pocket'] if isinstance(row['majority_vote_pocket'], str) else "No data"

    low_n_flag = is_low_n(row)
    flow_rows.append({
        "site_recipe": row['site_recipe'],
        "threshold": threshold,
        "prob_data": "Yes" if prob_data else "No",
        "prob_class": prob_class,
        "dens_class": dens_class,
        "sep_class": sep_class,
        "final_class": final_class,
        "low_n": low_n_flag
    })
flow_df = pd.DataFrame(flow_rows)




# --- For Sankey: define nodes and flows ---
# We'll build a branched structure: Threshold -> Has Data -> Prob -> Density -> Sep -> Final Class
# For simplicity, we'll only use the main flow (prob_data, prob_class, dens_class, sep_class, final_class)

# Only analyze a single threshold at a time for clarity (change this for each run)
for threshold in flow_df['threshold'].unique():
    print(f'Analysing and plotting threshold {threshold}')

    sub = flow_df[flow_df['threshold'] == threshold]

    # --- Only include UBSs with data ---
    sub = sub[sub['prob_data'] == "Yes"]

    # Define nodes and mapping
    nodes = []
    node_dict = {}

    def add_node(label):
        if label not in node_dict:
            node_dict[label] = len(nodes)
            nodes.append(label)

    root_label = f"UBSs with Data ({threshold})"
    add_node(root_label)

    for dens in ["Low", "Moderate", "High"]:
        add_node(f"Density: {dens}")
    for sep in ["Low", "Moderate", "High"]:
        add_node(f"Separation: {sep}")
    for pc in ["Ambiguous", "Moderate", "High"]:
        add_node(f"Prob: {pc}")
    for fc in ["apo", "holo", "tie"]:
        add_node(f"Class: {fc}")

    """
    for dens in ["Very low", "Low", "Moderate", "High"]:
        add_node(f"Density: {dens}")
    for sep in ["Very low", "Low", "Moderate", "High"]:
        add_node(f"Separation: {sep}")
    for pc in ["Ambiguous", "Low", "Moderate", "High"]:
        add_node(f"Prob: {pc}")
    for fc in ["apo", "holo", "tie"]:
        add_node(f"Class: {fc}")
    """

    source = []
    target = []
    value = []

    # Root -> Density
    for dens in ["Very low", "Low", "Moderate", "High"]:
        count = len(sub[sub['dens_class'] == dens])
        if count > 0:
            source.append(node_dict[root_label])
            target.append(node_dict[f"Density: {dens}"])
            value.append(count)

    # Density -> Separation
    for dens in ["Very low", "Low", "Moderate", "High"]:
        sub_dens = sub[sub['dens_class'] == dens]
        for sep in ["Very low", "Low", "Moderate", "High"]:
            count = len(sub_dens[sub_dens['sep_class'] == sep])
            if count > 0:
                source.append(node_dict[f"Density: {dens}"])
                target.append(node_dict[f"Separation: {sep}"])
                value.append(count)

    # Separation -> Probability
    for sep in ["Very low", "Low", "Moderate", "High"]:
        sub_sep = sub[sub['sep_class'] == sep]
        for pc in ["Ambiguous", "Low", "Moderate", "High"]:
            count = len(sub_sep[sub_sep['prob_class'] == pc])
            if count > 0:
                source.append(node_dict[f"Separation: {sep}"])
                target.append(node_dict[f"Prob: {pc}"])
                value.append(count)

    # Probability -> Class
    for pc in ["Ambiguous", "Low", "Moderate", "High"]:
        sub_pc = sub[sub['prob_class'] == pc]
        for fc in ["apo", "holo", "tie"]:
            count = len(sub_pc[sub_pc['final_class'] == fc])
            if count > 0:
                source.append(node_dict[f"Prob: {pc}"])
                target.append(node_dict[f"Class: {fc}"])
                value.append(count)

    # Coloring
    # After building your nodes and before the Sankey figure:
    link_colors = []
    for src, tgt, val in zip(source, target, value):
        from_label = nodes[src]
        to_label = nodes[tgt]
    
        # Section 1: UBSs with Data → Density (always light gray)
        if from_label == root_label and to_label.startswith("Density: "):
            color = "#f4f4f5"
        # Section 2: Density → Separation (gray gradient by density)
        elif from_label.startswith("Density: ") and to_label.startswith("Separation: "):
            dens = from_label.replace("Density: ", "")
            color = GRAY_DENSITY.get(dens, "#d1d5db")
        # Section 3: Separation → Probability (darker gray by separation)
        elif from_label.startswith("Separation: ") and to_label.startswith("Prob: "):
            sep = from_label.replace("Separation: ", "")
            color = GRAY_SEPARATION.get(sep, "#a3a3a3")
        # Section 4: Probability → Class (now split by actual class and probability level)
        elif from_label.startswith("Prob: ") and to_label.startswith("Class: "):
            prob = from_label.replace("Prob: ", "")
            cls = to_label.replace("Class: ", "")
            if cls == "apo":
                color = APO_CLASS.get(prob, "#6fa8dc")
            elif cls == "holo":
                color = HOLO_CLASS.get(prob, "#ea9999")
            else:
                color = TIE_CLASS
        else:
            color = "#e5e7eb"  # fallback
        link_colors.append(color)






    # -- Sankey plot --
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=18,
            thickness=18,
            line=dict(color="black", width=0.5),
            label=nodes,
            color="rgba(180,180,200,0.85)",
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=link_colors, #"rgba(100,140,200,0.2)"
        ))])

    fig.update_layout(
        title_text=f"UBS Pocket Classification Flow: {threshold}",
        font_size=12,
        width=1400,
        height=900,
        margin=dict(l=0, r=0, t=40, b=0)
    )

    fig.show()
    #fig.write_image(os.path.join(output_dir, f"sankey_plot_{threshold}_w_data.png"), width=1400, height=900)
    #fig.write_html(os.path.join(output_dir, f"sankey_plot_{threshold}_w_data.html"))
    sub.to_csv(os.path.join(output_dir, f"ubs_flow_branches_{threshold}_w_data.csv"), index=False)
    #sub.to_csv(os.path.join(output_dir, f"ubs_flow_branches_{threshold}.csv"), index=False)



print("Done! Adjust node coloring or Sankey structure as you like.")
