# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 13:52:53 2025

@author: Christos P. Feidakis
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"

import os
#import re


# Base color hex (apo=blue, holo=red, tie=gray)
APO_BASE = "#2563eb"
HOLO_BASE = "#b91c1c"
TIE_BASE = "#bbbbbb"


# In your coloring loop:
LEVEL_LABELS = ["Density", "Separation", "Probability", "Class"]


# Paler colors for earlier levels (RGBA with alpha)
def color_with_alpha(hexcolor, alpha):
    """Convert hex color to rgba string with given alpha (0-1)"""
    h = hexcolor.lstrip('#')
    return f'rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{alpha})'

# Transparency schedule (leftmost = most transparent)
ALPHAS = [0.2, 0.4, 0.6, 1.0]  # Density, Separation, Probability, Class

# ---------------------------------------------------

def blue_gradient0(n_high, N):
    base = np.array([178, 203, 229])    # #b2cbe5 light
    deep = np.array([27, 54, 102])      # #1b3666 deep
    shade = n_high / N
    rgb = (base * (1-shade) + deep * shade).astype(int)
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"

def red_gradient0(n_high, N):
    base = np.array([246, 177, 177])    # #f6b1b1 light
    deep = np.array([155, 28, 28])      # #9b1c1c deep
    shade = n_high / N
    rgb = (base * (1-shade) + deep * shade).astype(int)
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"

def gray_gradient0(n_high, N):
    minval = 170
    maxval = 245
    val = int(maxval - (n_high / N) * (maxval - minval))
    return f"rgb({val},{val},{val})"

def blue_gradient(n_high, N):
    if N == 0:
        return "rgb(178,203,229)"  # lightest blue as fallback
    base = np.array([178, 203, 229])
    deep = np.array([27, 54, 102])
    shade = n_high / N
    rgb = (base * (1-shade) + deep * shade).astype(int)
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"

def red_gradient(n_high, N):
    if N == 0:
        return "rgb(246,177,177)"  # lightest red as fallback
    base = np.array([246, 177, 177])
    deep = np.array([155, 28, 28])
    shade = n_high / N
    rgb = (base * (1-shade) + deep * shade).astype(int)
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"

def gray_gradient(n_high, N):
    if N == 0:
        return "rgb(238,238,238)"
    base = np.array([238, 238, 238])
    deep = np.array([136, 136, 136])
    shade = n_high / N
    rgb = (base * (1-shade) + deep * shade).astype(int)
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"



def extract_categories0(node_label, levels):
    # e.g., "Separation: High | Density: Low"
    fields = node_label.split(" | ")
    cats = []
    for i, lvl in enumerate(fields):
        # lvl is e.g. "Separation: High"
        part = lvl.split(":")[-1].strip()
        cats.append(part)
    return cats

def extract_categories(node_label):
    # E.g., "Probability: High | Separation: Moderate | Density: Low"
    return [part.split(": ")[-1] for part in node_label.split(" | ")]

def category_weight(cat):
    # Accepts "Low", "Moderate", "High", "Ambiguous"
    if cat in ["High"]: return 1.0
    if cat in ["Moderate"]: return 0.5
    return 0.0  # Low, Ambiguous, or unknown

# --- Helper functions for verbal binning ---
def classify_density(x):
    try: x = float(x)
    except: return "No data"
    if pd.isna(x): return "No data"
    if x <= 0.3: return "Low"
    elif x <= 0.7: return "Moderate"
    else: return "High"

def classify_separation(overlap):
    try: overlap = float(overlap)
    except: return "No data"
    if pd.isna(overlap): return "No data"
    sep = 1 - overlap
    return classify_density(sep)

def classify_probability(p):
    try: p = float(p)
    except: return "No data"
    if pd.isna(p): return "No data"
    if p <= 0.60: return "Ambiguous"
    elif p <= 0.70: return "Moderate"
    else: return "High"

def get_max_prob(row):
    try:
        pa = float(row.get('avg_P(apo)_mean_pocket', np.nan))
        ph = float(row.get('avg_P(holo)_mean_pocket', np.nan))
        if pd.isna(pa) and pd.isna(ph):
            return np.nan
        elif pd.isna(pa): return ph
        elif pd.isna(ph): return pa
        else: return max(pa, ph)
    except Exception:
        return np.nan

# Prepare dataframe with only rows with data
def has_prob_data(row):
    try:
        return not (pd.isna(row['avg_P(apo)_mean_pocket']) or pd.isna(row['avg_P(holo)_mean_pocket']))
    except:
        return False




def gray_by_weight(weight):
    # weight: 0 (lightest) to 1 (darkest)
    val = int(240 - weight * 120)  # 240 (light gray) to 120 (deep gray)
    return f"rgb({val},{val},{val})"




# ------------------ CONFIGURATION ------------------
# --- locate repo root so `common` is importable (works standalone / in Spyder) ---
import sys as _sys
from pathlib import Path as _Path
for _root in _Path(__file__).resolve().parents:
    if (_root / "common" / "config.py").is_file():
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))
        break
from common import config as cfg

# Path to your input CSV (legacy v1.2 UBS05+LJ subset; adjust as needed)
_legacy_dir = cfg.UBS_AGG_DIR / "06082025_122505_v1.2_Noneunplimit"
csv_path = str(_legacy_dir / "ubs_UBS05_and_LJ_subset.csv")
output_dir = str(_legacy_dir / "sankey_results")
os.makedirs(output_dir, exist_ok=True)

# --- Load data ---
df = pd.read_csv(csv_path, low_memory=False)

# Add this normalization if not already present:
def normalize_threshold(thresh):
    if isinstance(thresh, str) and "UBS_LJ" in thresh:
        return "UBS_LJ"
    return thresh
df['threshold'] = df['threshold'].apply(normalize_threshold)

# Loop over each threshold value
for threshold in df['threshold'].unique():
    print(f"Processing threshold: {threshold}")

    # --- Prepare dataframe with only rows with data and current threshold ---
    df_thr = df[df['threshold'] == threshold]

    flow_rows = []
    for idx, row in df_thr.iterrows():
        if not has_prob_data(row):
            continue
        dens = classify_density(row['avg_normalized_density_combined_mean_pocket'])
        sep = classify_separation(row['pair_overlap_mean_pocket'])
        prob = classify_probability(get_max_prob(row))
        cls = row['highest_p_mean_pocket'] if isinstance(row['highest_p_mean_pocket'], str) else "No data"
        # Only include valid steps
        if "No data" in (dens, sep, prob, cls): continue
        flow_rows.append({
            "site_recipe": row['site_recipe'],
            "Density": dens,
            "Separation": sep,
            "Probability": prob,
            "Class": cls,
        })

    flow_df = pd.DataFrame(flow_rows)
    # Remove TIE Classifications
    flow_df = flow_df[flow_df["Class"] != "tie"].copy()

    print(f"UBSs with data for threshold {threshold}: {len(flow_df)}")
    if flow_df.empty:
        print(f"No data for threshold {threshold}, skipping.")
        continue

    # --- Composite node and Sankey setup as before, but with outputs per threshold ---
    flow_df["Density_node"] = flow_df["Density"]
    flow_df["Separation_node"] = flow_df["Separation"] + " | " + flow_df["Density"]
    #flow_df["Probability_node"] = flow_df["Probability"]
    flow_df["Probability_node"] = flow_df["Probability"] + " | " + flow_df["Separation"] + " | " + flow_df["Density"]
    flow_df["Class_node"] = flow_df["Class"]  # Only 'apo', 'holo', 'tie'
    #flow_df["Class_node"] = flow_df["Class"] + " | " + flow_df["Probability"] + " | " + flow_df["Separation"] + " | " + flow_df["Density"]

    nodes = []
    node_labels = []

    # ---- For Density, Separation, Probability, keep original unique order ----
    for col in ["Density_node", "Separation_node"]:#, "Probability_node"]:
        vals = list(flow_df[col].unique())
        nodes.extend(vals)
        node_labels.extend(vals)

    # Only three probability bins at this level, in preferred order
    #for prob in ["High", "Moderate", "Ambiguous"]:
    #    if prob in flow_df["Probability_node"].unique():
    #        nodes.append(prob)
    #        node_labels.append(prob)

    #"""
    # Add specific ordering for Probability node
    # Define your preferred order for affinity
    prob_affinity_order = {'High': 0, 'Moderate': 1, 'Ambiguous': 2}
    # Helper to get main annotation and total path intensity
    def get_prob_anno_and_weight(node_label):
        cats = extract_categories(node_label)
        prob = cats[0]
        n_high = sum(category_weight(cat) for cat in cats)
        return prob, n_high
    # Gather all unique Probability nodes, extract their info
    prob_nodes = []
    for node_label in flow_df["Probability_node"].unique():
        prob, n_high = get_prob_anno_and_weight(node_label)
        prob_nodes.append((prob_affinity_order.get(prob, 99), -n_high, node_label))
    # Sort: High first, then Moderate, then Ambiguous, then by intensity (deepest first)
    prob_nodes_sorted = [x[2] for x in sorted(prob_nodes)]
    # For Probability nodes, use affinity-sorted order
    nodes.extend(prob_nodes_sorted)
    node_labels.extend(prob_nodes_sorted)
    #"""

    # --- For final Class layer, use only three anchors: apo, holo, tie ---
    for c in ['apo', 'holo']:#, 'tie']:
        nodes.append(c)
        node_labels.append(c)
    
    node_to_idx = {label: i for i, label in enumerate(node_labels)}

    """
    class_nodes = []
    for c in flow_df["Class_node"].unique():
        cats = extract_categories(c)
        cls = cats[0]
        path = cats[1:]
        n_high = sum(category_weight(cat) for cat in path)
        class_nodes.append((cls, n_high, c))

    sorted_class_nodes = []
    for cat in ['apo', 'holo', 'tie']:
        nodes_cat = [c for c in class_nodes if c[0] == cat]
        nodes_cat_sorted = sorted(nodes_cat, key=lambda x: -x[1])  # highest weight first
        sorted_class_nodes.extend([c[2] for c in nodes_cat_sorted])

    nodes.extend(sorted_class_nodes)
    node_labels.extend(sorted_class_nodes)
    node_to_idx = {label: i for i, label in enumerate(node_labels)}
    """
    #for col in ["Density_node", "Separation_node", "Probability_node", "Class_node"]:
    #    vals = list(flow_df[col].unique())
    #    nodes.extend(vals)
    #    node_labels.extend(vals)
    #node_to_idx = {label: i for i, label in enumerate(node_labels)}

    sources, targets, values, link_classes, link_probs, link_levels = [], [], [], [], [], []
    link_colors = []
    # Density → Separation
    for (d, s), g in flow_df.groupby(["Density_node", "Separation_node"]):
        sources.append(node_to_idx[d])
        targets.append(node_to_idx[s])
        values.append(len(g))
        dom_cls = g["Class"].value_counts().idxmax()
        dom_prob = g["Probability"].value_counts().idxmax()
        link_classes.append(dom_cls)
        link_probs.append(dom_prob)
        link_levels.append("Density")
    # Separation → Probability
    for (s, p), g in flow_df.groupby(["Separation_node", "Probability_node"]):
        sources.append(node_to_idx[s])
        targets.append(node_to_idx[p])
        values.append(len(g))
        dom_cls = g["Class"].value_counts().idxmax()
        dom_prob = g["Probability"].value_counts().idxmax()
        link_classes.append(dom_cls)
        link_probs.append(dom_prob)
        link_levels.append("Separation")
    # Probability → Class
    # Probability → Class (now grouped only by Class)

    #"""
    for (p, c), g in flow_df.groupby(["Probability_node", "Class"]):
        sources.append(node_to_idx[p])
        targets.append(node_to_idx[c])
        values.append(len(g))
        # For coloring, calculate the cumulative ancestor weight
        cats = extract_categories(p)  # probability_node carries all ancestors
        N = len(cats)
        n_high = sum(category_weight(cat) for cat in cats)
        link_classes.append(c)  # c is just 'apo', 'holo', or 'tie'
        link_probs.append(cats[0])  # probability annotation
        link_levels.append("Probability")
        # You may want to keep other tracking info for color gradients
    #"""
    # 3. Probability → Class (with sorted outgoing streams)
    """
    for p, group in flow_df.groupby("Probability_node"):
        # For this source node, find all target Class nodes and their weights
        class_links = []
        for c, g in group.groupby("Class_node"):
            cats = extract_categories(c)
            cls = cats[0]
            path = cats[1:]
            n_high = sum(category_weight(cat) for cat in path)
            class_links.append((cls, n_high, c, len(g)))  # (class, path weight, node label, count)
        # Sort: first by class group (apo, holo, tie), then by path weight descending
        class_links_sorted = sorted(class_links, key=lambda x: (['apo','holo','tie'].index(x[0]), -x[1]))
        for cls, n_high, c, count in class_links_sorted:
            sources.append(node_to_idx[p])
            targets.append(node_to_idx[c])
            values.append(count)
            # (append to any other tracking lists you need)
            dom_cls = cls
            dom_prob = extract_categories(c)[1]  # probability annotation for this stream
            link_classes.append(dom_cls)
            link_probs.append(dom_prob)
            link_levels.append("Probability")
    """
    """
    for (p, c), g in flow_df.groupby(["Probability_node", "Class_node"]):
        sources.append(node_to_idx[p])
        targets.append(node_to_idx[c])
        values.append(len(g))
        dom_cls = g["Class"].value_counts().idxmax()
        dom_prob = g["Probability"].value_counts().idxmax()
        link_classes.append(dom_cls)
        link_probs.append(dom_prob)
        link_levels.append("Probability")
    """

    # --- Coloring logic ---

    link_colors = []
    class_labels = {"apo", "holo"}#, "tie"}

    for src, tgt, val in zip(sources, targets, values):
        from_label = node_labels[src]
        to_label = node_labels[tgt]
        # For Class nodes, use ancestor path from the Probability node (source)
        if to_label in class_labels:
            """
            # from_label is now just "High", "Moderate", or "Ambiguous"
            n_high = category_weight(from_label)
            N = 1
            if to_label == "apo":
                color = blue_gradient(n_high, N)
            elif to_label == "holo":
                color = red_gradient(n_high, N)
            else:
                color = gray_gradient(n_high, N)
            """
            cats = extract_categories(from_label)  # ancestors from Probability node
            N = len(cats)
            n_high = sum(category_weight(cat) for cat in cats)
            if to_label == "apo":
                color = blue_gradient(n_high, N)
            elif to_label == "holo":
                color = red_gradient(n_high, N)
            else:
                color = gray_gradient(n_high, N)
            #"""
        else:
            cats = extract_categories(to_label)
            N = len(cats)
            n_high = sum(category_weight(cat) for cat in cats)
            color = gray_gradient(n_high, N)
        """
        to_label = node_labels[tgt]
        cats = extract_categories(to_label)
        # The class annotation is always the first category
        cls = cats[0]
        # If the label starts with a class value, it's a Class node!
        if cls in class_labels:
            # Only use the ancestors: Probability, Separation, Density
            cats_path = cats[1:]  # skip the class itself
            N = len(cats_path)
            n_high = sum(category_weight(cat) for cat in cats_path)
            # Now color by class using the compounded intensity
            if cls == "apo":
                color = blue_gradient(n_high, N)
            elif cls == "holo":
                color = red_gradient(n_high, N)
            else:
                color = gray_gradient(n_high, N)
        else:
            # For all non-class links, use all available categories
            N = len(cats)
            n_high = sum(category_weight(cat) for cat in cats)
            color = gray_gradient(n_high, N)
        """


        link_colors.append(color)



    # --- Sankey Plot ---
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=20,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node_labels,
            color="#f7fafc",
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
        )
    )])

    fig.update_layout(
        title_text=f"UBS Pocket Classification (Composite Path Sankey) — {threshold}",
        font_size=12,
        width=1600,
        height=900,
        margin=dict(l=0, r=0, t=40, b=0)
    )

    # Outputs are now per threshold:
    fig.show()
    #fig.write_html(os.path.join(output_dir, f"composite_sankey_{threshold}.html")) # Works
    #fig.write_html(os.path.join(output_dir, f"composite_sankey_{threshold}.svg"))
    #fig.write_image(os.path.join(output_dir, f"composite_sankey_{threshold}.png"), width=1600, height=900)
    flow_df.to_csv(os.path.join(output_dir, f"composite_sankey_{threshold}.csv"), index=False)

print("Done! Sankey diagrams and data exported per threshold.")

