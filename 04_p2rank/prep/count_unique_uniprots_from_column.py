# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 11:32:10 2025

@author: Christos P. Feidakis
"""
""" Naive approach to count uniprots in the column site_uniprots of the CSV file (works for Monomers only!!!) """
import pandas as pd
from pathlib import Path

# --- locate repo root so `common` is importable (works standalone / in Spyder) ---
import sys as _sys
from pathlib import Path as _Path
for _root in _Path(__file__).resolve().parents:
    if (_root / "common" / "config.py").is_file():
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))
        break
from common import config as cfg

# Reuse or set this to your file (legacy ST_LJ centers; adjust sub-path as needed):
INPUT_CSV = str(cfg.WORK_DIR / "afdb_ubs_sets" / "ubs_centers" / "LJ" / "sub_df_ST_LJ_ubs_filtered_annotated_with_centers_nUNP0.csv")

def count_unique_uniprots(input_csv: str) -> None:
    input_path = Path(input_csv)
    df = pd.read_csv(input_path, usecols=["site_uniprots"])

    n_unique = df["site_uniprots"].nunique()
    print(f"File: {input_path}")
    print(f"Number of rows: {len(df)}")
    print(f"Number of unique UniProt IDs in 'site_uniprots': {n_unique}")

    # If you also want to see them, uncomment:
    # uniques = df["site_uniprots"].unique()
    # print("Unique UniProt IDs:")
    # print(uniques)

count_unique_uniprots(INPUT_CSV)
