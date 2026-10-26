# -*- coding: utf-8 -*-
"""
IF-SitePred Stage 1 (batch, sharded): ESM-IF1 embeddings -> binding residues.
=============================================================================

Faithful batch driver around IF-SitePred's `predict_residues.py`
(github.com/annacarbery/binding-sites): loads the ESM-IF1 model + the 40 LightGBM
classifiers ONCE, then loops over a shard of receptors, so the ~0.6 GB model is
not re-loaded per structure (the repo script reloads it every call). Emits
`<pred_dir>/<target>/predicted_residues.json` exactly as the upstream script does.

Runs in the `ifs_esm` env (fair-esm, torch, torch-geometric/scatter/cluster,
lightgbm, biotite<1). GPU is used automatically if available; --threads caps CPU
threads per worker for sharded CPU runs. This same script is the per-shard worker
for the Aurum GPU array job.

    python ifs_batch_stage1.py --repo_dir <binding-sites> \
        --receptor_list receptors.txt --shard 0 --nshards 4 --threads 6
"""
from __future__ import annotations

import os, sys, argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo_dir", required=True, help="IF-SitePred repo (has models/, input/, predictions/).")
    p.add_argument("--receptor_list", required=True, help="one afdb_filename or target stem per line.")
    p.add_argument("--input_dir", default=None, help="dir with <target>.pdb (default <repo>/input).")
    p.add_argument("--pred_dir", default=None, help="output dir (default <repo>/predictions).")
    p.add_argument("--models_dir", default=None, help="LightGBM dir (default <repo>/models).")
    p.add_argument("--afdb_dir", default=None,
                   help="if set, convert <afdb_dir>/<target>.cif.gz -> input/<target>.pdb when the pdb "
                        "is missing (gemmi; used on Aurum so no pre-conversion step is needed).")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--threads", type=int, default=0, help="cap CPU threads (0 = leave default).")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                   help="cpu is the reliable default: fair-esm 2.0 + torch 2.11 has a device "
                        "bug in ESM-IF1's GVP encoder on cuda; ESM-IF1 (142M) is fast on CPU.")
    return p.parse_args()

args = parse_args()
if args.threads > 0:                       # must be set before torch/numpy import
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = str(args.threads)

import json, pickle
from pathlib import Path
import numpy as np
import torch
import esm

REPO = Path(args.repo_dir)
INPUT = Path(args.input_dir) if args.input_dir else REPO / "input"
PRED = Path(args.pred_dir) if args.pred_dir else REPO / "predictions"
MODELS = Path(args.models_dir) if args.models_dir else REPO / "models"
AFDB = Path(args.afdb_dir) if args.afdb_dir else None
if args.threads > 0:
    torch.set_num_threads(args.threads)


def get_chains(pdb_path):
    return list({l[21:22] for l in open(pdb_path).read().splitlines() if l.startswith("ATOM")})

def get_index_coords(pdb_path, chain):
    coords_resi = {}
    for a in open(pdb_path).read().splitlines():
        if a.startswith("ATOM") and a[21:22] == chain:
            key = ",".join([a[30:38].strip(), a[38:46].strip(), a[46:55].strip()])
            coords_resi[key] = int(a[22:26].strip())
    return coords_resi

def load_models(n, models_dir):
    return [pickle.load(open(models_dir / f"lgbm_if1_{i}.pkl", "rb")) for i in range(n)]


def targets_from_list(path, shard, nshards):
    t = []
    for line in open(path).read().splitlines():
        line = line.strip()
        if not line:
            continue
        t.append(line[:-7] if line.endswith(".cif.gz") else line)
    return t[shard::nshards]


def predict_one(target, model, alphabet, models):
    pdb = INPUT / f"{target}.pdb"
    if not pdb.is_file():
        # On Aurum we convert on the fly (no pre-conversion step); locally the
        # orchestrator pre-converts, so this branch is skipped (AFDB is None).
        if AFDB is None:
            return "no_pdb"
        src = AFDB / f"{target}.cif.gz"
        if not src.is_file():
            return "no_pdb"
        import gemmi
        st = gemmi.read_structure(str(src)); st.setup_entities()
        INPUT.mkdir(parents=True, exist_ok=True)
        pdb.write_text(st.make_pdb_string())
    outdir = PRED / target
    if (outdir / "predicted_residues.json").is_file():
        return "skip"
    chains = sorted(get_chains(pdb))
    if not chains:
        return "no_chain"
    chain = chains[0]
    structure = esm.inverse_folding.util.load_structure(str(pdb), chain)
    coords_resi = get_index_coords(pdb, chain)
    coords, seq = esm.inverse_folding.util.extract_coords_from_structure(structure)
    if1 = esm.inverse_folding.util.get_encoder_output(model, alphabet, coords).detach().cpu().numpy()
    preds = [np.rint(m.predict(if1)) for m in models]
    nmod = len(models)
    inter = [1 if sum(preds[a][i] for a in range(nmod)) == nmod else 0 for i in range(len(preds[0]))]
    res_nums = []
    for i in range(len(coords)):
        key = ",".join("{:.3f}".format(x) for x in coords[i][0])
        if inter[i] == 1:
            if key in coords_resi:
                res_nums.append(str(coords_resi[key]))
    outdir.mkdir(parents=True, exist_ok=True)
    json.dump([res_nums, chain], open(outdir / "predicted_residues.json", "w"))
    return f"ok({len(res_nums)})"


def main():
    targets = targets_from_list(args.receptor_list, args.shard, args.nshards)
    if args.device == "cpu":
        dev = "cpu"
    elif args.device == "cuda":
        dev = "cuda"
    else:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[stage1 shard {args.shard}/{args.nshards}] {len(targets)} targets  device={dev}", flush=True)
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.eval()
    if dev == "cuda":
        model = model.cuda()
    models = load_models(40, MODELS)
    n_ok = n_skip = n_err = 0
    for i, t in enumerate(targets, 1):
        try:
            r = predict_one(t, model, alphabet, models)
            if r == "skip": n_skip += 1
            elif r.startswith("ok"): n_ok += 1
            else: n_err += 1
        except Exception as exc:
            n_err += 1
            print(f"[ERR] {t}: {exc}", flush=True)
        if i % 50 == 0:
            print(f"  shard {args.shard}: {i}/{len(targets)} (ok={n_ok} skip={n_skip} err={n_err})", flush=True)
    print(f"[stage1 shard {args.shard} DONE] ok={n_ok} skip={n_skip} err={n_err}", flush=True)


if __name__ == "__main__":
    main()
