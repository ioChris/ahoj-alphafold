# -*- coding: utf-8 -*-
"""
IF-SitePred Stage 2 (batch, sharded): predicted residues -> ranked 3-D centres.
===============================================================================

Faithful batch driver around IF-SitePred's `predict_centres.py`: for each target,
build a 1.5 A pseudo-atom grid around the predicted binding residues (PyMOL),
DBSCAN-cluster it, and write up to 3 sites ranked by cluster size as
`<pred_dir>/<target>/{site,centre}_rank_{1,2,3}.xyz`. Loops over a shard so PyMOL
is initialised once. Runs in the `ifs_pymol` env (pymol-open-source + scikit-learn
+ pandas). CPU-only; this is the Stage-2 worker for the Aurum array job too.

    python ifs_batch_stage2.py --repo_dir <binding-sites> \
        --receptor_list receptors.txt --shard 0 --nshards 4
"""
from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np
from sklearn.cluster import DBSCAN
from pymol import cmd


def write_xyz_grid(coords, xyz_path):
    mn = [np.min(coords[:, i]) - 10 for i in range(3)]
    mx = [np.max(coords[:, i]) + 10 for i in range(3)]
    axes = [np.arange(mn[i], mx[i], 1.5) for i in range(3)]
    with open(xyz_path, "w") as w:
        w.write(f"{len(axes[0])*len(axes[1])*len(axes[2])}\npoint\n")
        for a in axes[0]:
            for b in axes[1]:
                for c in axes[2]:
                    w.write(f"PS {round(a,3):.3f} {round(b,3):.3f} {round(c,3):.3f}\n")


def write_coords(coords, filename):
    with open(filename, "w") as w:
        w.write(f"{len(coords)}\npoint\n")
        for c in coords:
            w.write(f"PS {c[0]:.3f} {c[1]:.3f} {c[2]:.3f}\n")


def get_final_cloud(pdb, xyz_path, final_preds, chain):
    cmd.load(str(pdb), "complex")
    cmd.extract("hets", "complex and HETATM")
    cmd.delete("hets")
    cmd.extract("chA", f"complex and chain {chain}")
    coords = np.array(cmd.get_coords("chA"))
    write_xyz_grid(coords, xyz_path)
    cmd.load(str(xyz_path), "cloud1")
    cmd.extract("cloud3", "cloud1 within 3 of chA")
    cmd.extract("cloud_bubble", "cloud1 within 6 of chA")
    p = 0
    for p in range(len(final_preds)):
        cmd.select(f"sele_{final_preds[p]}", f"chA and resi {final_preds[p]}")
        if cmd.count_atoms(f"cloud_bubble within 4.5 of sele_{final_preds[p]}") > 0:
            cmd.create("cloud", f"cloud_bubble within 4.5 of sele_{final_preds[p]}")
            break
    for i in final_preds[p + 1:]:
        try:
            cmd.select(f"sele_{i}", f"chA and resi {i}")
            cmd.select(f"cloud1_{i}", f"cloud_bubble within 4.5 of sele_{i}")
            if cmd.count_atoms(f"cloud1_{i}") > 0:
                cmd.copy_to("cloud", f"cloud1_{i}")
        except Exception:
            raise
    coords = cmd.get_coords("cloud")
    coord_list = [",".join(str(n) for n in i) for i in coords]
    repeats = sorted(i for i in coord_list if coord_list.count(i) > 2)
    return np.array([[float(x) for x in c.split(",")] for c in repeats])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo_dir", required=True)
    p.add_argument("--receptor_list", required=True)
    p.add_argument("--input_dir", default=None)
    p.add_argument("--pred_dir", default=None)
    p.add_argument("--xyz_dir", default=None)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    REPO = Path(args.repo_dir)
    INPUT = Path(args.input_dir) if args.input_dir else REPO / "input"
    PRED = Path(args.pred_dir) if args.pred_dir else REPO / "predictions"
    XYZ = Path(args.xyz_dir) if args.xyz_dir else REPO / "xyz"
    XYZ.mkdir(parents=True, exist_ok=True)

    targets = []
    for line in open(args.receptor_list).read().splitlines():
        line = line.strip()
        if line:
            targets.append(line[:-7] if line.endswith(".cif.gz") else line)
    targets = targets[args.shard::args.nshards]
    print(f"[stage2 shard {args.shard}/{args.nshards}] {len(targets)} targets", flush=True)

    n_ok = n_skip = n_err = 0
    for idx, target in enumerate(targets, 1):
        outdir = PRED / target
        rj = outdir / "predicted_residues.json"
        if not rj.is_file():
            n_skip += 1; continue
        if (outdir / "centre_rank_1.xyz").is_file():
            n_skip += 1; continue
        pdb = INPUT / f"{target}.pdb"
        try:
            preds, chain = json.load(open(rj))
            if not preds:
                n_err += 1; continue
            cmd.reinitialize()
            final = get_final_cloud(pdb, XYZ / f"xyz_{target}.xyz", preds, chain)
            if final is None or len(final) == 0:
                n_err += 1; continue
            labels = DBSCAN(eps=1.7, min_samples=2).fit(final).labels_
            sizes = {s: list(labels).count(s) for s in set(labels) if int(s) != -1}
            for rank in range(min(3, len(sizes))):
                biggest = max(sizes, key=sizes.get)
                site = np.array([final[i] for i in range(len(final)) if labels[i] == biggest])
                write_coords(site, outdir / f"site_rank_{rank+1}.xyz")
                ctr = site.mean(axis=0)
                with open(outdir / f"centre_rank_{rank+1}.xyz", "w") as w:
                    w.write("1\npoint\n")
                    w.write(f"PS {ctr[0]:.3f} {ctr[1]:.3f} {ctr[2]:.3f}\n")
                del sizes[biggest]
            n_ok += 1
        except Exception as exc:
            n_err += 1
            print(f"[ERR] {target}: {exc}", flush=True)
        finally:
            xf = XYZ / f"xyz_{target}.xyz"          # the grid file is large; drop it
            if xf.is_file():
                try: xf.unlink()
                except OSError: pass
        if idx % 50 == 0:
            print(f"  shard {args.shard}: {idx}/{len(targets)} (ok={n_ok} skip={n_skip} err={n_err})", flush=True)
    print(f"[stage2 shard {args.shard} DONE] ok={n_ok} skip={n_skip} err={n_err}", flush=True)


if __name__ == "__main__":
    main()
