# -*- coding: utf-8 -*-
"""fpocket's version and IF-SitePred's commit in the cluster images (Methods, Binding-site prediction).

The Methods name fpocket 4.2.3 (conda-forge build) and IF-SitePred commit fb9d778 (25 March 2024); this script
derives both. Neither image definition pins what it installs: 06_lbs_generality/hpc/fpocket.def runs `mamba install -c
conda-forge fpocket` and hpc/ifsitepred.def runs `git clone --depth 1 https://github.com/annacarbery/binding-sites`.
The build logs that would print `fpocket -ver` sit on the HPC cluster ($HOME/ahoj_lbs/code/hpc/logs/fpbuild-*.out) and
the clone's HEAD inside the image. Both can be settled without logging in, from two public histories against
the build date:

  * conda-forge's upload history of the fpocket package (anaconda.org API): which version an unpinned install
    resolved to on the build date;
  * the binding-sites repository's commit history (GitHub API): which commit a depth-1 clone of `main` received
    on the build date, and whether any commit followed.

The build date is bracketed by the public repo's own history (the hpc/ definitions were committed on 2026-07-09,
the full runs finished 2026-07-09, manifests `lbs_<tool>_manifest.json`) and by the local WSL rehearsal of the
same recipe: the `lbs` micromamba env was created on 2026-07-08 22:05 and resolved fpocket 4.2.3 h8d0616a_1
(its conda-meta/history), and the local clone made on 2026-07-09 01:24 is at fb9d7787 (`git log -1`).

Both public histories are cached in <manuscript_analyses_dir>/predictor_versions_cache.json, so the numbers
reproduce offline from the cache; without it (or with --refresh) they are fetched and the cache is written.

Usage
    python predictor_versions.py [--refresh] [--build-date 2026-07-09] [--local-wsl]
      --local-wsl   also read the WSL rehearsal's env history and clone HEAD (needs wsl.exe on this PC)
"""
import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import ma_paths as P

HERE = Path(__file__).resolve().parent
CACHE = Path(P.out("predictor_versions_cache.json"))
CONDA_URL = "https://api.anaconda.org/package/conda-forge/fpocket"
GH_URL = "https://api.github.com/repos/annacarbery/binding-sites/commits?sha=main&per_page=100"
GH_REPO_URL = "https://api.github.com/repos/annacarbery/binding-sites"
DEFAULT_BUILD_DATE = "2026-07-09"   # hpc/*.def committed; full runs finished (lbs_<tool>_manifest.json, 2026-07-09T08:52Z and 09:54Z)
WSL_SCRIPT = r"""
echo "lbs_env_created=$(head -1 "$HOME/micromamba/envs/lbs/conda-meta/history" | tr -d '=<> ')"
echo "lbs_env_fpocket=$(grep -o 'fpocket-[0-9.]*-[a-z0-9_]*' "$HOME/micromamba/envs/lbs/conda-meta/history" | head -1)"
echo "clone_head=$(git -C "$HOME/lbs_ifsitepred/binding-sites" log -1 --format='%H %cI' 2>/dev/null)"
echo "clone_dir_mtime=$(stat -c %y "$HOME/lbs_ifsitepred/binding-sites/.git" 2>/dev/null | cut -c1-19)"
"""


def fetch_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "ahoj-af-verify/1.0", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def load(refresh):
    if CACHE.exists() and not refresh:
        return json.loads(CACHE.read_text(encoding="utf-8")), "cache"
    conda = fetch_json(CONDA_URL)
    files = [{"version": f["version"], "build": f.get("attrs", {}).get("build", ""),
              "subdir": f.get("attrs", {}).get("subdir", ""), "upload_time": f.get("upload_time", "")[:19]}
             for f in conda.get("files", [])]
    commits = [{"sha": c["sha"], "date": c["commit"]["committer"]["date"],
                "message": c["commit"]["message"].split("\n")[0]} for c in fetch_json(GH_URL)]
    repo = fetch_json(GH_REPO_URL)
    data = {"fetched": __import__("datetime").date.today().isoformat(),
            "conda_forge_fpocket": {"latest_version": conda.get("latest_version"), "files": files},
            "binding_sites": {"default_branch": repo.get("default_branch"), "pushed_at": repo.get("pushed_at"),
                              "commits_main": commits}}
    CACHE.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return data, "fetched"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--build-date", default=DEFAULT_BUILD_DATE)
    ap.add_argument("--local-wsl", action="store_true")
    a = ap.parse_args()
    data, how = load(a.refresh)
    bd = a.build_date
    print(f"=== sources: {how} ({CACHE.name}, fetched {data['fetched']}); build date taken as {bd} ===")

    files = [f for f in data["conda_forge_fpocket"]["files"] if f["subdir"] == "linux-64"]
    before = sorted((f for f in files if f["upload_time"][:10] <= bd), key=lambda f: f["upload_time"])
    after = sorted((f for f in files if f["upload_time"][:10] > bd), key=lambda f: f["upload_time"])
    newest = before[-1]
    print("\n=== fpocket on conda-forge (linux-64) ===")
    print(f"  builds uploaded on or before {bd}: {len(before)}; newest: {newest['version']} {newest['build']} "
          f"({newest['upload_time'][:10]})")
    print(f"  builds uploaded after {bd}: " + (", ".join(f"{f['version']} {f['build']} ({f['upload_time'][:10]})" for f in after) or "none"))
    versions_after = {f["version"] for f in after}
    print(f"  -> an unpinned `mamba install -c conda-forge fpocket` on {bd} resolved to fpocket {newest['version']}"
          + ("" if versions_after <= {newest["version"]} else f"; NOTE a newer version appeared later: {sorted(versions_after)}"))
    prev = [f for f in before if f["version"] != newest["version"]]
    if prev:
        print(f"  previous version: {prev[-1]['version']} ({prev[-1]['upload_time'][:10]})")

    commits = data["binding_sites"]["commits_main"]
    print("\n=== annacarbery/binding-sites on GitHub ===")
    print(f"  default branch {data['binding_sites']['default_branch']}; last push {data['binding_sites']['pushed_at']}; "
          f"{len(commits)} commits on main")
    head_before = [c for c in commits if c["date"][:10] <= bd][0]
    later = [c for c in commits if c["date"][:10] > bd]
    print(f"  HEAD of main on {bd}: {head_before['sha'][:7]} ({head_before['date'][:10]}, \"{head_before['message']}\")")
    print(f"  commits after {bd}: {len(later)}")
    print(f"  -> `git clone --depth 1` on {bd} received {head_before['sha'][:7]}"
          + ("" if not later else " (later commits exist; the image's HEAD is this one only if built before them)"))
    models = [c for c in commits if "model" in c["message"].lower()]
    if models:
        print(f"  the LightGBM models (models/lgbm_if1_*.pkl) last changed in {models[0]['sha'][:7]} ({models[0]['date'][:10]}, \"{models[0]['message']}\")")

    if a.local_wsl:
        print("\n=== local WSL rehearsal of the same recipe ===")
        try:
            out = subprocess.run(["wsl.exe", "-e", "bash", "-c", WSL_SCRIPT], capture_output=True, text=True, timeout=120).stdout
            print("  " + out.strip().replace("\n", "\n  "))
        except Exception as e:  # noqa: BLE001
            print(f"  not available: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
