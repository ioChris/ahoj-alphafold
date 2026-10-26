#!/usr/bin/env python3
"""
Resumable, batch-granular launcher for the full_run_3 apo/holo reclassification.

Wraps ``batch_process_database.py``: it runs ONE AHoJ-DB batch directory per pipeline
invocation (via the config's ``batch_selection``), records completion with a per-batch
sentinel inside the mirror, and re-reads an editable control table before every batch so
batches can be skipped (claimed by another machine) while the run is live.

Guarantees
----------
* **Resumable.** A kill leaves in-progress batches without a sentinel, so a restart re-runs
  only unfinished, non-skipped batches. Per-entry outputs are idempotent overwrites, so
  re-running a partially done batch is safe.
* **Runtime skip.** Edit ``batches.tsv`` (flip a batch's action ``run`` -> ``skip``) at any
  time; the launcher picks it up before it reaches that batch. The launcher never writes
  that file and you never write sentinels, so there is no write-write race.
* **Mergeable by copying directories.** Each machine writes its own local mirror. Because
  batches are disjoint, ``rsync -a`` / ``cp -a`` one machine's mirror over another's merges
  cleanly, and the ``.batch_complete`` sentinels travel with the batch dirs. Summaries are
  written under ``<summary_dir>/<host>/`` so they never collide across machines; ``concat``
  stitches them.

Subcommands
-----------
  init     write the editable control table (batches.tsv) if it does not exist
  run      process pending, non-skipped batches in order  (the main loop)
  status   print a live table: done / skipped / pending + ETA
  concat   merge every per-batch summary CSV into single MERGED csvs

Examples
--------
  python run_full_batch_launcher.py init   --config .local_tests/run_config_full_run_3.local.json
  python run_full_batch_launcher.py run    --config .local_tests/run_config_full_run_3.local.json
  python run_full_batch_launcher.py status --config .local_tests/run_config_full_run_3.local.json
  python run_full_batch_launcher.py concat --config .local_tests/run_config_full_run_3.local.json
"""
import argparse
import glob
import json
import os
import socket
import subprocess
import sys
import time

SENTINEL_NAME = ".batch_complete"
FUNC = "do_kde_plots_and_classification"  # summary-subdir suffix the pipeline uses


# --------------------------------------------------------------------------- helpers
def load_config(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def host():
    return socket.gethostname()


def natural_key(b):
    """Sort '000'..'560' numerically, falling back to string for odd names."""
    return (0, int(b)) if b.isdigit() else (1, b)


def all_batches(root_data_dir):
    """Every immediate sub-directory of the input root, in natural order."""
    return sorted(
        (d for d in os.listdir(root_data_dir)
         if os.path.isdir(os.path.join(root_data_dir, d))),
        key=natural_key,
    )


def control_dir(cfg):
    """Per-machine launcher bookkeeping lives beside the summary output."""
    d = os.path.join(cfg["summary_dir"], "_control")
    os.makedirs(d, exist_ok=True)
    return d


def host_summary_dir(cfg):
    """Host-namespaced summary root so run_* dirs never collide across machines."""
    return os.path.join(cfg["summary_dir"], host())


def table_path(cfg):
    return os.path.join(control_dir(cfg), "batches.tsv")


def sentinel_path(cfg, batch):
    return os.path.join(cfg["root_mirror_dir"], batch, SENTINEL_NAME)


def is_done(cfg, batch):
    return os.path.isfile(sentinel_path(cfg, batch))


def read_skips(cfg):
    """Return the set of batches marked 'skip' in the control table (read fresh)."""
    path = table_path(cfg)
    skips = set()
    if not os.path.isfile(path):
        return skips
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            batch = parts[0]
            action = parts[1].lower() if len(parts) > 1 else "run"
            if action == "skip":
                skips.add(batch)
    return skips


def write_sentinel(cfg, batch, payload):
    """Atomic per-batch completion marker inside the mirror (travels on copy-merge)."""
    dest = sentinel_path(cfg, batch)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    os.replace(tmp, dest)


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def latest_manifest(cfg):
    """Newest run_manifest.json under this host's summary dir (just-finished batch)."""
    runs = sorted(glob.glob(os.path.join(host_summary_dir(cfg), f"run_*_{FUNC}", "run_manifest.json")))
    if not runs:
        return {}
    try:
        with open(runs[-1], "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def per_batch_config(cfg, base_config_path, batch):
    """Write a one-batch config: base + batch_selection + host-namespaced summary_dir."""
    gen_dir = os.path.join(control_dir(cfg), "configs_gen")
    os.makedirs(gen_dir, exist_ok=True)
    one = dict(cfg)
    one["batch_selection"] = [batch]
    one["summary_dir"] = host_summary_dir(cfg)
    path = os.path.join(gen_dir, f"{batch}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(one, fh, indent=2)
    return path


# --------------------------------------------------------------------------- commands
def cmd_init(args, cfg):
    path = table_path(cfg)
    if os.path.isfile(path) and not args.force:
        print(f"control table already exists: {path}  (use --force to regenerate)")
        return
    batches = all_batches(cfg["root_data_dir"])
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(
            "# full_run_3 batch control table - edit the 'action' column while the run is live.\n"
            "#   action = run   -> process this batch on this machine (default)\n"
            "#   action = skip  -> do NOT process here (claimed by another machine / not needed)\n"
            "# The launcher re-reads this file before every batch. Completion is tracked by a\n"
            "# .batch_complete sentinel inside each batch's mirror dir - NOT here - so editing this\n"
            "# file never conflicts with the launcher and a restart resumes only unfinished batches.\n"
            "#\n"
            "# batch\taction\n"
        )
        for b in batches:
            fh.write(f"{b}\trun\n")
    print(f"wrote control table for {len(batches)} batches: {path}")


def cmd_run(args, cfg):
    base = args.config
    batches = all_batches(cfg["root_data_dir"])
    if getattr(args, "reverse", False):
        batches = list(reversed(batches))            # other machine: last batch -> first
    if not os.path.isfile(table_path(cfg)):
        cmd_init(args, cfg)
    logs = os.path.join(control_dir(cfg), "logs")
    os.makedirs(logs, exist_ok=True)

    done_now = failed = skipped = 0
    for batch in batches:
        skips = read_skips(cfg)                       # fresh each iteration -> live edits
        if batch in skips:
            skipped += 1
            continue
        if is_done(cfg, batch):
            continue

        cfg_path = per_batch_config(cfg, base, batch)
        log_path = os.path.join(logs, f"{batch}.log")
        remaining = sum(1 for b in batches
                        if b not in skips and not is_done(cfg, b))
        print(f"[{time.strftime('%H:%M:%S')}] batch {batch}  "
              f"(~{remaining} left on this host)  -> {log_path}", flush=True)

        t0 = time.time()
        with open(log_path, "w", encoding="utf-8") as lf:
            rc = subprocess.call(
                [sys.executable, "batch_process_database.py", "--config", cfg_path],
                stdout=lf, stderr=subprocess.STDOUT,
            )
        elapsed = time.time() - t0

        if rc == 0:
            man = latest_manifest(cfg)
            write_sentinel(cfg, batch, {
                "batch": batch,
                "host": host(),
                "git_commit": man.get("git_commit") or git_commit(),
                "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "elapsed_seconds": round(elapsed, 1),
                "counts": (man.get("results", {}) or {}).get("counts", man.get("counts")),
            })
            done_now += 1
            print(f"           done in {elapsed/60:.1f} min", flush=True)
        else:
            failed += 1
            print(f"           FAILED rc={rc}  (see {log_path}) - no sentinel written", flush=True)
            if args.stop_on_error:
                print("stopping: --stop-on-error set", flush=True)
                break

    print(f"\nrun finished: completed={done_now}  failed={failed}  skipped={skipped}")
    if failed:
        print("re-run the same command to retry failed batches (they have no sentinel).")


def cmd_status(args, cfg):
    batches = all_batches(cfg["root_data_dir"])
    skips = read_skips(cfg)
    done = [b for b in batches if is_done(cfg, b)]
    skipped = [b for b in batches if b in skips and b not in done]
    pending = [b for b in batches if b not in skips and b not in done]

    # ETA from mean elapsed of completed batches on this host.
    elapsed = []
    for b in done:
        try:
            with open(sentinel_path(cfg, b), "r", encoding="utf-8") as fh:
                s = json.load(fh)
            if s.get("host") == host() and s.get("elapsed_seconds"):
                elapsed.append(s["elapsed_seconds"])
        except Exception:
            pass
    mean = sum(elapsed) / len(elapsed) if elapsed else None

    reverse = getattr(args, "reverse", False)
    print(f"host {host()}  |  total {len(batches)}  "
          f"done {len(done)}  skipped {len(skipped)}  pending {len(pending)}"
          f"{'  [reverse]' if reverse else ''}")
    if mean:
        eta_h = mean * len(pending) / 3600
        print(f"mean {mean/60:.1f} min/batch over {len(elapsed)} done  ->  "
              f"ETA ~{eta_h:.1f} h for {len(pending)} pending")
    if done:
        # Completed frontier - forward machine advances its max, reverse machine lowers its min.
        # When the two ends' frontiers cross (fwd max >= rev min), stop both and merge.
        dsorted = sorted(done, key=natural_key)
        print(f"done range: {dsorted[0]}..{dsorted[-1]}  "
              f"(frontier: {'lowest ' + dsorted[0] if reverse else 'highest ' + dsorted[-1]})")
    if pending:
        order = list(reversed(pending)) if reverse else pending
        head = " ".join(order[:12])
        print(f"next pending: {head}{' ...' if len(order) > 12 else ''}")
    if skipped:
        print(f"skipped ({len(skipped)}): {skipped[0]}..{skipped[-1]}")


def _zip_beside(path):
    """Reproduce the pipeline's zip_file(): write <path>.zip (DEFLATE, arcname=basename)."""
    import zipfile
    zpath = path + ".zip"
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(path, arcname=os.path.basename(path))
    return zpath


def per_batch_run_dirs(cfg):
    """Every per-batch summary run dir across all hosts, excluding a merged (top-level) run dir.

    Per-batch runs live at <summary_dir>/<host>/run_*_FUNC/ (depth 2); a merged output written by
    this command sits directly under <summary_dir> (depth 1), so it is filtered out and never
    re-merged into itself.
    """
    base = os.path.abspath(cfg["summary_dir"])
    dirs = []
    for man in glob.glob(os.path.join(base, "**", f"run_*_{FUNC}", "run_manifest.json"),
                         recursive=True):
        run_dir = os.path.dirname(man)
        if os.path.dirname(run_dir) == base:          # merged output -> skip
            continue
        dirs.append(run_dir)
    return sorted(dirs)


def _merge_job_logs(run_dirs, out_dir):
    """Concatenate the four *_jobs.txt category files (dedup on job name); return counts."""
    counts = {}
    for cat in ("Success", "File Not Found", "Error", "Discarded"):
        fname = f"{cat}_jobs.txt"
        seen, lines = set(), []
        for d in run_dirs:
            p = os.path.join(d, fname)
            if not os.path.isfile(p):
                continue
            with open(p, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    key = line.split(":", 1)[0] if cat == "Discarded" else line
                    if key in seen:
                        continue
                    seen.add(key)
                    lines.append(line)
        with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as fh:
            fh.write("".join(l + "\n" for l in lines))
        counts[cat] = len(lines)
    return counts


def _write_merged_parameters(run_dirs, out_dir, cfg):
    """parameters.json as the monolithic run would write it: batch_selection cleared, base summary_dir."""
    params = None
    for d in run_dirs:
        p = os.path.join(d, "parameters.json")
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as fh:
                params = json.load(fh)
            break
    if params is None:
        params = dict(cfg)
    params["batch_selection"] = []
    params["summary_dir"] = cfg["summary_dir"]
    with open(os.path.join(out_dir, "parameters.json"), "w", encoding="utf-8") as fh:
        json.dump(params, fh, indent=1)


def _write_merged_manifest(run_dirs, out_dir, cfg, counts, qs_df, qm_df):
    """run_manifest.json with the pipeline's schema, aggregated over the merged batch runs."""
    hosts, commits, seeds, nprocs = set(), set(), set(), set()
    elapsed, n_batches = 0.0, 0
    for d in run_dirs:
        p = os.path.join(d, "run_manifest.json")
        if not os.path.isfile(p):
            continue
        with open(p, "r", encoding="utf-8") as fh:
            m = json.load(fh)
        hosts.add(m.get("hostname"))
        if m.get("git_commit"):
            commits.add(m["git_commit"])
        elapsed += m.get("elapsed_seconds") or 0
        if m.get("random_seed") is not None:
            seeds.add(m["random_seed"])
        if m.get("num_processes") is not None:
            nprocs.add(m["num_processes"])
        n_batches += (m.get("input", {}) or {}).get("n_batch_dirs", 1)

    def _one(s):
        s = sorted(x for x in s if x is not None)
        return s[0] if len(s) == 1 else s

    manifest = {
        "run_id": os.path.basename(out_dir),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "processing_function": FUNC,
        "merged": True,
        "merged_from_runs": len(run_dirs),
        "git_commit": _one(commits),
        "hostname": _one(hosts),
        "num_processes": _one(nprocs),
        "random_seed": _one(seeds),
        "input": {"root_data_dir": cfg.get("root_data_dir"), "n_batch_dirs": n_batches},
        "results": {"counts": counts, "total_jobs": sum(counts.values())},
        "outputs": {
            "summary_dir": out_dir,
            "query_summaries_csv": os.path.join(out_dir, f"query_summaries_{FUNC}.csv"),
            "n_query_summary_rows": int(len(qs_df)) if qs_df is not None else 0,
            "n_query_summary_cols": int(qs_df.shape[1]) if qs_df is not None else 0,
            "query_metrics_csv": os.path.join(out_dir, f"query_metrics_{FUNC}.csv") if qm_df is not None else None,
            "n_query_metrics_cols": int(qm_df.shape[1]) if qm_df is not None else 0,
        },
        "elapsed_seconds": round(elapsed, 2),   # summed compute across batches (not wall-clock)
    }
    with open(os.path.join(out_dir, "run_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)


def _clean_intermediates(base, out_dir):
    """Remove the per-batch host summary subtrees, leaving only the canonical merged dir + _control."""
    import shutil
    removed = 0
    for name in os.listdir(base):
        p = os.path.join(base, name)
        if not os.path.isdir(p) or p == os.path.abspath(out_dir) or name == "_control":
            continue
        if glob.glob(os.path.join(p, f"run_*_{FUNC}")):   # a host dir holding per-batch runs
            shutil.rmtree(p)
            removed += 1
    print(f"  cleaned {removed} intermediate host summary dir(s)")


def cmd_concat(args, cfg):
    """Assemble one canonical run dir (identical layout/filenames to a single monolithic run)."""
    import pandas as pd
    base = os.path.abspath(cfg["summary_dir"])
    run_dirs = per_batch_run_dirs(cfg)
    if not run_dirs:
        print(f"no per-batch run dirs found under {base}")
        return
    out_dir = os.path.join(base, f"run_{time.strftime('%d%m%Y_%H%M%S', time.gmtime())}_{FUNC}")
    os.makedirs(out_dir, exist_ok=True)

    def merge_csv(basename):
        frames = []
        for d in run_dirs:
            p = os.path.join(d, basename)
            if not os.path.isfile(p):
                continue
            try:
                frames.append(pd.read_csv(p))
            except pd.errors.EmptyDataError:
                continue                       # a batch with 0 Success rows writes an empty CSV
        if not frames:
            return None
        df = pd.concat(frames, ignore_index=True)
        if "job_name" in df.columns:
            df = df.drop_duplicates("job_name")
        out = os.path.join(out_dir, basename)
        df.to_csv(out, index=False)
        _zip_beside(out)                       # canonical <name>.csv.zip beside the raw csv
        return df

    qs_df = merge_csv(f"query_summaries_{FUNC}.csv")
    qm_df = merge_csv(f"query_metrics_{FUNC}.csv")
    for extra in ("apo_summaries.csv", "holo_summaries.csv"):
        merge_csv(extra)

    counts = _merge_job_logs(run_dirs, out_dir)
    _write_merged_parameters(run_dirs, out_dir, cfg)
    _write_merged_manifest(run_dirs, out_dir, cfg, counts, qs_df, qm_df)

    print(f"merged {len(run_dirs)} batch runs -> {out_dir}")
    if qs_df is not None:
        print(f"  query_summaries_{FUNC}.csv : {qs_df.shape[0]} rows x {qs_df.shape[1]} cols (+ .csv.zip)")
    if qm_df is not None:
        print(f"  query_metrics_{FUNC}.csv   : {qm_df.shape[0]} rows x {qm_df.shape[1]} cols (+ .csv.zip)")
    print(f"  counts: {counts}")
    if args.clean:
        _clean_intermediates(base, out_dir)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("init", "run", "status", "concat"):
        sp = sub.add_parser(name)
        sp.add_argument("--config", required=True, help="Path to the full-run JSON config.")
        if name == "init":
            sp.add_argument("--force", action="store_true", help="Overwrite an existing table.")
        if name == "run":
            sp.add_argument("--force", action="store_true", help=argparse.SUPPRESS)
            sp.add_argument("--stop-on-error", action="store_true",
                            help="Stop the loop on the first batch that fails (default: keep going).")
            sp.add_argument("--reverse", action="store_true",
                            help="Process batches from last to first (for a second machine running "
                                 "toward the first; stop both ends when they meet and merge).")
        if name == "status":
            sp.add_argument("--reverse", action="store_true",
                            help="Show the frontier/pending order for a reverse (last->first) run.")
        if name == "concat":
            sp.add_argument("--clean", action="store_true",
                            help="After merging, delete the per-batch host summary subtrees, "
                                 "leaving only the canonical merged run dir.")
    args = p.parse_args()
    cfg = load_config(args.config)
    {"init": cmd_init, "run": cmd_run, "status": cmd_status, "concat": cmd_concat}[args.cmd](args, cfg)


if __name__ == "__main__":
    main()
