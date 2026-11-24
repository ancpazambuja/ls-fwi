#!/usr/bin/env python3
"""Extract an inversion campaign from MLflow into tidy CSVs for analysis.

Reusable across experiments: point it at an MLflow experiment id and it
discovers the sweep structure from the runs themselves (refit lineage from
the run names / refit runs, groups from the naming conventions).

Usage:
    # this campaign (defaults reproduce the 2026-11 extract):
    python3 notebooks/data/extract_from_mlflow.py

    # another experiment:
    python3 notebooks/data/extract_from_mlflow.py \
        --experiment-id 42 --out-dir notebooks/data/exp42 \
        [--mlflow-uri http://host:port] [--history-metric train_loss]

Produces in --out-dir:
    runs.csv                  one row per run (ALL runs, with cleaning flags)
    train_loss_history.csv    per-step history of --history-metric, clean kept runs
    inversion_per_model.csv   per-model scalars for the clean kept runs (from the
                              *_values metric series; step t = t-th smallest
                              inverted index, per task_processing.log_evaluation_results)
    refit_baselines.csv       the refit models' final metrics (all metric keys)

Group conventions (from the run naming; see DATA_DICTIONARY.md):
    tst__ prefix            -> tst_sanity   (sanity/debug runs)
    '<n>-samples' in name   -> debug_<n>samples (non-full sample variants = debug)
    no logged params        -> param_less  (earlier launcher generation)
    otherwise               -> sweep_full_samples
Cleaning policy (user-confirmed 2026-11-24): skip the debug/tst/param-less
groups; within the sweep, per combo (refit, latent_init, lr, max_epochs)
keep the LAST run by start time unless it has no valid metrics; duplicates
are flagged, never deleted.

Only the Python stdlib is required.
"""
import argparse
import csv
import json
import os
import re
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone

DEFAULT_URI = "http://gr14sm02.ep.petrobras.com.br:31947"
DEFAULT_EXPERIMENT = "97"          # encdec.highres.inversion
DEFAULT_HISTORY_METRIC = "train_loss"

IDENTITY = ["run_id", "run_name", "group", "kept_in_clean_set",
            "has_valid_metrics", "combo", "refit_short", "arch_family",
            "kernel", "seed", "start_time_iso", "start_time_ms", "status"]
PARAMS = ["latent_init", "lr", "max_epochs", "lr_scheduler", "patience",
          "model_size", "test_perc"]

SAMPLES_RE = re.compile(r"(?:^|_)([A-Za-z0-9]+-samples)(?:_|$)")
ARCH_RE = re.compile(r"(transencdec|encdec)-arch")
HEX32_RE = re.compile(r"[0-9a-f]{32}")
KERNEL_RE = re.compile(r"_([0-9]+)x[0-9]+_")


def api(uri, path, payload=None):
    url = uri + "/api/2.0/mlflow" + path
    if payload is None:
        req = urllib.request.Request(url)
    else:
        req = urllib.request.Request(url,
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def fetch_all_runs(uri, experiment_id):
    runs, token = [], None
    while True:
        payload = {"experiment_ids": [experiment_id], "max_results": 1000,
                   "run_view_type": "ALL"}
        if token:
            payload["page_token"] = token
        d = api(uri, "/runs/search", payload)
        runs += d.get("runs", [])
        token = d.get("next_page_token")
        if not token:
            return runs


def classify(name, has_params):
    if name.startswith("tst__"):
        return "tst_sanity"
    m = SAMPLES_RE.search(name)
    if m and m.group(1) != "full-samples":
        return "debug_" + m.group(1).replace("-", "")
    if not has_params:
        return "param_less"
    return "sweep_full_samples"


def has_valid_metrics(metrics):
    return any(isinstance(v, (int, float)) and v == v and abs(v) != float("inf")
               for v in metrics.values())


def fetch_history(uri, run_id, key):
    out, token, fell_back = [], None, False
    while True:
        path = f"/metrics/get-history?run_id={run_id}&metric_key={key}"
        if token:
            path += f"&page_token={token}"
        try:
            if fell_back:
                d = api(uri, path)
            else:
                d = api(uri, path + "&max_results=25000")
        except Exception:
            if fell_back:
                raise
            fell_back = True   # older API without pagination support
            d = api(uri, path)
        out += d.get("metrics", [])
        token = d.get("next_page_token")
        if not token:
            return out


def iso(ms):
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S") if ms else ""


def write_csv(path, fields, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT)
    ap.add_argument("--mlflow-uri", default=DEFAULT_URI)
    ap.add_argument("--out-dir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--history-metric", default=DEFAULT_HISTORY_METRIC)
    ap.add_argument("--models-to-invert", default="1,10,20",
                    help="comma-separated inverted-model indices, in the order "
                         "they appear in the processing code; step t of the "
                         "*_values metric series maps to the t-th smallest index")
    args = ap.parse_args()
    uri = args.mlflow_uri.rstrip("/")

    raw = fetch_all_runs(uri, args.experiment_id)
    print(f"fetched {len(raw)} runs from experiment {args.experiment_id}")

    # ---- pass 1: build run rows, keep params/metrics dicts for column unions
    rows, param_dicts, metric_dicts = [], [], []
    for r in raw:
        info, data = r["info"], r["data"]
        params = {p["key"]: p["value"] for p in data.get("params", [])}
        metrics = {m["key"]: m["value"] for m in data.get("metrics", [])}
        tags = {t["key"]: t["value"] for t in data.get("tags", [])}
        name = tags.get("mlflow.runName", "")
        refit_full = HEX32_RE.search(name).group(0) if HEX32_RE.search(name) else ""
        refit_short = refit_full[:8]
        am = ARCH_RE.search(name)
        has_params = bool(params)
        group = classify(name, has_params)
        combo = ""
        if group == "sweep_full_samples":
            combo = (f"{refit_short}|{params.get('latent_init')}|"
                     f"lr={params.get('lr')}|ep={params.get('max_epochs')}")
        start_ms = int(info.get("start_time", 0))
        rows.append({
            "run_id": info["run_id"], "run_name": name, "group": group,
            "kept_in_clean_set": "", "has_valid_metrics": has_valid_metrics(metrics),
            "combo": combo, "refit_short": refit_short,
            "arch_family": am.group(1) if am else "", "kernel": "", "seed": "",
            "latent_init": params.get("latent_init", ""),
            "lr": params.get("lr", ""), "max_epochs": params.get("max_epochs", ""),
            "lr_scheduler": params.get("lr_scheduler", ""),
            "patience": params.get("patience", ""),
            "model_size": params.get("model_size", ""),
            "test_perc": params.get("_x_test_perc", ""),
            "start_time_iso": iso(start_ms), "start_time_ms": start_ms,
            "status": info.get("status", ""),
        })
        param_dicts.append(params)
        metric_dicts.append(metrics)

    # dynamic metric columns: everything inversion-related + loss/duration
    metric_keys = sorted({k for d in metric_dicts for k in d
                          if k.startswith("inversion_") or k in ("train_loss", "dur")})
    fields = IDENTITY + PARAMS + metric_keys
    for row, md in zip(rows, metric_dicts):
        for k in metric_keys:
            row[k] = md.get(k, "")

    # ---- refit registry, derived from the data (no hardcoded ids)
    registry = {}   # short -> dict(full, arch, kernel, seed)
    for r in rows:
        if r["group"] != "sweep_full_samples" or not r["refit_short"]:
            continue
        short = r["refit_short"]
        if short not in registry:
            registry[short] = {"full": HEX32_RE.search(r["run_name"]).group(0),
                               "arch": r["arch_family"]}
    for short, reg in registry.items():
        d = api(uri, f"/runs/get?run_id={reg['full']}")["run"]
        params = {p["key"]: p["value"] for p in d["data"].get("params", [])}
        km = KERNEL_RE.search(d["info"].get("run_name", "") or "")
        reg["kernel"] = int(km.group(1)) if km else ""
        reg["seed"] = params.get("_x_seed", "")
        reg["metrics"] = {m["key"]: m["value"] for m in d["data"].get("metrics", [])}
    for r in rows:
        reg = registry.get(r["refit_short"])
        if reg:
            r["kernel"] = reg.get("kernel", "")
            r["seed"] = reg.get("seed", "")

    # ---- cleaning: keep-last per combo within the sweep group
    # Selection is CHRONOLOGICAL ONLY (sort by start time, take the last):
    # no metric/score is ever consulted, so no performance bias. The only
    # exception, per the user's policy, is when the last run has no valid
    # metrics — then fall back to the last *valid* run (never fired in the
    # 2026-11 campaign: 0 fallbacks).
    by_combo = defaultdict(list)
    for row in rows:
        if row["group"] == "sweep_full_samples":
            by_combo[row["combo"]].append(row)
    kept = []
    for combo, rs in by_combo.items():
        rs.sort(key=lambda r: (r["start_time_ms"], r["run_id"]))
        kept_run = rs[-1] if rs[-1]["has_valid_metrics"] else None
        if kept_run is None:
            for r in reversed(rs[:-1]):
                if r["has_valid_metrics"]:
                    kept_run = r
                    break
        if kept_run is None:   # whole combo invalid: keep last anyway
            kept_run = rs[-1]
        kept_run["kept_in_clean_set"] = True
        kept.append(kept_run)
    for row in rows:   # non-kept sweep rows are explicit False (never empty)
        if row["group"] == "sweep_full_samples" and row["kept_in_clean_set"] == "":
            row["kept_in_clean_set"] = False

    counts = defaultdict(int)
    for row in rows:
        counts[row["group"]] += 1
    n_dup = sum(1 for r in rows
                if r["group"] == "sweep_full_samples" and not r["kept_in_clean_set"])
    print("groups:", dict(counts))
    print(f"clean kept: {len(kept)} combos (duplicates flagged: {n_dup})")

    os.makedirs(args.out_dir, exist_ok=True)
    write_csv(os.path.join(args.out_dir, "runs.csv"), fields, rows)

    # ---- per-step history for the clean kept runs
    hist_fields = ["run_id", "combo", "refit_short", "latent_init", "lr",
                   "max_epochs", "step", "value"]
    hist_path = os.path.join(args.out_dir, "train_loss_history.csv")
    n_rows, n_failed = 0, 0
    with open(hist_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(hist_fields)
        for i, row in enumerate(sorted(kept, key=lambda r: r["combo"]), 1):
            try:
                hist = fetch_history(uri, row["run_id"], args.history_metric)
            except Exception as e:
                print(f"  history FAILED {row['combo']}: {e}")
                n_failed += 1
                continue
            for m in hist:
                w.writerow([row["run_id"], row["combo"], row["refit_short"],
                            row["latent_init"], row["lr"], row["max_epochs"],
                            m.get("step", ""), m.get("value", "")])
                n_rows += 1
            if i % 50 == 0:
                print(f"  histories {i}/{len(kept)}")
                f.flush()
    print(f"history rows: {n_rows} across {len(kept) - n_failed} runs "
          f"(metric: {args.history_metric})")

    # ---- per-model results: the *_values metrics carry the per-model scalars,
    # logged with step = position in the sorted inverted-model list
    # (task_processing.log_evaluation_results: enumerate over the per-model
    # values). step t -> sorted(models_to_invert)[t].
    model_ids = sorted(int(x) for x in args.models_to_invert.split(","))
    VALUES_METRICS = ["inversion_mse_values", "inversion_mae_values",
                      "inversion_ssim_values"]
    pm_fields = ["run_id", "combo", "refit_short", "arch_family", "kernel",
                 "seed", "latent_init", "lr", "max_epochs", "model_step",
                 "model_idx", "mse", "mae", "ssim"]
    pm_path = os.path.join(args.out_dir, "inversion_per_model.csv")
    n_pm, n_pm_failed = 0, 0
    with open(pm_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=pm_fields)
        w.writeheader()
        for i, row in enumerate(sorted(kept, key=lambda r: r["combo"]), 1):
            try:
                series = {k: fetch_history(uri, row["run_id"], k)
                          for k in VALUES_METRICS}
            except Exception as e:
                print(f"  per-model FAILED {row['combo']}: {e}")
                n_pm_failed += 1
                continue
            ordered, ok = {}, True
            for k, pts in series.items():
                pts = sorted(pts, key=lambda m: m.get("step", 0))
                vals = [m.get("value", "") for m in pts]
                if len(vals) != len(model_ids):
                    ok = False
                    break
                ordered[k] = vals
            if not ok:
                print(f"  per-model SKIPPED {row['combo']}: "
                      f"expected {len(model_ids)} values per metric")
                n_pm_failed += 1
                continue
            for t, idx in enumerate(model_ids):
                w.writerow({
                    "run_id": row["run_id"], "combo": row["combo"],
                    "refit_short": row["refit_short"],
                    "arch_family": row["arch_family"], "kernel": row["kernel"],
                    "seed": row["seed"], "latent_init": row["latent_init"],
                    "lr": row["lr"], "max_epochs": row["max_epochs"],
                    "model_step": t, "model_idx": idx,
                    "mse": ordered["inversion_mse_values"][t],
                    "mae": ordered["inversion_mae_values"][t],
                    "ssim": ordered["inversion_ssim_values"][t],
                })
                n_pm += 1
            if i % 50 == 0:
                print(f"  per-model {i}/{len(kept)}")
                f.flush()
    print(f"per-model rows: {n_pm} across {len(kept) - n_pm_failed} runs")

    # ---- refit baselines: all metric keys of each refit run
    base_fields = ["refit_run_id", "refit_short", "arch_family", "kernel", "seed"]
    base_metric_keys = sorted({k for reg in registry.values() for k in reg.get("metrics", {})})
    base_rows = []
    for short, reg in sorted(registry.items()):
        row = {"refit_run_id": reg["full"], "refit_short": short,
               "arch_family": reg["arch"], "kernel": reg["kernel"], "seed": reg["seed"]}
        for k in base_metric_keys:
            row[k] = reg.get("metrics", {}).get(k, "")
        base_rows.append(row)
    write_csv(os.path.join(args.out_dir, "refit_baselines.csv"),
              base_fields + base_metric_keys, base_rows)


if __name__ == "__main__":
    main()
