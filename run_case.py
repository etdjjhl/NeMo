#!/usr/bin/env python3
"""
run_case.py — PhysicsNeMo Sym three-fin 2D heat-sink PINN runner.

Drives heat_sink.py via Hydra sys.argv injection, then parses monitor logs,
GPU stats, and generates a markdown baseline report with charts.
"""

import argparse
import csv
import glob
import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 1. Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PhysicsNeMo Sym heat-sink runner")
    p.add_argument("--case-dir",  required=True,  help="Path to three_fin_2d/ directory")
    p.add_argument("--out-dir",   required=True,  help="Output directory for this run")
    p.add_argument("--max-steps", type=int, default=10_000, help="Training steps (default 10000)")
    p.add_argument("--seed",      type=int, default=42,     help="Random seed (default 42)")
    p.add_argument("--mode",      choices=["baseline", "param", "transient"], default="baseline",
                   help="baseline=original heat_sink.py; param=heat_sink_param.py; transient=heat_sink_transient.py (default baseline)")
    p.add_argument("--local-disk", default="",
                   help="Fast local disk path (e.g. /home/featurize/data). "
                        "Training outputs symlinked there for I/O speed, copied back after.")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Seed everything
# ─────────────────────────────────────────────────────────────────────────────

def set_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# 3. Collect metadata
# ─────────────────────────────────────────────────────────────────────────────

def collect_metadata(args: argparse.Namespace, start_time: float) -> dict:
    meta: dict = {
        "timestamp": datetime.utcfromtimestamp(start_time).isoformat() + "Z",
        "params": {
            "max_steps": args.max_steps,
            "seed": args.seed,
            "mode": args.mode,
            "case_dir": str(args.case_dir),
            "out_dir": str(args.out_dir),
        },
        "environment": {},
        "versions": {},
    }

    # Git hash (best-effort)
    try:
        result = subprocess.run(
            ["git", "-C", args.case_dir, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            meta["git_hash"] = result.stdout.strip()
    except Exception:
        meta["git_hash"] = "unavailable"

    # GPU info
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,driver_version,memory.total,compute_cap",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            gpus = []
            for line in result.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 4:
                    gpus.append({
                        "name": parts[0],
                        "driver": parts[1],
                        "vram_mib": int(parts[2]),
                        "compute_cap": parts[3],
                    })
            meta["environment"]["gpus"] = gpus
    except Exception:
        meta["environment"]["gpus"] = []

    # Python / package versions
    meta["versions"]["python"] = sys.version

    for pkg in ("torch", "physicsnemo", "physicsnemo.sym", "numpy", "matplotlib"):
        try:
            mod = importlib.import_module(pkg)
            meta["versions"][pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            meta["versions"][pkg] = "not installed"

    return meta


# ─────────────────────────────────────────────────────────────────────────────
# 4. Local-disk symlink helpers
# ─────────────────────────────────────────────────────────────────────────────
# NOTE: We symlink *subdirectories* inside the hydra run dir (not the run dir
# itself) to avoid breaking physicsnemo's add_hydra_run_path() which requires
# os.getcwd() to be a real subpath of the original case_dir.  Symlinking the
# run dir causes os.getcwd() to resolve to the target on local_disk, which
# makes Path.relative_to(org_dir) fail.

# Heavy I/O subdirectories created by physicsnemo during training:
_SYMLINK_SUBDIRS = [
    "network_checkpoints",
    "monitors",
    "inferencers",
    "validators",
]


def _setup_local_symlink(
    case_dir: str, hydra_run_dir: str, local_disk: str, mode_tag: str
) -> str | None:
    """Symlink heavy subdirs inside case_dir/hydra_run_dir → local_disk.

    The run dir itself stays as a real directory so that Hydra's chdir keeps
    os.getcwd() inside case_dir.

    Returns the local_train_dir base path if symlinks were created, None otherwise.
    """
    if not local_disk:
        return None

    run_name = os.path.basename(hydra_run_dir)  # "run" or "run_param"
    local_train_dir = os.path.join(local_disk, "nemo_train", mode_tag, run_name)
    run_path = os.path.join(case_dir, hydra_run_dir)

    # If run_path is currently a symlink (from old strategy), convert it back
    if os.path.islink(run_path):
        old_target = os.path.realpath(run_path)
        os.remove(run_path)
        if os.path.isdir(old_target):
            shutil.copytree(old_target, run_path)
            print(f"[run_case] Converted old symlink back to real dir: {run_path}")
        else:
            os.makedirs(run_path, exist_ok=True)

    # Ensure the run dir exists as a real directory
    os.makedirs(run_path, exist_ok=True)

    for subdir in _SYMLINK_SUBDIRS:
        local_sub = os.path.join(local_train_dir, subdir)
        link_sub = os.path.join(run_path, subdir)

        os.makedirs(local_sub, exist_ok=True)

        # Move existing real subdir contents to local
        if os.path.isdir(link_sub) and not os.path.islink(link_sub):
            print(f"[run_case] Moving existing {link_sub} → {local_sub}")
            for item in os.listdir(link_sub):
                src = os.path.join(link_sub, item)
                dst = os.path.join(local_sub, item)
                if os.path.exists(dst):
                    if os.path.isdir(dst):
                        shutil.rmtree(dst)
                    else:
                        os.remove(dst)
                shutil.move(src, dst)
            os.rmdir(link_sub)

        # Remove stale symlink
        if os.path.islink(link_sub):
            os.remove(link_sub)
        elif os.path.exists(link_sub):
            shutil.rmtree(link_sub)

        os.symlink(local_sub, link_sub)
        print(f"[run_case] Symlink: {link_sub} → {local_sub}")

    return local_train_dir


def _teardown_local_symlink(
    case_dir: str, hydra_run_dir: str, local_train_dir: str | None
) -> None:
    """Replace subdir symlinks with real copies of the data from local disk."""
    if local_train_dir is None:
        return

    run_path = os.path.join(case_dir, hydra_run_dir)

    for subdir in _SYMLINK_SUBDIRS:
        link_sub = os.path.join(run_path, subdir)
        local_sub = os.path.join(local_train_dir, subdir)

        if os.path.islink(link_sub):
            os.remove(link_sub)
            if os.path.isdir(local_sub):
                shutil.copytree(local_sub, link_sub)
                print(f"[run_case] Copied back: {local_sub} → {link_sub}")
            else:
                os.makedirs(link_sub, exist_ok=True)
                print(f"[run_case] WARNING: local subdir not found, created empty: {link_sub}")
        elif not os.path.exists(link_sub) and os.path.isdir(local_sub):
            shutil.copytree(local_sub, link_sub)
            print(f"[run_case] Copied back (no link): {local_sub} → {link_sub}")

    print(f"[run_case] Teardown complete for {run_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5 & 6. Run heat_sink via subprocess (most reliable for Hydra config resolution)
# ─────────────────────────────────────────────────────────────────────────────

def run_heat_sink(case_dir: str, max_steps: int, out_dir: str, local_disk: str = "") -> None:
    """Run heat_sink.py as a subprocess from within case_dir.

    Using subprocess instead of importlib ensures Hydra resolves config_path="conf"
    relative to the script file, exactly as if the user ran it directly.
    """
    case_dir = os.path.abspath(case_dir)
    heat_sink_path = os.path.join(case_dir, "heat_sink.py")

    if not os.path.isfile(heat_sink_path):
        raise FileNotFoundError(f"heat_sink.py not found at {heat_sink_path}")

    freq = max(100, max_steps // 20)
    # hydra.run.dir must be a subpath of cwd (case_dir) due to physicsnemo
    # add_hydra_run_path() using Path.relative_to(). Use a relative path.
    hydra_run_dir = "outputs/run"

    overrides = [
        f"training.max_steps={max_steps}",
        f"training.rec_monitor_freq={freq}",
        f"training.rec_validation_freq={freq}",
        f"training.rec_inference_freq={freq}",
        f"hydra.run.dir={hydra_run_dir}",
    ]

    # Setup local-disk symlink if requested
    local_train_dir = _setup_local_symlink(case_dir, hydra_run_dir, local_disk, "baseline")

    cmd = [sys.executable, heat_sink_path] + overrides
    print(f"[run_case] Running: {' '.join(cmd)}")
    print(f"[run_case] cwd: {case_dir}")

    try:
        result = subprocess.run(cmd, cwd=case_dir)
    finally:
        # Teardown: replace symlink with real copy
        _teardown_local_symlink(case_dir, hydra_run_dir, local_train_dir)

    # Copy hydra outputs from case_dir into out_dir for archival
    src = os.path.join(case_dir, hydra_run_dir)
    if os.path.isdir(src):
        dst = os.path.join(out_dir, "hydra_outputs")
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"[run_case] Copied hydra outputs: {src} → {dst}")

    if result.returncode != 0:
        raise RuntimeError(f"heat_sink.py exited with code {result.returncode}")


def run_heat_sink_param(case_dir: str, max_steps: int, out_dir: str, local_disk: str = "") -> None:
    """Run heat_sink_param.py as a subprocess from within case_dir.

    Using subprocess instead of importlib ensures Hydra resolves config_path="conf"
    relative to the script file, exactly as if the user ran it directly.
    """
    case_dir = os.path.abspath(case_dir)
    heat_sink_path = os.path.join(case_dir, "heat_sink_param.py")

    if not os.path.isfile(heat_sink_path):
        raise FileNotFoundError(f"heat_sink_param.py not found at {heat_sink_path}")

    freq = max(100, max_steps // 20)
    hydra_run_dir = "outputs/run_param"

    overrides = [
        f"training.max_steps={max_steps}",
        f"training.rec_monitor_freq={freq}",
        f"training.rec_validation_freq={freq}",
        f"training.rec_inference_freq={freq}",
        f"hydra.run.dir={hydra_run_dir}",
    ]

    # Setup local-disk symlink if requested
    local_train_dir = _setup_local_symlink(case_dir, hydra_run_dir, local_disk, "param")

    cmd = [sys.executable, heat_sink_path] + overrides
    print(f"[run_case] Running: {' '.join(cmd)}")
    print(f"[run_case] cwd: {case_dir}")

    try:
        result = subprocess.run(cmd, cwd=case_dir)
    finally:
        # Teardown: replace symlink with real copy
        _teardown_local_symlink(case_dir, hydra_run_dir, local_train_dir)

    # Copy hydra outputs from case_dir into out_dir for archival
    src = os.path.join(case_dir, hydra_run_dir)
    if os.path.isdir(src):
        dst = os.path.join(out_dir, "hydra_outputs")
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"[run_case] Copied hydra outputs: {src} → {dst}")

    if result.returncode != 0:
        raise RuntimeError(f"heat_sink_param.py exited with code {result.returncode}")


def run_heat_sink_transient(case_dir: str, max_steps: int, out_dir: str, local_disk: str = "") -> None:
    """Run heat_sink_transient.py as a subprocess from within case_dir."""
    case_dir = os.path.abspath(case_dir)
    heat_sink_path = os.path.join(case_dir, "heat_sink_transient.py")

    if not os.path.isfile(heat_sink_path):
        raise FileNotFoundError(f"heat_sink_transient.py not found at {heat_sink_path}")

    freq = max(100, max_steps // 20)
    hydra_run_dir = "outputs/run_transient"

    overrides = [
        f"training.max_steps={max_steps}",
        f"training.rec_monitor_freq={freq}",
        f"training.rec_validation_freq={freq}",
        f"training.rec_inference_freq={freq}",
        f"hydra.run.dir={hydra_run_dir}",
    ]

    # Setup local-disk symlink if requested
    local_train_dir = _setup_local_symlink(case_dir, hydra_run_dir, local_disk, "transient")

    cmd = [sys.executable, heat_sink_path] + overrides
    print(f"[run_case] Running: {' '.join(cmd)}")
    print(f"[run_case] cwd: {case_dir}")

    try:
        result = subprocess.run(cmd, cwd=case_dir)
    finally:
        # Teardown: replace symlink with real copy
        _teardown_local_symlink(case_dir, hydra_run_dir, local_train_dir)

    # Copy hydra outputs from case_dir into out_dir for archival
    src = os.path.join(case_dir, hydra_run_dir)
    if os.path.isdir(src):
        dst = os.path.join(out_dir, "hydra_outputs")
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"[run_case] Copied hydra outputs: {src} → {dst}")

    if result.returncode != 0:
        raise RuntimeError(f"heat_sink_transient.py exited with code {result.returncode}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Parse monitor logs
# ─────────────────────────────────────────────────────────────────────────────

def parse_monitor_logs(out_dir: str) -> dict[str, list[dict]]:
    """Read Hydra/PhysicsNeMo monitor CSVs from hydra_outputs/monitors/."""
    monitors: dict[str, list[dict]] = {}
    patterns = [
        os.path.join(out_dir, "hydra_outputs", "monitors", "*.csv"),
        os.path.join(out_dir, "hydra_outputs", "**", "monitors", "*.csv"),
        os.path.join(out_dir, "monitors", "*.csv"),
        os.path.join(out_dir, "**", "*.csv"),
    ]
    found_files: list[str] = []
    for pat in patterns:
        found_files.extend(glob.glob(pat, recursive=True))
    found_files = list(dict.fromkeys(found_files))  # deduplicate, preserve order

    for fpath in found_files:
        name = Path(fpath).stem
        rows: list[dict] = []
        try:
            with open(fpath, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Convert numeric strings
                    parsed = {}
                    for k, v in row.items():
                        k = k.strip()
                        try:
                            parsed[k] = float(v)
                        except (ValueError, TypeError):
                            parsed[k] = v
                    rows.append(parsed)
        except Exception as e:
            print(f"[run_case] Warning: could not read {fpath}: {e}")
        if rows:
            monitors[name] = rows

    return monitors


# ─────────────────────────────────────────────────────────────────────────────
# 7. Parse GPU stats
# ─────────────────────────────────────────────────────────────────────────────

def parse_gpu_stats(out_dir: str) -> dict:
    """Parse nvidia-smi --query-gpu CSV output from gpu_stats.log.

    Expected format (one line per sample, comma-separated, nounits):
        timestamp, util_pct, mem_used_MiB, mem_total_MiB, temp_C, power_W
    e.g.: 2026/02/27 03:43:11.398, 4, 0, 24576, 38, 74.26
    """
    log_path = os.path.join(out_dir, "gpu_stats.log")
    result: dict = {"source": log_path, "samples": [], "summary": {}}

    if not os.path.isfile(log_path):
        result["error"] = "gpu_stats.log not found"
        return result

    # Fixed column names matching the --query-gpu fields used in run_case.sh
    HEADER = ["timestamp", "util_pct", "mem_used_MiB", "mem_total_MiB", "temp_C", "power_W"]
    NUMERIC = ["util_pct", "mem_used_MiB", "mem_total_MiB", "temp_C", "power_W"]

    samples: list[dict] = []

    with open(log_path) as f:
        for line in f:
            line = line.rstrip()
            # Skip blank lines and comment/header lines
            if not line or line.startswith("#"):
                continue
            # CSV: "2026/02/27 03:43:11.398, 4, 0, 24576, 38, 74.26"
            # timestamp contains a space, so split on ", " with maxsplit=len(HEADER)-1
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < len(HEADER):
                continue
            # Reconstruct timestamp (first two comma-parts are date and time)
            # Actually timestamp from nvidia-smi uses "/" and has no comma inside,
            # so a simple comma-split of 6 fields works fine.
            sample: dict = {}
            for col, val in zip(HEADER, parts):
                if col == "timestamp":
                    sample[col] = val
                else:
                    try:
                        sample[col] = float(val) if val not in ("", "-", "N/A", "[N/A]") else None
                    except ValueError:
                        sample[col] = None
            samples.append(sample)

    result["samples"] = samples
    result["header"] = HEADER

    if samples:
        summary: dict = {}
        for col in NUMERIC:
            vals = [s[col] for s in samples if s.get(col) is not None]
            if vals:
                summary[col] = {
                    "mean": float(np.mean(vals)),
                    "max":  float(np.max(vals)),
                    "min":  float(np.min(vals)),
                }
        result["summary"] = summary

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 8. Generate charts
# ─────────────────────────────────────────────────────────────────────────────

def generate_charts(monitors: dict, gpu_stats: dict, out_dir: str) -> list[str]:
    """Save per-monitor time-series charts and a combined key-metrics chart."""
    charts_dir = os.path.join(out_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)
    saved: list[str] = []

    # ── Per-monitor charts ──
    for name, rows in monitors.items():
        if not rows:
            continue
        # Find step/iteration column
        step_col = None
        for candidate in ("step", "Step", "iteration", "Iteration", "epoch"):
            if candidate in rows[0]:
                step_col = candidate
                break

        numeric_cols = [
            k for k, v in rows[0].items()
            if isinstance(v, float) and k != step_col
        ]
        if not numeric_cols:
            continue

        steps = (
            [r[step_col] for r in rows]
            if step_col
            else list(range(len(rows)))
        )

        fig, axes = plt.subplots(
            len(numeric_cols), 1,
            figsize=(10, 3 * len(numeric_cols)),
            squeeze=False
        )
        fig.suptitle(f"Monitor: {name}", fontsize=13)

        for ax, col in zip(axes[:, 0], numeric_cols):
            vals = [r.get(col) for r in rows]
            vals_clean = [v if v is not None else float("nan") for v in vals]
            ax.plot(steps, vals_clean, linewidth=1.2)
            ax.set_ylabel(col, fontsize=9)
            ax.set_xlabel(step_col or "sample", fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fpath = os.path.join(charts_dir, f"monitor_{name}.png")
        plt.savefig(fpath, dpi=120)
        plt.close(fig)
        saved.append(fpath)

    # ── Combined key-metrics chart ──
    key_patterns = ["loss", "residual", "imbalance", "error", "temperature", "velocity"]

    fig_rows: list[tuple] = []  # (label, steps, vals)
    for name, rows in monitors.items():
        if not rows:
            continue
        step_col = None
        for candidate in ("step", "Step", "iteration", "Iteration"):
            if candidate in rows[0]:
                step_col = candidate
                break
        steps = (
            [r[step_col] for r in rows]
            if step_col
            else list(range(len(rows)))
        )
        for col in rows[0]:
            if not isinstance(rows[0][col], float):
                continue
            if col == step_col:
                continue
            if any(p in col.lower() for p in key_patterns):
                vals = [r.get(col, float("nan")) for r in rows]
                fig_rows.append((f"{name}/{col}", steps, vals))

    if fig_rows:
        n = len(fig_rows)
        fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), squeeze=False)
        fig.suptitle("Key Metrics", fontsize=13)
        for ax, (label, steps, vals) in zip(axes[:, 0], fig_rows):
            ax.plot(steps, vals, linewidth=1.2)
            ax.set_ylabel(label, fontsize=8)
            ax.set_xlabel("step", fontsize=9)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fpath = os.path.join(charts_dir, "combined_key_metrics.png")
        plt.savefig(fpath, dpi=120)
        plt.close(fig)
        saved.append(fpath)

    # ── GPU utilisation chart ──
    samples = gpu_stats.get("samples", [])
    header  = gpu_stats.get("header", [])
    gpu_metric_map = {
        "sm":   "SM Utilization (%)",
        "mem":  "Memory Utilization (%)",
        "fb":   "Framebuffer Used (MiB)",
        "temp": "Temperature (°C)",
        "pwr":  "Power (W)",
    }

    gpu_cols = [c for c in header if c in gpu_metric_map and samples]
    if samples and gpu_cols:
        t = list(range(len(samples)))
        fig, axes = plt.subplots(len(gpu_cols), 1,
                                 figsize=(10, 3 * len(gpu_cols)),
                                 squeeze=False)
        fig.suptitle("GPU Monitoring (nvidia-smi dmon)", fontsize=13)
        for ax, col in zip(axes[:, 0], gpu_cols):
            vals = [s.get(col) for s in samples]
            vals_clean = [v if v is not None else float("nan") for v in vals]
            ax.plot(t, vals_clean, linewidth=1.2, color="tab:orange")
            ax.set_ylabel(gpu_metric_map[col], fontsize=9)
            ax.set_xlabel("sample (×5 s)", fontsize=9)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fpath = os.path.join(charts_dir, "gpu_stats.png")
        plt.savefig(fpath, dpi=120)
        plt.close(fig)
        saved.append(fpath)

    return saved


# ─────────────────────────────────────────────────────────────────────────────
# 9. Generate baseline report
# ─────────────────────────────────────────────────────────────────────────────

def generate_baseline_report(
    meta: dict,
    monitors: dict,
    gpu_stats: dict,
    charts: list[str],
    out_dir: str,
    elapsed_sec: float,
    run_error: str | None,
    mode: str = "baseline",
) -> str:
    """Write baseline_report.md and return its path."""
    report_path = os.path.join(out_dir, "baseline_report.md")
    lines: list[str] = []
    w = lines.append

    mode_titles = {"param": " [Parameterized]", "transient": " [Transient]"}
    title_tag = mode_titles.get(mode, "")
    w(f"# PhysicsNeMo Sym — Baseline Report{title_tag}")
    w("")
    w(f"**Generated:** {datetime.utcnow().isoformat()}Z")
    w("")

    # ── Environment ──
    w("## Environment")
    w("")
    w("| Key | Value |")
    w("|-----|-------|")
    gpus = meta.get("environment", {}).get("gpus", [])
    if gpus:
        g = gpus[0]
        w(f"| GPU | {g.get('name', 'N/A')} |")
        w(f"| Driver | {g.get('driver', 'N/A')} |")
        w(f"| VRAM | {g.get('vram_mib', 'N/A')} MiB |")
        w(f"| Compute Cap | {g.get('compute_cap', 'N/A')} |")
    for pkg in ("python", "torch", "physicsnemo", "physicsnemo.sym", "numpy"):
        ver = meta.get("versions", {}).get(pkg, "N/A")
        w(f"| {pkg} | {ver} |")
    w("")

    # ── Run Parameters ──
    w("## Run Parameters")
    w("")
    w("| Parameter | Value |")
    w("|-----------|-------|")
    params = meta.get("params", {})
    w(f"| max_steps | {params.get('max_steps', 'N/A')} |")
    w(f"| seed | {params.get('seed', 'N/A')} |")
    w(f"| git_hash | {meta.get('git_hash', 'N/A')} |")
    w(f"| timestamp | {meta.get('timestamp', 'N/A')} |")
    w("")

    # ── Runtime ──
    w("## Runtime")
    w("")
    mins, secs = divmod(int(elapsed_sec), 60)
    hrs, mins = divmod(mins, 60)
    w(f"- **Total elapsed:** {hrs:02d}h {mins:02d}m {secs:02d}s ({elapsed_sec:.1f} s)")
    if params.get("max_steps"):
        steps_per_sec = params["max_steps"] / max(elapsed_sec, 1)
        w(f"- **Throughput:** {steps_per_sec:.1f} steps/s")
    if run_error:
        w(f"- **Run error:** `{run_error}`")
    w("")

    # ── GPU Usage ──
    w("## GPU Usage")
    w("")
    summary = gpu_stats.get("summary", {})
    if summary:
        w("| Metric | Mean | Max | Min |")
        w("|--------|------|-----|-----|")
        gpu_labels = {
            "util_pct":     "GPU Util (%)",
            "mem_used_MiB": "Mem Used (MiB)",
            "temp_C":       "Temp (°C)",
            "power_W":      "Power (W)",
        }
        for col, label in gpu_labels.items():
            if col in summary:
                s = summary[col]
                w(f"| {label} | {s['mean']:.1f} | {s['max']:.1f} | {s['min']:.1f} |")
    else:
        w("*GPU stats not available.*")
    w("")

    # ── Numerical Stability ──
    w("## Numerical Stability")
    w("")
    # Look for mass_imbalance or similar in monitors
    stability_found = False
    for name, rows in monitors.items():
        if not rows:
            continue
        for col in rows[0]:
            if "imbalance" in col.lower() or "continuity" in col.lower():
                first_val = rows[0].get(col, float("nan"))
                last_val  = rows[-1].get(col, float("nan"))
                trend = "decreasing ✓" if last_val < first_val else "increasing ✗"
                w(f"- **{name}/{col}**: first={first_val:.4e}, last={last_val:.4e} ({trend})")
                stability_found = True
    if not stability_found:
        w("- Mass imbalance / continuity residuals not found in monitor logs.")
        w("  (OpenFOAM validation CSVs require NGC account; monitors still available.)")
    w("")

    # ── Monitor Results ──
    w("## Monitor Results (final step values)")
    w("")
    if monitors:
        for name, rows in monitors.items():
            if not rows:
                continue
            last = rows[-1]
            w(f"### {name}")
            w("")
            w("| Metric | Final Value |")
            w("|--------|-------------|")
            for k, v in last.items():
                if isinstance(v, float):
                    w(f"| {k} | {v:.6g} |")
            w("")
    else:
        w("*No monitor CSV files found.*")
        w("")
        w("> **Note:** This run used the default 10 000-step budget (2% of full 500 000 steps).")
        w("> Loss curves will show early-phase trends only; results are **not converged**.")
        w("")

    # ── Charts ──
    w("## Charts")
    w("")
    if charts:
        for c in charts:
            rel = os.path.relpath(c, out_dir)
            stem = Path(c).stem
            w(f"![{stem}]({rel})")
            w("")
    else:
        w("*No charts generated.*")
    w("")

    # ── Caveats ──
    w("## Caveats")
    w("")
    w("- Run used **10 000 steps** (default, ~2% of full 500 000). "
      "Use `MAX_STEPS=500000 bash run_case.sh` for a converged solution.")
    w("- OpenFOAM reference CSVs (requires NGC account) are absent; "
      "validators are skipped but PINN monitors remain functional.")
    w("- Charts show early-phase loss decrease; temperature/velocity fields "
      "will not yet be physically accurate at 10 k steps.")
    w("")

    report_text = "\n".join(lines)
    with open(report_path, "w") as f:
        f.write(report_text)

    return report_path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    start_time = time.time()

    # Set seeds before anything else
    set_seeds(args.seed)

    # Ensure output directory exists
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    local_disk = args.local_disk.strip()

    print(f"[run_case] Output directory: {out_dir}")
    print(f"[run_case] mode={args.mode}, max_steps={args.max_steps}, seed={args.seed}")
    if local_disk:
        print(f"[run_case] local_disk={local_disk}")

    # Collect metadata
    meta = collect_metadata(args, start_time)
    if local_disk:
        meta["params"]["local_disk"] = local_disk
    meta_path = os.path.join(out_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[run_case] Metadata saved: {meta_path}")

    # Run training (capture error for report but don't crash immediately)
    run_error: str | None = None
    try:
        if args.mode == "transient":
            print(f"[run_case] Starting heat_sink_transient training...")
            run_heat_sink_transient(args.case_dir, args.max_steps, out_dir, local_disk)
            print(f"[run_case] heat_sink_transient training complete.")
        elif args.mode == "param":
            print(f"[run_case] Starting heat_sink_param training...")
            run_heat_sink_param(args.case_dir, args.max_steps, out_dir, local_disk)
            print(f"[run_case] heat_sink_param training complete.")
        else:
            print(f"[run_case] Starting heat_sink training...")
            run_heat_sink(args.case_dir, args.max_steps, out_dir, local_disk)
            print(f"[run_case] heat_sink training complete.")
    except Exception as e:
        run_error = str(e)
        print(f"[run_case] ERROR during training: {e}")
        traceback.print_exc()

    elapsed = time.time() - start_time
    print(f"[run_case] Elapsed: {elapsed:.1f} s")

    # Read actual training step count from checkpoint
    optim_path = os.path.join(out_dir, "hydra_outputs", "optim_checkpoint.0.pth")
    actual_step = None
    if os.path.isfile(optim_path):
        import torch
        ckpt = torch.load(optim_path, map_location="cpu", weights_only=False)
        actual_step = ckpt.get("step")
        print(f"[run_case] Checkpoint step: {actual_step}")

    # Update metadata with elapsed time and training steps
    meta["elapsed_sec"] = elapsed
    meta["run_error"] = run_error
    meta["training_steps"] = {
        "requested": args.max_steps,
        "actual": actual_step,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Parse outputs
    print("[run_case] Parsing monitor logs...")
    monitors = parse_monitor_logs(out_dir)
    print(f"[run_case] Found {len(monitors)} monitor file(s): {list(monitors.keys())}")

    print("[run_case] Parsing GPU stats...")
    gpu_stats = parse_gpu_stats(out_dir)

    # Generate charts
    print("[run_case] Generating charts...")
    charts = generate_charts(monitors, gpu_stats, out_dir)
    print(f"[run_case] Charts saved: {charts}")

    # Generate report
    print("[run_case] Writing baseline report...")
    report_path = generate_baseline_report(
        meta, monitors, gpu_stats, charts, out_dir, elapsed, run_error,
        mode=args.mode,
    )
    print(f"[run_case] Report: {report_path}")

    # Final summary
    print("")
    print("=" * 50)
    print(f"  Done! elapsed={elapsed:.1f}s")
    print(f"  Report : {report_path}")
    print(f"  Charts : {os.path.join(out_dir, 'charts')}/")
    print(f"  Meta   : {meta_path}")
    if run_error:
        print(f"  WARNING: run completed with error: {run_error}")
    print("=" * 50)


if __name__ == "__main__":
    main()
