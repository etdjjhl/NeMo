#!/usr/bin/env python3
"""
validate_transient.py — Post-training validation for the transient PINN model.

Generates field plots at time snapshots, compares t=t_max with OpenFOAM
steady-state reference, plots time evolution at probe points, and runs
physical trend checks.

Usage:
    python validate_transient.py --checkpoint-dir cases/three_fin_2d/outputs/run_transient
    python validate_transient.py --checkpoint-dir outputs/transient/latest/hydra_outputs
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np

try:
    import torch
except ImportError:
    print("ERROR: torch is required. Activate the nemo conda environment.")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

T_MAX = 10.0
SNAPSHOT_TIMES = [0.0, 2.5, 5.0, T_MAX]
# Probe points must be in the fluid domain (outside heat sink fins).
# Fins cover x in [-1, 0], y in [-0.30,-0.20], [-0.05,+0.05], [+0.20,+0.30].
# (0.5, 0.0) is downstream of the fins; (-0.5, 0.12) is in the gap between fins.
PROBE_POINTS = [(0.5, 0.0), (-2.0, 0.0), (2.0, 0.0), (-0.5, 0.12)]
FIELDS = ["u", "v", "p", "c"]

CHANNEL_LENGTH = (-2.5, 2.5)
CHANNEL_WIDTH = (-0.5, 0.5)
HEAT_SINK_ORIGIN_BASE = (-1.0, -0.3)
NR_FINS = 3
GAP = 0.25
FIN_LENGTH = 1.0
FIN_THICKNESS = 0.1


def parse_args():
    p = argparse.ArgumentParser(description="Validate transient PINN model")
    p.add_argument(
        "--checkpoint-dir", required=True,
        help="Directory containing network_checkpoints/ from transient training"
    )
    p.add_argument(
        "--openfoam-csv", default="",
        help="Path to OpenFOAM reference CSV (heat_sink_zeroEq_Pr5_mesh20.csv)"
    )
    p.add_argument(
        "--output-dir", default="transient_validation",
        help="Output directory for plots and report (default: transient_validation)"
    )
    p.add_argument(
        "--n-grid", type=int, default=200,
        help="Grid resolution per axis for field plots (default: 200)"
    )
    p.add_argument(
        "--n-time", type=int, default=100,
        help="Number of time samples for probe evolution (default: 100)"
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_heat_sink_rects():
    """Return list of (x0, y0, x1, y1) for each fin rectangle."""
    rects = []
    origin = list(HEAT_SINK_ORIGIN_BASE)
    for i in range(NR_FINS):
        x0, y0 = origin
        x1, y1 = x0 + FIN_LENGTH, y0 + FIN_THICKNESS
        rects.append((x0, y0, x1, y1))
        origin[1] += GAP
    return rects


def point_in_heat_sink(x, y, rects):
    """Check if points are inside any heat sink fin."""
    inside = np.zeros(len(x), dtype=bool)
    for x0, y0, x1, y1 in rects:
        inside |= (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    return inside


def make_domain_grid(n_grid, rects):
    """Create a grid of points inside the channel but outside the heat sink."""
    x_lin = np.linspace(CHANNEL_LENGTH[0], CHANNEL_LENGTH[1], n_grid)
    y_lin = np.linspace(CHANNEL_WIDTH[0], CHANNEL_WIDTH[1], n_grid)
    xx, yy = np.meshgrid(x_lin, y_lin)
    x_flat, y_flat = xx.ravel(), yy.ravel()
    mask = ~point_in_heat_sink(x_flat, y_flat, rects)
    return x_flat[mask], y_flat[mask], xx, yy, mask


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def _find_artifact_dir(checkpoint_dir):
    """Locate the directory containing .hydra/config.yaml and network checkpoints."""
    root = Path(checkpoint_dir).resolve()
    # Check common locations
    candidates = [
        root,
        root / "hydra_outputs",
        root / "outputs" / "run_transient",
    ]
    for path in candidates:
        hydra_cfg = path / ".hydra" / "config.yaml"
        has_ckpts = list(path.glob("flow_network*.pth")) or \
                    list((path / "network_checkpoints").glob("flow_network*.pth")) if (path / "network_checkpoints").is_dir() else []
        if hydra_cfg.exists():
            return path
    # Fallback: search recursively for .hydra/config.yaml
    for hydra_cfg in root.rglob(".hydra/config.yaml"):
        return hydra_cfg.parent.parent
    raise FileNotFoundError(
        f"Could not find .hydra/config.yaml under {root}. "
        "Point --checkpoint-dir to the Hydra run directory."
    )


def load_models(checkpoint_dir):
    """Load flow_net and heat_net by reading the training Hydra config.

    Reconstructs the exact network architecture from .hydra/config.yaml
    (same approach as compare_openfoam_csv.load_models), then loads weights.
    """
    from omegaconf import OmegaConf
    from physicsnemo.sym.hydra import instantiate_arch
    from physicsnemo.sym.key import Key

    artifact_dir = _find_artifact_dir(checkpoint_dir)
    cfg_path = artifact_dir / ".hydra" / "config.yaml"
    cfg = OmegaConf.load(cfg_path)

    input_keys = [Key("x"), Key("y"), Key("t")]

    flow_net = instantiate_arch(
        input_keys=input_keys,
        output_keys=[Key("u"), Key("v"), Key("p")],
        cfg=cfg.arch.fully_connected,
    )
    heat_net = instantiate_arch(
        input_keys=input_keys,
        output_keys=[Key("c")],
        cfg=cfg.arch.fully_connected,
    )

    # Find checkpoint files (in artifact_dir or artifact_dir/network_checkpoints)
    ckpt_dir = artifact_dir
    if not list(ckpt_dir.glob("flow_network*.pth")):
        ckpt_dir = artifact_dir / "network_checkpoints"
    flow_ckpts = sorted(ckpt_dir.glob("flow_network*.pth"))
    heat_ckpts = sorted(ckpt_dir.glob("heat_network*.pth"))
    if not flow_ckpts or not heat_ckpts:
        raise FileNotFoundError(f"Checkpoint files not found in {artifact_dir} or {ckpt_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flow_net.load_state_dict(torch.load(flow_ckpts[-1], map_location=device, weights_only=False))
    heat_net.load_state_dict(torch.load(heat_ckpts[-1], map_location=device, weights_only=False))
    flow_net.to(device).eval()
    heat_net.to(device).eval()
    return flow_net, heat_net, device


@torch.no_grad()
def predict(flow_net, heat_net, x, y, t_val, device=None):
    """Run inference at given (x, y, t) points. Returns dict of numpy arrays."""
    if device is None:
        device = torch.device("cpu")
    n = len(x)
    x_t = torch.as_tensor(x, dtype=torch.float32, device=device).reshape(-1, 1)
    y_t = torch.as_tensor(y, dtype=torch.float32, device=device).reshape(-1, 1)
    t_t = torch.full_like(x_t, t_val)

    model_in = {"x": x_t, "y": y_t, "t": t_t}
    flow_out = flow_net(model_in)
    heat_out = heat_net(model_in)

    return {
        "u": flow_out["u"].cpu().numpy().ravel(),
        "v": flow_out["v"].cpu().numpy().ravel(),
        "p": flow_out["p"].cpu().numpy().ravel(),
        "c": heat_out["c"].cpu().numpy().ravel(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_fields_at_time(flow_net, heat_net, t_val, x, y, xx, yy, mask, rects, out_dir, device=None):
    """Plot u, v, p, c fields at a given time snapshot."""
    preds = predict(flow_net, heat_net, x, y, t_val, device)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"Fields at t = {t_val:.1f} s", fontsize=14)

    for ax, field in zip(axes.ravel(), FIELDS):
        # Reconstruct full grid with NaN for heat sink
        full = np.full(xx.ravel().shape, np.nan)
        full[mask] = preds[field]
        z = full.reshape(xx.shape)
        im = ax.pcolormesh(xx, yy, z, shading="auto", cmap="coolwarm")
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(field)
        ax.set_aspect("equal")
        # Draw heat sink outlines
        for x0, y0, x1, y1 in rects:
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                       fill=True, color="gray", alpha=0.5))

    plt.tight_layout()
    tag = f"t{t_val:.1f}".replace(".", "p")
    fpath = os.path.join(out_dir, f"fields_{tag}.png")
    plt.savefig(fpath, dpi=150)
    plt.close(fig)
    return fpath


def plot_probe_evolution(flow_net, heat_net, n_time, out_dir, device=None):
    """Plot time evolution of all fields at probe points."""
    t_arr = np.linspace(0, T_MAX, n_time)

    fig, axes = plt.subplots(len(FIELDS), 1, figsize=(10, 3 * len(FIELDS)))
    fig.suptitle("Time Evolution at Probe Points", fontsize=14)

    for ax, field in zip(axes, FIELDS):
        for px, py in PROBE_POINTS:
            vals = []
            for t_val in t_arr:
                pred = predict(flow_net, heat_net,
                               np.array([px]), np.array([py]), t_val, device)
                vals.append(pred[field][0])
            ax.plot(t_arr, vals, label=f"({px},{py})")
        ax.set_ylabel(field)
        ax.set_xlabel("t (s)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fpath = os.path.join(out_dir, "probe_evolution.png")
    plt.savefig(fpath, dpi=150)
    plt.close(fig)
    return fpath


# ─────────────────────────────────────────────────────────────────────────────
# OpenFOAM comparison
# ─────────────────────────────────────────────────────────────────────────────

def compare_openfoam(flow_net, heat_net, csv_path, out_dir, device=None):
    """Compare t=t_max predictions with OpenFOAM steady-state reference."""
    try:
        from physicsnemo.sym.utils.io import csv_to_dict
    except ImportError:
        warnings.warn("Cannot import csv_to_dict; skipping OpenFOAM comparison.")
        return None

    base_temp = 293.498
    nu = 0.01
    mapping = {
        "Points:0": "x", "Points:1": "y",
        "U:0": "u", "U:1": "v", "p": "p", "T": "c",
    }
    data = csv_to_dict(csv_path, mapping)
    data["c"] = (data["c"] - base_temp) / 273.15

    x, y = data["x"].ravel(), data["y"].ravel()
    preds = predict(flow_net, heat_net, x, y, T_MAX, device)

    errors = {}
    for field in FIELDS:
        ref = data[field].ravel()
        pred = preds[field]
        mae = np.mean(np.abs(pred - ref))
        rmse = np.sqrt(np.mean((pred - ref) ** 2))
        errors[field] = {"MAE": mae, "RMSE": rmse}

    # Scatter plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"PINN (t={T_MAX}) vs OpenFOAM Steady-State", fontsize=14)
    for ax, field in zip(axes.ravel(), FIELDS):
        ref = data[field].ravel()
        pred = preds[field]
        ax.scatter(ref, pred, s=1, alpha=0.3)
        lims = [min(ref.min(), pred.min()), max(ref.max(), pred.max())]
        ax.plot(lims, lims, "r--", linewidth=1)
        ax.set_xlabel(f"OpenFOAM {field}")
        ax.set_ylabel(f"PINN {field}")
        ax.set_title(f"{field}: MAE={errors[field]['MAE']:.4f}, RMSE={errors[field]['RMSE']:.4f}")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fpath = os.path.join(out_dir, "openfoam_comparison.png")
    plt.savefig(fpath, dpi=150)
    plt.close(fig)
    return errors


# ─────────────────────────────────────────────────────────────────────────────
# Physical trend checks
# ─────────────────────────────────────────────────────────────────────────────

def physical_trend_checks(flow_net, heat_net, device=None):
    """Check that physical trends are correct.

    All check points must be in the fluid domain (outside heat sink fins).
    Fins cover x in [-1, 0], y bands [-0.30,-0.20], [-0.05,+0.05], [+0.20,+0.30].
    """
    checks = []

    # Fluid-domain check points: upstream, fin-gap, downstream
    x_pts = np.array([-2.0, -0.5, 0.5])
    y_pts = np.array([0.0, 0.12, 0.0])
    pred_t0 = predict(flow_net, heat_net, x_pts, y_pts, 0.0, device)
    speed_t0 = np.sqrt(pred_t0["u"] ** 2 + pred_t0["v"] ** 2)
    max_speed_t0 = speed_t0.max()
    checks.append({
        "name": "IC: speed near zero at t=0",
        "value": float(max_speed_t0),
        "threshold": 0.5,
        "passed": max_speed_t0 < 0.5,
    })

    # At t=0, temperature should be near zero (ambient)
    max_c_t0 = np.abs(pred_t0["c"]).max()
    checks.append({
        "name": "IC: temperature near ambient at t=0",
        "value": float(max_c_t0),
        "threshold": 0.1,
        "passed": max_c_t0 < 0.1,
    })

    # At t=t_max, speed should be established at downstream center (0.5, 0)
    pred_tmax = predict(flow_net, heat_net,
                        np.array([0.5]), np.array([0.0]), T_MAX, device)
    speed_tmax = np.sqrt(pred_tmax["u"][0] ** 2 + pred_tmax["v"][0] ** 2)
    checks.append({
        "name": "Steady: speed at (0.5,0) at t=t_max > 0.5",
        "value": float(speed_tmax),
        "threshold": 0.5,
        "passed": speed_tmax > 0.5,
    })

    # Speed should increase from t=0 to t=t_max at downstream center
    # Use index 2 which is (0.5, 0.0)
    checks.append({
        "name": "Trend: speed increases from t=0 to t=t_max at (0.5,0)",
        "value": f"{float(speed_t0[2]):.4f} -> {float(speed_tmax):.4f}",
        "threshold": "monotonic increase",
        "passed": speed_tmax > speed_t0[2],
    })

    return checks


# ─────────────────────────────────────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(out_dir, field_plots, probe_plot, errors, checks):
    """Generate transient_validation_report.md."""
    lines = []
    w = lines.append

    w("# Transient PINN Validation Report")
    w("")
    w(f"**Time range:** t in [0, {T_MAX}] s")
    w(f"**Snapshot times:** {SNAPSHOT_TIMES}")
    w(f"**Probe points:** {PROBE_POINTS}")
    w("")

    # Physical trend checks
    w("## Physical Trend Checks")
    w("")
    w("| Check | Value | Threshold | Passed |")
    w("|-------|-------|-----------|--------|")
    n_passed = 0
    for ch in checks:
        status = "YES" if ch["passed"] else "NO"
        if ch["passed"]:
            n_passed += 1
        w(f"| {ch['name']} | {ch['value']} | {ch['threshold']} | {status} |")
    w("")
    w(f"**Result: {n_passed}/{len(checks)} checks passed.**")
    w("")

    # OpenFOAM comparison
    if errors:
        w("## OpenFOAM Comparison (t=t_max vs steady-state)")
        w("")
        w("| Field | MAE | RMSE |")
        w("|-------|-----|------|")
        for field in FIELDS:
            if field in errors:
                e = errors[field]
                w(f"| {field} | {e['MAE']:.6f} | {e['RMSE']:.6f} |")
        w("")
    else:
        w("## OpenFOAM Comparison")
        w("")
        w("*OpenFOAM reference CSV not provided or not available.*")
        w("")

    # Field plots
    w("## Field Snapshots")
    w("")
    for fp in field_plots:
        rel = os.path.relpath(fp, out_dir)
        stem = Path(fp).stem
        w(f"![{stem}]({rel})")
        w("")

    # Probe evolution
    if probe_plot:
        w("## Probe Point Time Evolution")
        w("")
        rel = os.path.relpath(probe_plot, out_dir)
        w(f"![probe_evolution]({rel})")
        w("")

    report_path = os.path.join(out_dir, "transient_validation_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    return report_path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[validate] Loading models from {args.checkpoint_dir}...")
    flow_net, heat_net, device = load_models(args.checkpoint_dir)
    print("[validate] Models loaded.")

    rects = make_heat_sink_rects()
    x, y, xx, yy, mask = make_domain_grid(args.n_grid, rects)

    # Field plots at snapshots
    print("[validate] Generating field plots...")
    field_plots = []
    for t_val in SNAPSHOT_TIMES:
        print(f"  t = {t_val:.1f}")
        fp = plot_fields_at_time(flow_net, heat_net, t_val, x, y, xx, yy, mask, rects, out_dir, device)
        field_plots.append(fp)

    # Probe evolution
    print("[validate] Generating probe evolution plots...")
    probe_plot = plot_probe_evolution(flow_net, heat_net, args.n_time, out_dir, device)

    # OpenFOAM comparison
    errors = None
    if args.openfoam_csv and os.path.isfile(args.openfoam_csv):
        print(f"[validate] Comparing with OpenFOAM: {args.openfoam_csv}")
        errors = compare_openfoam(flow_net, heat_net, args.openfoam_csv, out_dir, device)
    else:
        print("[validate] No OpenFOAM CSV provided; skipping comparison.")

    # Physical trend checks
    print("[validate] Running physical trend checks...")
    checks = physical_trend_checks(flow_net, heat_net, device)
    for ch in checks:
        status = "PASS" if ch["passed"] else "FAIL"
        print(f"  [{status}] {ch['name']}: {ch['value']}")

    # Generate report
    print("[validate] Generating report...")
    report_path = generate_report(out_dir, field_plots, probe_plot, errors, checks)
    print(f"[validate] Report: {report_path}")
    print(f"[validate] Plots:  {out_dir}/")
    print("[validate] Done.")


if __name__ == "__main__":
    main()
