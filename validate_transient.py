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
PROBE_POINTS = [(0.0, 0.0), (-2.0, 0.0), (2.0, 0.0), (0.0, 0.4)]
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

def load_models(checkpoint_dir):
    """Load flow_net and heat_net from checkpoint directory."""
    ckpt_dir = os.path.join(checkpoint_dir, "network_checkpoints")
    if not os.path.isdir(ckpt_dir):
        # Try one level up
        alt = os.path.join(checkpoint_dir, "outputs", "run_transient", "network_checkpoints")
        if os.path.isdir(alt):
            ckpt_dir = alt
        else:
            raise FileNotFoundError(
                f"network_checkpoints not found in {checkpoint_dir} or {alt}"
            )

    flow_path = os.path.join(ckpt_dir, "flow_network.0.pth")
    heat_path = os.path.join(ckpt_dir, "heat_network.0.pth")

    if not os.path.isfile(flow_path):
        raise FileNotFoundError(f"Flow network checkpoint not found: {flow_path}")
    if not os.path.isfile(heat_path):
        raise FileNotFoundError(f"Heat network checkpoint not found: {heat_path}")

    # Load state dicts to infer architecture
    flow_state = torch.load(flow_path, map_location="cpu", weights_only=False)
    heat_state = torch.load(heat_path, map_location="cpu", weights_only=False)

    # Try to use physicsnemo to reconstruct the architecture
    try:
        from physicsnemo.sym.models.fully_connected import FullyConnectedArch
        from physicsnemo.sym.key import Key

        flow_net = FullyConnectedArch(
            input_keys=[Key("x"), Key("y"), Key("t")],
            output_keys=[Key("u"), Key("v"), Key("p")],
        )
        heat_net = FullyConnectedArch(
            input_keys=[Key("x"), Key("y"), Key("t")],
            output_keys=[Key("c")],
        )
        flow_net.load_state_dict(flow_state)
        heat_net.load_state_dict(heat_state)
    except Exception as e:
        warnings.warn(f"Could not load via FullyConnectedArch: {e}. Trying raw state dict approach.")
        # Fallback: build matching linear layers from state dict
        flow_net = _build_net_from_state(flow_state, 3, 3)
        heat_net = _build_net_from_state(heat_state, 3, 1)

    flow_net.eval()
    heat_net.eval()
    return flow_net, heat_net


def _build_net_from_state(state_dict, n_in, n_out):
    """Build a simple sequential model matching the state dict layer shapes."""
    layers = []
    keys = sorted([k for k in state_dict if "weight" in k])
    for i, wkey in enumerate(keys):
        bkey = wkey.replace("weight", "bias")
        w = state_dict[wkey]
        b = state_dict.get(bkey)
        linear = torch.nn.Linear(w.shape[1], w.shape[0])
        linear.weight.data = w
        if b is not None:
            linear.bias.data = b
        layers.append(linear)
        if i < len(keys) - 1:
            layers.append(torch.nn.SiLU())
    return torch.nn.Sequential(*layers)


@torch.no_grad()
def predict(flow_net, heat_net, x, y, t_val):
    """Run inference at given (x, y, t) points. Returns dict of numpy arrays."""
    n = len(x)
    inp = torch.tensor(
        np.column_stack([x, y, np.full(n, t_val)]),
        dtype=torch.float32,
    )

    # Handle both physicsnemo arch and raw sequential
    try:
        flow_out = flow_net({"x": inp[:, 0:1], "y": inp[:, 1:2], "t": inp[:, 2:3]})
        heat_out = heat_net({"x": inp[:, 0:1], "y": inp[:, 1:2], "t": inp[:, 2:3]})
        u = flow_out["u"].numpy().ravel()
        v = flow_out["v"].numpy().ravel()
        p = flow_out["p"].numpy().ravel()
        c = heat_out["c"].numpy().ravel()
    except (TypeError, KeyError):
        flow_out = flow_net(inp)
        heat_out = heat_net(inp)
        u = flow_out[:, 0].numpy()
        v = flow_out[:, 1].numpy()
        p = flow_out[:, 2].numpy()
        c = heat_out[:, 0].numpy()

    return {"u": u, "v": v, "p": p, "c": c}


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_fields_at_time(flow_net, heat_net, t_val, x, y, xx, yy, mask, rects, out_dir):
    """Plot u, v, p, c fields at a given time snapshot."""
    preds = predict(flow_net, heat_net, x, y, t_val)

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


def plot_probe_evolution(flow_net, heat_net, n_time, out_dir):
    """Plot time evolution of all fields at probe points."""
    t_arr = np.linspace(0, T_MAX, n_time)

    fig, axes = plt.subplots(len(FIELDS), 1, figsize=(10, 3 * len(FIELDS)))
    fig.suptitle("Time Evolution at Probe Points", fontsize=14)

    for ax, field in zip(axes, FIELDS):
        for px, py in PROBE_POINTS:
            vals = []
            for t_val in t_arr:
                pred = predict(flow_net, heat_net,
                               np.array([px]), np.array([py]), t_val)
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

def compare_openfoam(flow_net, heat_net, csv_path, out_dir):
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
    preds = predict(flow_net, heat_net, x, y, T_MAX)

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

def physical_trend_checks(flow_net, heat_net):
    """Check that physical trends are correct."""
    checks = []

    # At t=0, speed should be near zero everywhere
    x_pts = np.array([0.0, -2.0, 2.0])
    y_pts = np.array([0.0, 0.0, 0.0])
    pred_t0 = predict(flow_net, heat_net, x_pts, y_pts, 0.0)
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

    # At t=t_max, speed should be established (~1.5 at center)
    pred_tmax = predict(flow_net, heat_net,
                        np.array([0.0]), np.array([0.0]), T_MAX)
    speed_tmax = np.sqrt(pred_tmax["u"][0] ** 2 + pred_tmax["v"][0] ** 2)
    checks.append({
        "name": "Steady: speed at (0,0) at t=t_max > 0.5",
        "value": float(speed_tmax),
        "threshold": 0.5,
        "passed": speed_tmax > 0.5,
    })

    # Speed should increase from t=0 to t=t_max at center
    checks.append({
        "name": "Trend: speed increases from t=0 to t=t_max at (0,0)",
        "value": f"{float(speed_t0[0]):.4f} -> {float(speed_tmax):.4f}",
        "threshold": "monotonic increase",
        "passed": speed_tmax > speed_t0[0],
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
    flow_net, heat_net = load_models(args.checkpoint_dir)
    print("[validate] Models loaded.")

    rects = make_heat_sink_rects()
    x, y, xx, yy, mask = make_domain_grid(args.n_grid, rects)

    # Field plots at snapshots
    print("[validate] Generating field plots...")
    field_plots = []
    for t_val in SNAPSHOT_TIMES:
        print(f"  t = {t_val:.1f}")
        fp = plot_fields_at_time(flow_net, heat_net, t_val, x, y, xx, yy, mask, rects, out_dir)
        field_plots.append(fp)

    # Probe evolution
    print("[validate] Generating probe evolution plots...")
    probe_plot = plot_probe_evolution(flow_net, heat_net, args.n_time, out_dir)

    # OpenFOAM comparison
    errors = None
    if args.openfoam_csv and os.path.isfile(args.openfoam_csv):
        print(f"[validate] Comparing with OpenFOAM: {args.openfoam_csv}")
        errors = compare_openfoam(flow_net, heat_net, args.openfoam_csv, out_dir)
    else:
        print("[validate] No OpenFOAM CSV provided; skipping comparison.")

    # Physical trend checks
    print("[validate] Running physical trend checks...")
    checks = physical_trend_checks(flow_net, heat_net)
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
