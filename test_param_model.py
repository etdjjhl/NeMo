#!/usr/bin/env python3
"""
test_param_model.py — Compare parameterized PINN vs baseline PINN.

At a fixed inlet_vel, both models should produce similar fields.
Computes per-field MAE and generates comparison plots.

Usage:
    python test_param_model.py \
        --param-dir outputs/latest \
        --base-dir  outputs/20260226_094257 \
        --case-dir  cases/three_fin_2d \
        --inlet-vel 1.5
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare param vs baseline PINN models")
    p.add_argument("--param-dir", required=True,
                   help="Output dir of parameterized run (contains hydra_outputs/)")
    p.add_argument("--base-dir",  required=True,
                   help="Output dir of baseline run (contains hydra_outputs/)")
    p.add_argument("--case-dir",  required=True,
                   help="Path to cases/three_fin_2d/")
    p.add_argument("--inlet-vel", type=float, default=1.5,
                   help="inlet_vel value to query both models at (default 1.5)")
    p.add_argument("--n-points",  type=int, default=500,
                   help="Number of test points sampled from domain interior (default 500)")
    p.add_argument("--out-dir",   default="test_results",
                   help="Output directory for report and plots (default test_results)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint discovery
# ─────────────────────────────────────────────────────────────────────────────

def find_checkpoint_dir(run_dir: str) -> str | None:
    """Search for the network checkpoint directory under run_dir."""
    candidates = [
        os.path.join(run_dir, "hydra_outputs", "network_checkpoint"),
        os.path.join(run_dir, "hydra_outputs"),
        os.path.join(run_dir, "network_checkpoint"),
    ]
    # Also search recursively for any directory named network_checkpoint
    for root, dirs, _ in os.walk(run_dir):
        for d in dirs:
            if d == "network_checkpoint":
                candidates.append(os.path.join(root, d))

    for c in candidates:
        if os.path.isdir(c):
            # Check it contains at least one .pth file
            pth_files = list(Path(c).glob("*.pth"))
            if pth_files:
                return c
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model_from_checkpoint(ckpt_dir: str, is_param: bool):
    """Load flow_network and heat_network from a checkpoint directory.

    Returns a dict: {"flow": model, "heat": model} or raises.
    """
    import torch
    from physicsnemo.sym.hydra import instantiate_arch
    from physicsnemo.sym.key import Key
    from omegaconf import OmegaConf

    # Load saved arch config if available, else use defaults
    arch_cfg_path = os.path.join(ckpt_dir, "..", "cfg.yaml")

    # Build minimal arch config using OmegaConf
    arch_defaults = OmegaConf.create({
        "fully_connected": {
            "_target_": "physicsnemo.sym.models.fully_connected.FullyConnectedArch",
            "layer_size": 512,
            "nr_layers": 6,
            "skip_connections": False,
            "activation_fn": "silu",
            "adaptive_activations": False,
            "weight_norm": True,
        }
    })

    if is_param:
        input_keys_flow = [Key("x"), Key("y"), Key("inlet_vel")]
        input_keys_heat = [Key("x"), Key("y"), Key("inlet_vel")]
    else:
        input_keys_flow = [Key("x"), Key("y")]
        input_keys_heat = [Key("x"), Key("y")]

    flow_net = instantiate_arch(
        input_keys=input_keys_flow,
        output_keys=[Key("u"), Key("v"), Key("p")],
        cfg=arch_defaults.fully_connected,
    )
    heat_net = instantiate_arch(
        input_keys=input_keys_heat,
        output_keys=[Key("c")],
        cfg=arch_defaults.fully_connected,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Try to load weights
    flow_ckpt = os.path.join(ckpt_dir, "flow_network.pth")
    heat_ckpt = os.path.join(ckpt_dir, "heat_network.pth")

    if not os.path.isfile(flow_ckpt):
        raise FileNotFoundError(f"flow_network.pth not found in {ckpt_dir}")
    if not os.path.isfile(heat_ckpt):
        raise FileNotFoundError(f"heat_network.pth not found in {ckpt_dir}")

    flow_net.load_state_dict(torch.load(flow_ckpt, map_location=device))
    heat_net.load_state_dict(torch.load(heat_ckpt, map_location=device))

    flow_net.to(device).eval()
    heat_net.to(device).eval()

    return {"flow": flow_net, "heat": heat_net, "device": device}


# ─────────────────────────────────────────────────────────────────────────────
# Test point generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_test_points(case_dir: str, n_points: int):
    """Sample interior points from the heat-sink geometry.

    Falls back to a simple grid if geometry import fails.
    """
    try:
        sys.path.insert(0, case_dir)
        from physicsnemo.sym.geometry.primitives_2d import Rectangle, Channel2D
        from physicsnemo.sym.geometry import Parameterization

        channel_length = (-2.5, 2.5)
        channel_width = (-0.5, 0.5)
        heat_sink_origin = (-1, -0.3)
        nr_heat_sink_fins = 3
        gap = 0.15 + 0.1
        heat_sink_length = 1.0
        heat_sink_fin_thickness = 0.1

        channel = Channel2D(
            (channel_length[0], channel_width[0]),
            (channel_length[1], channel_width[1])
        )
        heat_sink = Rectangle(
            heat_sink_origin,
            (
                heat_sink_origin[0] + heat_sink_length,
                heat_sink_origin[1] + heat_sink_fin_thickness,
            ),
        )
        for i in range(1, nr_heat_sink_fins):
            heat_sink_origin = (heat_sink_origin[0], heat_sink_origin[1] + gap)
            fin = Rectangle(
                heat_sink_origin,
                (
                    heat_sink_origin[0] + heat_sink_length,
                    heat_sink_origin[1] + heat_sink_fin_thickness,
                ),
            )
            heat_sink = heat_sink + fin
        geo = channel - heat_sink

        pts = geo.sample_interior(n_points)
        x = pts["x"].flatten()
        y = pts["y"].flatten()
        print(f"[test] Sampled {len(x)} interior points from geometry.")
        return x, y

    except Exception as e:
        warnings.warn(f"Geometry sampling failed ({e}); using random grid fallback.")
        # Simple fallback: random points in channel bounding box
        rng = np.random.default_rng(42)
        x = rng.uniform(-2.5, 2.5, n_points)
        y = rng.uniform(-0.5, 0.5, n_points)
        return x, y


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(models: dict, x: np.ndarray, y: np.ndarray,
                  inlet_vel: float, is_param: bool) -> dict:
    """Run forward pass and return field arrays."""
    import torch

    device = models["device"]
    x_t = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(1)
    y_t = torch.tensor(y, dtype=torch.float32, device=device).unsqueeze(1)

    with torch.no_grad():
        if is_param:
            vel_t = torch.full_like(x_t, inlet_vel)
            flow_in = {"x": x_t, "y": y_t, "inlet_vel": vel_t}
            heat_in = {"x": x_t, "y": y_t, "inlet_vel": vel_t}
        else:
            flow_in = {"x": x_t, "y": y_t}
            heat_in = {"x": x_t, "y": y_t}

        flow_out = models["flow"](flow_in)
        heat_out = models["heat"](heat_in)

    result = {}
    for k, v in {**flow_out, **heat_out}.items():
        result[k] = v.squeeze(1).cpu().numpy()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def scatter_comparison(x, y, vals_param, vals_base, field_name: str,
                       out_path: str, inlet_vel: float) -> None:
    """Side-by-side scatter plots for param vs baseline, plus difference."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(
        f"Field: {field_name}  |  inlet_vel = {inlet_vel:.2f} m/s",
        fontsize=13
    )

    vmin = min(vals_param.min(), vals_base.min())
    vmax = max(vals_param.max(), vals_base.max())

    sc0 = axes[0].scatter(x, y, c=vals_param, cmap="RdBu_r",
                          vmin=vmin, vmax=vmax, s=6)
    axes[0].set_title("Parameterized model")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
    plt.colorbar(sc0, ax=axes[0])

    sc1 = axes[1].scatter(x, y, c=vals_base, cmap="RdBu_r",
                          vmin=vmin, vmax=vmax, s=6)
    axes[1].set_title("Baseline model")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("y")
    plt.colorbar(sc1, ax=axes[1])

    diff = vals_param - vals_base
    abs_max = np.abs(diff).max()
    sc2 = axes[2].scatter(x, y, c=diff, cmap="bwr",
                          vmin=-abs_max, vmax=abs_max, s=6)
    axes[2].set_title("Difference (param − base)")
    axes[2].set_xlabel("x"); axes[2].set_ylabel("y")
    plt.colorbar(sc2, ax=axes[2])

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[test] Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def write_report(out_dir: str, mae_table: dict, param_dir: str,
                 base_dir: str, inlet_vel: float, charts: list[str]) -> str:
    from datetime import datetime

    report_path = os.path.join(out_dir, "comparison_report.md")
    lines: list[str] = []
    w = lines.append

    w("# Parameterized vs Baseline PINN — Comparison Report")
    w("")
    w(f"**Generated:** {datetime.utcnow().isoformat()}Z")
    w(f"**inlet_vel (query point):** {inlet_vel:.2f} m/s")
    w("")

    w("## Inputs")
    w("")
    w(f"- **Parameterized model dir:** `{os.path.abspath(param_dir)}`")
    w(f"- **Baseline model dir:** `{os.path.abspath(base_dir)}`")
    w("")

    w("## MAE Summary")
    w("")
    w("| Field | MAE | Param mean | Base mean |")
    w("|-------|-----|-----------|----------|")
    for field, stats in mae_table.items():
        w(f"| {field} | {stats['mae']:.4e} | {stats['param_mean']:.4e} | {stats['base_mean']:.4e} |")
    w("")

    w("## Notes")
    w("")
    w("- Both models queried at the **same** `inlet_vel` value.")
    w("- Low MAE indicates the parameterized model has learned a consistent")
    w("  representation of the baseline solution.")
    w("- At early training (10k steps) models are not converged — MAE values")
    w("  may be large. Run with `MAX_STEPS=300000` for meaningful comparison.")
    w("")

    w("## Charts")
    w("")
    for c in charts:
        rel = os.path.relpath(c, out_dir)
        stem = Path(c).stem
        w(f"![{stem}]({rel})")
        w("")

    text = "\n".join(lines)
    with open(report_path, "w") as f:
        f.write(text)
    return report_path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[test] Parameterized run dir : {args.param_dir}")
    print(f"[test] Baseline run dir      : {args.base_dir}")
    print(f"[test] inlet_vel query       : {args.inlet_vel}")
    print(f"[test] Output dir            : {out_dir}")

    # ── Find checkpoints ──────────────────────────────────────────────────────
    param_ckpt = find_checkpoint_dir(args.param_dir)
    base_ckpt  = find_checkpoint_dir(args.base_dir)

    if param_ckpt is None:
        sys.exit(f"[test] ERROR: No checkpoint found under {args.param_dir}")
    if base_ckpt is None:
        sys.exit(f"[test] ERROR: No checkpoint found under {args.base_dir}")

    print(f"[test] Param checkpoint: {param_ckpt}")
    print(f"[test] Base  checkpoint: {base_ckpt}")

    # ── Load models ───────────────────────────────────────────────────────────
    print("[test] Loading parameterized model...")
    param_models = load_model_from_checkpoint(param_ckpt, is_param=True)

    print("[test] Loading baseline model...")
    base_models  = load_model_from_checkpoint(base_ckpt,  is_param=False)

    # ── Sample test points ────────────────────────────────────────────────────
    print(f"[test] Generating {args.n_points} test points...")
    x, y = generate_test_points(args.case_dir, args.n_points)

    # ── Run inference ─────────────────────────────────────────────────────────
    print("[test] Running parameterized model inference...")
    param_out = run_inference(param_models, x, y, args.inlet_vel, is_param=True)

    print("[test] Running baseline model inference...")
    base_out  = run_inference(base_models,  x, y, args.inlet_vel, is_param=False)

    # ── Compute MAE ───────────────────────────────────────────────────────────
    common_fields = sorted(set(param_out.keys()) & set(base_out.keys()))
    print(f"[test] Common output fields: {common_fields}")

    mae_table: dict = {}
    for field in common_fields:
        pv = param_out[field]
        bv = base_out[field]
        mae = float(np.mean(np.abs(pv - bv)))
        mae_table[field] = {
            "mae": mae,
            "param_mean": float(np.mean(pv)),
            "base_mean":  float(np.mean(bv)),
        }
        print(f"  {field:6s}: MAE={mae:.4e}  param_mean={mae_table[field]['param_mean']:.4e}"
              f"  base_mean={mae_table[field]['base_mean']:.4e}")

    # ── Generate comparison plots ─────────────────────────────────────────────
    charts: list[str] = []
    for field in ["u", "c"]:
        if field not in param_out or field not in base_out:
            print(f"[test] Field '{field}' not found in outputs, skipping plot.")
            continue
        out_path = os.path.join(out_dir, f"field_{field}_comparison.png")
        scatter_comparison(
            x, y,
            param_out[field], base_out[field],
            field_name=field,
            out_path=out_path,
            inlet_vel=args.inlet_vel,
        )
        charts.append(out_path)

    # ── Write report ──────────────────────────────────────────────────────────
    report_path = write_report(
        out_dir, mae_table, args.param_dir, args.base_dir, args.inlet_vel, charts
    )

    print("")
    print("=" * 50)
    print(f"  Test complete!")
    print(f"  Report : {report_path}")
    for c in charts:
        print(f"  Chart  : {c}")
    print("=" * 50)


if __name__ == "__main__":
    main()
