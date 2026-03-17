"""
Visualize parameterized heat-sink predictions against the official OpenFOAM
reference and check physical trends across inlet velocities.

This script is intentionally anchored at the only official reference condition
available for the 2D case:
    inlet_vel = 1.5

It produces:
1. OpenFOAM reference plots at inlet_vel = 1.5
2. Parameterized PINN vs OpenFOAM comparison at inlet_vel = 1.5
3. PINN sweep plots at inlet_vel = 1.0 / 1.5 / 2.5
4. A markdown report with trend checks based on basic physics

Example:
    /home/featurize/work/env_conda/nemo/bin/python cases/three_fin_2d/visualize.py \
        --run-dir outputs/latest \
        --out-dir outputs/param_trend_check
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


CASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CASE_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from compare_openfoam_csv import (
    detect_model_type,
    find_artifact_dir,
    load_models,
    load_openfoam_csv,
    run_inference,
)


CHANNEL_X = (-2.5, 2.5)
CHANNEL_Y = (-0.5, 0.5)
FIN_ORIGIN_X = -1.0
FIN_ORIGIN_Y = -0.3
FIN_LENGTH = 1.0
FIN_THICKNESS = 0.1
GAP = 0.25
N_FINS = 3

HEAT_SINK_TEMP = 350.0
BASE_TEMP = 293.498
C_WALL = (HEAT_SINK_TEMP - BASE_TEMP) / 273.15
DEFAULT_CSV_PATH = CASE_DIR / "openfoam" / "heat_sink_zeroEq_Pr5_mesh20.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "param_trend_check"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize parameterized PINN and check inlet-velocity trends."
    )
    p.add_argument(
        "--run-dir",
        default=str(REPO_ROOT / "outputs" / "latest"),
        help="Parameterized run dir, e.g. outputs/latest or outputs/20260317_033129",
    )
    p.add_argument(
        "--csv-path",
        default=str(DEFAULT_CSV_PATH),
        help=f"OpenFOAM CSV path (default: {DEFAULT_CSV_PATH})",
    )
    p.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help=f"Output dir for plots/report (default: {DEFAULT_OUT_DIR})",
    )
    p.add_argument(
        "--velocities",
        nargs="+",
        type=float,
        default=[1.0, 1.5, 2.5],
        help="Velocity sweep to visualize (default: 1.0 1.5 2.5)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=8192,
        help="Inference batch size over OpenFOAM points (default: 8192)",
    )
    p.add_argument(
        "--boundary-strip",
        type=float,
        default=0.15,
        help="Strip width near inlet/outlet used for summary metrics (default: 0.15)",
    )
    return p.parse_args()


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def get_fin_rects() -> list[tuple[float, float, float, float]]:
    rects = []
    y0 = FIN_ORIGIN_Y
    for _ in range(N_FINS):
        rects.append((FIN_ORIGIN_X, y0, FIN_ORIGIN_X + FIN_LENGTH, y0 + FIN_THICKNESS))
        y0 += GAP
    return rects


def points_inside_rects(x: np.ndarray, y: np.ndarray, rects: list[tuple[float, float, float, float]]) -> np.ndarray:
    mask = np.zeros_like(x, dtype=bool)
    for x0, y0, x1, y1 in rects:
        mask |= (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    return mask


def build_triangulation(x: np.ndarray, y: np.ndarray, rects: list[tuple[float, float, float, float]]) -> mtri.Triangulation:
    tri = mtri.Triangulation(x, y)
    cx = x[tri.triangles].mean(axis=1)
    cy = y[tri.triangles].mean(axis=1)
    tri.set_mask(points_inside_rects(cx, cy, rects))
    return tri


def add_fin_patches(ax, rects: list[tuple[float, float, float, float]]) -> None:
    for x0, y0, x1, y1 in rects:
        ax.add_patch(
            patches.Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                linewidth=0.8,
                edgecolor="black",
                facecolor="dimgray",
                alpha=0.85,
                zorder=3,
            )
        )


def to_plot_fields(fields: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = dict(fields)
    out["speed"] = np.sqrt(out["u"] ** 2 + out["v"] ** 2)
    return out


def compute_anchor_metrics(
    pred: dict[str, np.ndarray], ref: dict[str, np.ndarray]
) -> dict[str, dict[str, float]]:
    table: dict[str, dict[str, float]] = {}
    for field in ("u", "v", "p", "c"):
        pv = pred[field]
        rv = ref[field]
        diff = pv - rv
        ref_l2 = float(np.linalg.norm(rv))
        table[field] = {
            "mae": float(np.mean(np.abs(diff))),
            "rmse": float(math.sqrt(np.mean(diff ** 2))),
            "max_abs": float(np.max(np.abs(diff))),
            "rel_l2": float(np.linalg.norm(diff) / ref_l2) if ref_l2 > 0 else float("nan"),
        }
    return table


def compute_summary(
    x: np.ndarray,
    y: np.ndarray,
    fields: dict[str, np.ndarray],
    inlet_vel: float,
    boundary_strip: float,
) -> dict[str, float]:
    speed = np.sqrt(fields["u"] ** 2 + fields["v"] ** 2)
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    inlet_mask = x <= (x_min + boundary_strip)
    outlet_mask = x >= (x_max - boundary_strip)

    if inlet_mask.sum() < 10:
        inlet_mask = x <= np.quantile(x, 0.03)
    if outlet_mask.sum() < 10:
        outlet_mask = x >= np.quantile(x, 0.97)

    return {
        "inlet_vel": inlet_vel,
        "u_max": float(np.max(fields["u"])),
        "speed_max": float(np.max(speed)),
        "speed_mean": float(np.mean(speed)),
        "p_inlet_mean": float(np.mean(fields["p"][inlet_mask])),
        "p_outlet_mean": float(np.mean(fields["p"][outlet_mask])),
        "pressure_drop": float(np.mean(fields["p"][inlet_mask]) - np.mean(fields["p"][outlet_mask])),
        "c_mean": float(np.mean(fields["c"])),
        "c_outlet_mean": float(np.mean(fields["c"][outlet_mask])),
        "c_inlet_mean": float(np.mean(fields["c"][inlet_mask])),
        "c_max": float(np.max(fields["c"])),
    }


def trend_status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def monotonic_increasing(values: list[float], eps: float = 1e-6) -> bool:
    return all(values[i + 1] > values[i] + eps for i in range(len(values) - 1))


def monotonic_decreasing(values: list[float], eps: float = 1e-6) -> bool:
    return all(values[i + 1] < values[i] - eps for i in range(len(values) - 1))


def build_trend_checks(
    sweep_summaries: dict[float, dict[str, float]]
) -> list[dict[str, str]]:
    velocities = sorted(sweep_summaries)
    u_max_values = [sweep_summaries[v]["u_max"] for v in velocities]
    pressure_drop_values = [sweep_summaries[v]["pressure_drop"] for v in velocities]
    c_mean_values = [sweep_summaries[v]["c_mean"] for v in velocities]
    c_out_values = [sweep_summaries[v]["c_outlet_mean"] for v in velocities]

    checks: list[dict[str, str]] = []
    checks.append(
        {
            "name": "Flow acceleration with inlet velocity",
            "expected": "u_max(1.0) < u_max(1.5) < u_max(2.5)",
            "observed": " < ".join(f"{v:.4f}" for v in u_max_values),
            "status": trend_status(monotonic_increasing(u_max_values)),
        }
    )
    checks.append(
        {
            "name": "Pressure drop rises with inlet velocity",
            "expected": "dp(1.0) < dp(1.5) < dp(2.5)",
            "observed": " < ".join(f"{v:.4f}" for v in pressure_drop_values),
            "status": trend_status(monotonic_increasing(pressure_drop_values)),
        }
    )
    checks.append(
        {
            "name": "Mean fluid temperature falls with faster flow",
            "expected": "c_mean(1.0) > c_mean(1.5) > c_mean(2.5)",
            "observed": " > ".join(f"{v:.4f}" for v in c_mean_values),
            "status": trend_status(monotonic_decreasing(c_mean_values)),
        }
    )
    checks.append(
        {
            "name": "Outlet temperature falls with faster flow",
            "expected": "c_out(1.0) > c_out(1.5) > c_out(2.5)",
            "observed": " > ".join(f"{v:.4f}" for v in c_out_values),
            "status": trend_status(monotonic_decreasing(c_out_values)),
        }
    )

    inlet_bc_ok = all(abs(sweep_summaries[v]["c_inlet_mean"]) < 0.02 for v in velocities)
    checks.append(
        {
            "name": "Inlet temperature stays near boundary condition",
            "expected": "|c_inlet_mean| < 0.02 for all velocities",
            "observed": ", ".join(
                f"{v:.1f}:{sweep_summaries[v]['c_inlet_mean']:.4f}" for v in velocities
            ),
            "status": trend_status(inlet_bc_ok),
        }
    )

    pressure_direction_ok = all(
        sweep_summaries[v]["p_inlet_mean"] > sweep_summaries[v]["p_outlet_mean"]
        for v in velocities
    )
    checks.append(
        {
            "name": "Pressure remains higher at inlet than outlet",
            "expected": "p_inlet_mean > p_outlet_mean for all velocities",
            "observed": ", ".join(
                f"{v:.1f}:{sweep_summaries[v]['p_inlet_mean']:.4f}>{sweep_summaries[v]['p_outlet_mean']:.4f}"
                for v in velocities
            ),
            "status": trend_status(pressure_direction_ok),
        }
    )

    gap_acceleration_ok = all(sweep_summaries[v]["u_max"] > v for v in velocities)
    checks.append(
        {
            "name": "Gap acceleration remains physically plausible",
            "expected": "u_max > inlet_vel for all velocities",
            "observed": ", ".join(
                f"{v:.1f}:{sweep_summaries[v]['u_max']:.4f}>{v:.1f}" for v in velocities
            ),
            "status": trend_status(gap_acceleration_ok),
        }
    )

    return checks


def plot_field(ax, tri: mtri.Triangulation, values: np.ndarray, title: str, cmap: str, clim: tuple[float, float] | None, rects) -> None:
    im = ax.tripcolor(tri, values, shading="gouraud", cmap=cmap)
    if clim is not None:
        im.set_clim(*clim)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    add_fin_patches(ax, rects)
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def save_reference_plot(tri, ref_fields, rects, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("OpenFOAM Reference  |  inlet_vel = 1.5", fontsize=13)

    panels = [
        ("speed", "Speed |u| [m/s]", "viridis"),
        ("p", "Pressure", "coolwarm"),
        ("c", f"Normalized Temperature c\nwall BC ≈ {C_WALL:.3f}", "hot"),
    ]

    for ax, (key, title, cmap) in zip(axes, panels):
        values = ref_fields[key]
        clim = (float(np.min(values)), float(np.max(values)))
        plot_field(ax, tri, values, title, cmap, clim, rects)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_anchor_comparison_plot(tri, ref_fields, pred_fields, rects, out_path: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    fig.suptitle("Parameterized PINN vs OpenFOAM  |  inlet_vel = 1.5", fontsize=13)

    panels = [
        ("speed", "Speed |u| [m/s]", "viridis"),
        ("p", "Pressure", "coolwarm"),
        ("c", f"Normalized Temperature c\nwall BC ≈ {C_WALL:.3f}", "hot"),
    ]

    for col, (key, title, cmap) in enumerate(panels):
        ref_vals = ref_fields[key]
        pred_vals = pred_fields[key]
        diff_vals = pred_vals - ref_vals
        clim = (min(float(np.min(ref_vals)), float(np.min(pred_vals))), max(float(np.max(ref_vals)), float(np.max(pred_vals))))
        diff_max = float(np.max(np.abs(diff_vals)))

        plot_field(axes[0, col], tri, ref_vals, f"OpenFOAM 1.5\n{title}", cmap, clim, rects)
        plot_field(axes[1, col], tri, pred_vals, f"PINN 1.5\n{title}", cmap, clim, rects)
        plot_field(
            axes[2, col],
            tri,
            diff_vals,
            f"Difference (PINN - OpenFOAM)\n{title}",
            "bwr",
            (-diff_max, diff_max),
            rects,
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_sweep_plot(
    tri,
    sweep_fields: dict[float, dict[str, np.ndarray]],
    velocities: list[float],
    rects,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(len(velocities), 3, figsize=(15, 4 * len(velocities)))
    if len(velocities) == 1:
        axes = np.array([axes])
    fig.suptitle("Parameterized PINN Sweep", fontsize=13)

    speed_min = min(float(np.min(sweep_fields[v]["speed"])) for v in velocities)
    speed_max = max(float(np.max(sweep_fields[v]["speed"])) for v in velocities)
    p_min = min(float(np.min(sweep_fields[v]["p"])) for v in velocities)
    p_max = max(float(np.max(sweep_fields[v]["p"])) for v in velocities)
    c_min = min(float(np.min(sweep_fields[v]["c"])) for v in velocities)
    c_max = max(float(np.max(sweep_fields[v]["c"])) for v in velocities)

    for row, vel in enumerate(velocities):
        fields = sweep_fields[vel]
        plot_field(
            axes[row, 0],
            tri,
            fields["speed"],
            f"inlet_vel = {vel:.2f}\nSpeed |u| [m/s]",
            "viridis",
            (speed_min, speed_max),
            rects,
        )
        plot_field(
            axes[row, 1],
            tri,
            fields["p"],
            f"inlet_vel = {vel:.2f}\nPressure",
            "coolwarm",
            (p_min, p_max),
            rects,
        )
        plot_field(
            axes[row, 2],
            tri,
            fields["c"],
            f"inlet_vel = {vel:.2f}\nNormalized Temperature c",
            "hot",
            (c_min, c_max),
            rects,
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_trend_plot(
    sweep_summaries: dict[float, dict[str, float]],
    ref_summary: dict[str, float],
    out_path: Path,
) -> None:
    velocities = sorted(sweep_summaries)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("Trend Metrics vs Inlet Velocity", fontsize=13)

    plots = [
        ("u_max", "u_max", "u_max"),
        ("pressure_drop", "Pressure Drop", "p_inlet_mean - p_outlet_mean"),
        ("c_mean", "Mean Temperature c", "c_mean"),
        ("c_outlet_mean", "Outlet Temperature c", "c_outlet_mean"),
    ]

    for ax, (key, title, ylabel) in zip(axes.ravel(), plots):
        ys = [sweep_summaries[v][key] for v in velocities]
        ax.plot(velocities, ys, marker="o", label="Parameterized PINN")
        ax.scatter([1.5], [ref_summary[key]], marker="*", s=180, label="OpenFOAM @1.5")
        ax.set_title(title)
        ax.set_xlabel("inlet_vel")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_report(
    out_dir: Path,
    artifact_dir: Path,
    csv_path: Path,
    velocities: list[float],
    anchor_metrics: dict[str, dict[str, float]],
    ref_summary: dict[str, float],
    sweep_summaries: dict[float, dict[str, float]],
    trend_checks: list[dict[str, str]],
    chart_paths: list[Path],
) -> Path:
    report_path = out_dir / "trend_report.md"
    lines: list[str] = []
    w = lines.append

    w("# Parameterized Heat-Sink Trend Check")
    w("")
    w(f"**Generated:** {datetime.utcnow().isoformat()}Z")
    w(f"**Artifact dir:** `{artifact_dir}`")
    w(f"**CSV path:** `{csv_path}`")
    w(f"**Velocity sweep:** {', '.join(f'{v:.2f}' for v in velocities)}")
    w("")

    w("## What This Checks")
    w("")
    w("- The official OpenFOAM reference only exists for `inlet_vel = 1.5`.")
    w("- So the script uses `1.5` as an anchor condition, then checks whether the")
    w("  parameterized model behaves in a physically reasonable direction at `1.0` and `2.5`.")
    w("- This is a trend check, not a full CFD validation for `1.0` and `2.5`.")
    w("")

    w("## Physical Expectations")
    w("")
    w("- Higher inlet velocity should increase flow speed in the channel and fin gaps.")
    w("- Higher inlet velocity should increase inlet-to-outlet pressure drop.")
    w("- Higher inlet velocity should reduce average fluid temperature and outlet temperature,")
    w("  because stronger convection removes heat more effectively.")
    w("- Inlet temperature should stay near the prescribed boundary condition (`c = 0`).")
    w("")

    w("## Anchor Check at inlet_vel = 1.5")
    w("")
    w("| Field | MAE | RMSE | Max Abs | Rel L2 |")
    w("|-------|-----|------|---------|--------|")
    for field in ("u", "v", "p", "c"):
        row = anchor_metrics[field]
        w(
            f"| {field} | {row['mae']:.4e} | {row['rmse']:.4e} | "
            f"{row['max_abs']:.4e} | {row['rel_l2']:.4e} |"
        )
    w("")

    w("## OpenFOAM Reference Summary at 1.5")
    w("")
    w("| Metric | Value |")
    w("|--------|-------|")
    for key in ("u_max", "pressure_drop", "c_mean", "c_outlet_mean", "c_inlet_mean"):
        w(f"| {key} | {ref_summary[key]:.4f} |")
    w("")

    w("## Sweep Summary")
    w("")
    w("| inlet_vel | u_max | speed_max | pressure_drop | c_mean | c_outlet_mean | c_inlet_mean |")
    w("|-----------|------:|----------:|--------------:|-------:|--------------:|-------------:|")
    for vel in velocities:
        row = sweep_summaries[vel]
        w(
            f"| {vel:.2f} | {row['u_max']:.4f} | {row['speed_max']:.4f} | "
            f"{row['pressure_drop']:.4f} | {row['c_mean']:.4f} | "
            f"{row['c_outlet_mean']:.4f} | {row['c_inlet_mean']:.4f} |"
        )
    w("")

    w("## Trend Checks")
    w("")
    w("| Check | Expected | Observed | Status |")
    w("|-------|----------|----------|--------|")
    for check in trend_checks:
        w(
            f"| {check['name']} | {check['expected']} | {check['observed']} | {check['status']} |"
        )
    w("")

    w("## Charts")
    w("")
    for chart in chart_paths:
        rel = chart.relative_to(out_dir)
        w(f"![{chart.stem}]({rel.as_posix()})")
        w("")

    report_path.write_text("\n".join(lines))
    return report_path


def main() -> None:
    args = parse_args()
    run_dir = resolve_repo_path(args.run_dir)
    csv_path = resolve_repo_path(args.csv_path)
    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    artifact_dir = find_artifact_dir(str(run_dir))
    model_type = detect_model_type(artifact_dir, "auto")
    if model_type != "param":
        raise ValueError(
            f"This script is designed for the parameterized model, but detected model_type='{model_type}'. "
            "Use a parameterized run dir."
        )

    models = load_models(artifact_dir, model_type)
    ref = load_openfoam_csv(str(csv_path))
    rects = get_fin_rects()
    tri = build_triangulation(ref["x"], ref["y"], rects)

    velocities = sorted(set(args.velocities))
    if 1.5 not in velocities:
        velocities.append(1.5)
        velocities = sorted(velocities)

    sweep_raw: dict[float, dict[str, np.ndarray]] = {}
    for vel in velocities:
        sweep_raw[vel] = run_inference(
            models=models,
            x=ref["x"],
            y=ref["y"],
            inlet_vel=vel,
            batch_size=args.batch_size,
        )

    ref_fields = to_plot_fields(ref)
    sweep_fields = {vel: to_plot_fields(raw) for vel, raw in sweep_raw.items()}

    anchor_metrics = compute_anchor_metrics(sweep_raw[1.5], ref)
    ref_summary = compute_summary(
        x=ref["x"],
        y=ref["y"],
        fields=ref,
        inlet_vel=1.5,
        boundary_strip=args.boundary_strip,
    )
    sweep_summaries = {
        vel: compute_summary(
            x=ref["x"],
            y=ref["y"],
            fields=sweep_raw[vel],
            inlet_vel=vel,
            boundary_strip=args.boundary_strip,
        )
        for vel in velocities
    }
    trend_checks = build_trend_checks(sweep_summaries)

    chart_paths = [
        out_dir / "openfoam_reference_1p5.png",
        out_dir / "param_vs_openfoam_1p5.png",
        out_dir / "param_sweep.png",
        out_dir / "trend_metrics.png",
    ]
    save_reference_plot(tri, ref_fields, rects, chart_paths[0])
    save_anchor_comparison_plot(tri, ref_fields, sweep_fields[1.5], rects, chart_paths[1])
    save_sweep_plot(tri, sweep_fields, velocities, rects, chart_paths[2])
    save_trend_plot(sweep_summaries, ref_summary, chart_paths[3])

    report_path = write_report(
        out_dir=out_dir,
        artifact_dir=artifact_dir,
        csv_path=csv_path,
        velocities=velocities,
        anchor_metrics=anchor_metrics,
        ref_summary=ref_summary,
        sweep_summaries=sweep_summaries,
        trend_checks=trend_checks,
        chart_paths=chart_paths,
    )

    print(f"[visualize] artifact_dir : {artifact_dir}")
    print(f"[visualize] csv_path     : {csv_path}")
    print(f"[visualize] out_dir      : {out_dir}")
    print(f"[visualize] velocities   : {', '.join(f'{v:.2f}' for v in velocities)}")
    print("[visualize] anchor @1.5 :")
    for field in ("u", "v", "p", "c"):
        row = anchor_metrics[field]
        print(
            f"  {field}: MAE={row['mae']:.4e} RMSE={row['rmse']:.4e} "
            f"MAX={row['max_abs']:.4e} REL_L2={row['rel_l2']:.4e}"
        )
    print("[visualize] trend checks :")
    for check in trend_checks:
        print(f"  {check['status']:4s}  {check['name']}")
    print(f"[visualize] report       : {report_path}")


if __name__ == "__main__":
    main()
