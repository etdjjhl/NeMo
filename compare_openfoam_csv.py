#!/usr/bin/env python3
"""
Compare a trained heat-sink PINN against the official OpenFOAM CSV reference.

Supports both:
- baseline model: heat_sink.py
- parameterized model: heat_sink_param.py

Example:
    /home/featurize/work/env_conda/nemo/bin/python compare_openfoam_csv.py \
        --run-dir outputs/20260317_033129 \
        --csv-path cases/three_fin_2d/openfoam/heat_sink_zeroEq_Pr5_mesh20.csv \
        --inlet-vel 1.5
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASE_TEMP = 293.498
DEFAULT_CSV_PATH = "cases/three_fin_2d/openfoam/heat_sink_zeroEq_Pr5_mesh20.csv"
FIELDS = ("u", "v", "p", "c")


@dataclass
class LoadedModels:
    flow: object
    heat: object
    device: object
    model_type: str
    artifact_dir: Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare trained PINN outputs against OpenFOAM CSV reference."
    )
    p.add_argument(
        "--run-dir",
        required=True,
        help="Training output dir, e.g. outputs/20260317_033129 or cases/.../outputs/run_param",
    )
    p.add_argument(
        "--csv-path",
        default=DEFAULT_CSV_PATH,
        help=f"Path to OpenFOAM CSV (default: {DEFAULT_CSV_PATH})",
    )
    p.add_argument(
        "--model-type",
        choices=("auto", "baseline", "param"),
        default="auto",
        help="Model type. 'auto' detects from Hydra config when possible.",
    )
    p.add_argument(
        "--inlet-vel",
        type=float,
        default=1.5,
        help="inlet velocity fed to parameterized model (default: 1.5)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=8192,
        help="Inference batch size over CSV points (default: 8192)",
    )
    p.add_argument(
        "--plot-max-points",
        type=int,
        default=20000,
        help="Maximum points to scatter-plot per field (metrics still use all points).",
    )
    p.add_argument(
        "--out-dir",
        default="openfoam_compare",
        help="Directory for markdown report and charts (default: openfoam_compare)",
    )
    return p.parse_args()


def _has_model_files(path: Path) -> bool:
    return bool(list(path.glob("flow_network*.pth"))) and bool(
        list(path.glob("heat_network*.pth"))
    )


def find_artifact_dir(run_dir: str) -> Path:
    root = Path(run_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Run dir does not exist: {root}")

    candidates = [root]
    candidates.extend(p for p in root.rglob("*") if p.is_dir())

    for path in candidates:
        if _has_model_files(path):
            return path

    raise FileNotFoundError(
        f"Could not find model checkpoints under {root}. "
        "Expected files like flow_network.0.pth / heat_network.0.pth."
    )


def detect_model_type(artifact_dir: Path, requested: str) -> str:
    if requested != "auto":
        return requested

    cfg_path = artifact_dir / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        return "baseline"

    try:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(cfg_path)
        custom = cfg.get("custom", None)
        if custom is not None and custom.get("parameterized", False):
            return "param"
    except Exception:
        pass

    return "baseline"


def load_models(artifact_dir: Path, model_type: str) -> LoadedModels:
    import torch
    from omegaconf import OmegaConf
    from physicsnemo.sym.hydra import instantiate_arch
    from physicsnemo.sym.key import Key

    cfg_path = artifact_dir / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Hydra config not found: {cfg_path}")

    cfg = OmegaConf.load(cfg_path)
    if model_type == "param":
        input_keys = [Key("x"), Key("y"), Key("inlet_vel")]
    else:
        input_keys = [Key("x"), Key("y")]

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

    flow_ckpts = sorted(artifact_dir.glob("flow_network*.pth"))
    heat_ckpts = sorted(artifact_dir.glob("heat_network*.pth"))
    if not flow_ckpts or not heat_ckpts:
        raise FileNotFoundError(f"Checkpoint files not found in {artifact_dir}")

    flow_ckpt = flow_ckpts[-1]
    heat_ckpt = heat_ckpts[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    flow_net.load_state_dict(torch.load(flow_ckpt, map_location=device, weights_only=False))
    heat_net.load_state_dict(torch.load(heat_ckpt, map_location=device, weights_only=False))
    flow_net.to(device).eval()
    heat_net.to(device).eval()

    return LoadedModels(
        flow=flow_net,
        heat=heat_net,
        device=device,
        model_type=model_type,
        artifact_dir=artifact_dir,
    )


def load_openfoam_csv(csv_path: str) -> dict[str, np.ndarray]:
    path = Path(csv_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"OpenFOAM CSV not found: {path}\n"
            "Download the supplemental materials from NVIDIA NGC and place "
            f"'heat_sink_zeroEq_Pr5_mesh20.csv' at '{DEFAULT_CSV_PATH}', or pass --csv-path."
        )

    cols: dict[str, list[float]] = {
        "x": [],
        "y": [],
        "u": [],
        "v": [],
        "p": [],
        "sdf": [],
        "nu": [],
        "c_abs": [],
    }

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = ("Points:0", "Points:1", "U:0", "U:1", "p", "T")
        missing = [key for key in required if key not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"CSV missing required columns {missing}. Found columns: {reader.fieldnames}"
            )

        for row in reader:
            cols["x"].append(float(row["Points:0"]))
            cols["y"].append(float(row["Points:1"]))
            cols["u"].append(float(row["U:0"]))
            cols["v"].append(float(row["U:1"]))
            cols["p"].append(float(row["p"]))
            cols["c_abs"].append(float(row["T"]))
            cols["sdf"].append(float(row.get("d", 0.0)))
            cols["nu"].append(float(row.get("nuT", 0.0)))

    arr = {key: np.asarray(value, dtype=np.float32) for key, value in cols.items()}

    # Match the exact preprocessing used in heat_sink.py / heat_sink_param.py.
    arr["c"] = (arr["c_abs"] - BASE_TEMP) / 273.15
    return arr


def run_inference(
    models: LoadedModels,
    x: np.ndarray,
    y: np.ndarray,
    inlet_vel: float,
    batch_size: int,
) -> dict[str, np.ndarray]:
    import torch

    outputs: dict[str, list[np.ndarray]] = {key: [] for key in FIELDS}

    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))
            x_t = torch.as_tensor(x[start:end], dtype=torch.float32, device=models.device).unsqueeze(1)
            y_t = torch.as_tensor(y[start:end], dtype=torch.float32, device=models.device).unsqueeze(1)

            if models.model_type == "param":
                vel_t = torch.full_like(x_t, inlet_vel)
                model_in = {"x": x_t, "y": y_t, "inlet_vel": vel_t}
            else:
                model_in = {"x": x_t, "y": y_t}

            flow_out = models.flow(model_in)
            heat_out = models.heat(model_in)

            outputs["u"].append(flow_out["u"].squeeze(1).cpu().numpy())
            outputs["v"].append(flow_out["v"].squeeze(1).cpu().numpy())
            outputs["p"].append(flow_out["p"].squeeze(1).cpu().numpy())
            outputs["c"].append(heat_out["c"].squeeze(1).cpu().numpy())

    return {key: np.concatenate(chunks) for key, chunks in outputs.items()}


def compute_metrics(pred: dict[str, np.ndarray], ref: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    table: dict[str, dict[str, float]] = {}
    for field in FIELDS:
        pv = pred[field]
        rv = ref[field]
        diff = pv - rv
        ref_l2 = float(np.linalg.norm(rv))
        table[field] = {
            "mae": float(np.mean(np.abs(diff))),
            "rmse": float(math.sqrt(np.mean(diff ** 2))),
            "max_abs": float(np.max(np.abs(diff))),
            "rel_l2": float(np.linalg.norm(diff) / ref_l2) if ref_l2 > 0 else float("nan"),
            "pred_mean": float(np.mean(pv)),
            "ref_mean": float(np.mean(rv)),
        }
    return table


def _plot_indices(n: int, plot_max_points: int) -> np.ndarray:
    if n <= plot_max_points:
        return np.arange(n)
    rng = np.random.default_rng(42)
    return np.sort(rng.choice(n, size=plot_max_points, replace=False))


def save_spatial_plot(
    x: np.ndarray,
    y: np.ndarray,
    pred: np.ndarray,
    ref: np.ndarray,
    field: str,
    out_path: Path,
    plot_max_points: int,
) -> None:
    idx = _plot_indices(len(x), plot_max_points)
    xs = x[idx]
    ys = y[idx]
    pv = pred[idx]
    rv = ref[idx]
    dv = pv - rv

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"{field}: PINN vs OpenFOAM", fontsize=13)

    vmin = min(float(np.min(pv)), float(np.min(rv)))
    vmax = max(float(np.max(pv)), float(np.max(rv)))

    sc0 = axes[0].scatter(xs, ys, c=pv, s=4, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    axes[0].set_title("PINN prediction")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    plt.colorbar(sc0, ax=axes[0])

    sc1 = axes[1].scatter(xs, ys, c=rv, s=4, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    axes[1].set_title("OpenFOAM reference")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    plt.colorbar(sc1, ax=axes[1])

    abs_max = float(np.max(np.abs(dv)))
    sc2 = axes[2].scatter(xs, ys, c=dv, s=4, cmap="bwr", vmin=-abs_max, vmax=abs_max)
    axes[2].set_title("Difference (PINN - OpenFOAM)")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    plt.colorbar(sc2, ax=axes[2])

    for ax in axes:
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def save_parity_plot(
    pred: np.ndarray,
    ref: np.ndarray,
    field: str,
    out_path: Path,
    plot_max_points: int,
) -> None:
    idx = _plot_indices(len(pred), plot_max_points)
    pv = pred[idx]
    rv = ref[idx]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(rv, pv, s=4, alpha=0.45)

    lo = min(float(np.min(rv)), float(np.min(pv)))
    hi = max(float(np.max(rv)), float(np.max(pv)))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0, linestyle="--")
    ax.set_title(f"{field}: parity plot")
    ax.set_xlabel("OpenFOAM")
    ax.set_ylabel("PINN")
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def write_report(
    out_dir: Path,
    models: LoadedModels,
    csv_path: str,
    inlet_vel: float,
    n_points: int,
    metrics: dict[str, dict[str, float]],
    charts: list[Path],
) -> Path:
    report_path = out_dir / "comparison_report.md"
    lines: list[str] = []
    w = lines.append

    w("# PINN vs OpenFOAM — Comparison Report")
    w("")
    w(f"**Generated:** {datetime.utcnow().isoformat()}Z")
    w(f"**Model type:** `{models.model_type}`")
    w(f"**Artifact dir:** `{models.artifact_dir}`")
    w(f"**CSV path:** `{Path(csv_path).expanduser().resolve()}`")
    w(f"**Reference points:** {n_points}")
    if models.model_type == "param":
        w(f"**inlet_vel:** {inlet_vel:.4f}")
    w("")

    w("## Notes")
    w("")
    w("- `c` is compared in the same normalized space used by the training validator:")
    w(f"  `c = (T - {BASE_TEMP}) / 273.15`.")
    w("- Metrics are computed on all CSV points.")
    w("- Spatial plots may be downsampled for readability, but the report table is not.")
    w("")

    w("## Metrics")
    w("")
    w("| Field | MAE | RMSE | Max Abs | Rel L2 | PINN mean | OpenFOAM mean |")
    w("|-------|-----|------|---------|--------|-----------|---------------|")
    for field in FIELDS:
        row = metrics[field]
        w(
            f"| {field} | {row['mae']:.4e} | {row['rmse']:.4e} | {row['max_abs']:.4e} | "
            f"{row['rel_l2']:.4e} | {row['pred_mean']:.4e} | {row['ref_mean']:.4e} |"
        )
    w("")

    w("## Charts")
    w("")
    for chart in charts:
        rel = chart.relative_to(out_dir)
        w(f"![{chart.stem}]({rel.as_posix()})")
        w("")

    report_path.write_text("\n".join(lines))
    return report_path


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    artifact_dir = find_artifact_dir(args.run_dir)
    model_type = detect_model_type(artifact_dir, args.model_type)
    models = load_models(artifact_dir, model_type)
    ref = load_openfoam_csv(args.csv_path)
    pred = run_inference(
        models=models,
        x=ref["x"],
        y=ref["y"],
        inlet_vel=args.inlet_vel,
        batch_size=args.batch_size,
    )
    metrics = compute_metrics(pred, ref)

    charts: list[Path] = []
    for field in FIELDS:
        spatial_path = out_dir / f"{field}_spatial.png"
        parity_path = out_dir / f"{field}_parity.png"
        save_spatial_plot(
            x=ref["x"],
            y=ref["y"],
            pred=pred[field],
            ref=ref[field],
            field=field,
            out_path=spatial_path,
            plot_max_points=args.plot_max_points,
        )
        save_parity_plot(
            pred=pred[field],
            ref=ref[field],
            field=field,
            out_path=parity_path,
            plot_max_points=args.plot_max_points,
        )
        charts.extend([spatial_path, parity_path])

    report_path = write_report(
        out_dir=out_dir,
        models=models,
        csv_path=args.csv_path,
        inlet_vel=args.inlet_vel,
        n_points=len(ref["x"]),
        metrics=metrics,
        charts=charts,
    )

    print(f"[compare] artifact_dir: {artifact_dir}")
    print(f"[compare] model_type : {model_type}")
    print(f"[compare] csv_path   : {Path(args.csv_path).expanduser().resolve()}")
    print(f"[compare] points     : {len(ref['x'])}")
    if model_type == "param":
        print(f"[compare] inlet_vel  : {args.inlet_vel}")
    for field in FIELDS:
        row = metrics[field]
        print(
            f"[compare] {field}: "
            f"MAE={row['mae']:.4e} RMSE={row['rmse']:.4e} "
            f"MAX={row['max_abs']:.4e} REL_L2={row['rel_l2']:.4e}"
        )
    print(f"[compare] report     : {report_path}")


if __name__ == "__main__":
    main()
