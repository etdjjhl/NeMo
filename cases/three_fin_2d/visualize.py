"""
Visualize flow and temperature fields from trained PINN checkpoints.
Run from cases/three_fin_2d/:
    python visualize.py [inlet_vel]
Default inlet_vel = 1.5
"""
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from physicsnemo.sym.models.fully_connected import FullyConnectedArch
from physicsnemo.sym.key import Key

# ── Geometry constants (must match heat_sink_param.py) ──────────────────────
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
C_WALL = (HEAT_SINK_TEMP - BASE_TEMP) / 273.15   # ≈ 0.207

CHECKPOINT_DIR = "outputs/run_param"


# ── Geometry helpers ─────────────────────────────────────────────────────────
def get_fin_rects():
    rects = []
    y0 = FIN_ORIGIN_Y
    for _ in range(N_FINS):
        rects.append((FIN_ORIGIN_X, y0, FIN_ORIGIN_X + FIN_LENGTH, y0 + FIN_THICKNESS))
        y0 += GAP
    return rects


def inside_fin(X, Y, rects):
    mask = np.zeros(X.shape, dtype=bool)
    for (x0, y0, x1, y1) in rects:
        mask |= (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
    return mask


def add_fin_patches(ax, rects):
    for (x0, y0, x1, y1) in rects:
        ax.add_patch(patches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            linewidth=0.8, edgecolor="black", facecolor="dimgray", alpha=0.85, zorder=3
        ))


# ── Load networks ────────────────────────────────────────────────────────────
def load_networks():
    flow_net = FullyConnectedArch(
        input_keys=[Key("x"), Key("y"), Key("inlet_vel")],
        output_keys=[Key("u"), Key("v"), Key("p")],
    )
    heat_net = FullyConnectedArch(
        input_keys=[Key("x"), Key("y"), Key("inlet_vel")],
        output_keys=[Key("c")],
    )
    flow_state = torch.load(f"{CHECKPOINT_DIR}/flow_network.0.pth",
                            map_location="cpu", weights_only=False)
    heat_state = torch.load(f"{CHECKPOINT_DIR}/heat_network.0.pth",
                            map_location="cpu", weights_only=False)
    flow_net.load_state_dict(flow_state)
    heat_net.load_state_dict(heat_state)
    flow_net.eval()
    heat_net.eval()
    return flow_net, heat_net


# ── Inference ────────────────────────────────────────────────────────────────
@torch.no_grad()
def infer(flow_net, heat_net, inlet_vel, nx=300, ny=100):
    xs = np.linspace(CHANNEL_X[0], CHANNEL_X[1], nx)
    ys = np.linspace(CHANNEL_Y[0], CHANNEL_Y[1], ny)
    X, Y = np.meshgrid(xs, ys)
    rects = get_fin_rects()
    fluid_mask = ~inside_fin(X, Y, rects)

    x_t = torch.tensor(X.ravel(), dtype=torch.float32).unsqueeze(1)
    y_t = torch.tensor(Y.ravel(), dtype=torch.float32).unsqueeze(1)
    v_t = torch.full_like(x_t, inlet_vel)

    flow_out = flow_net({"x": x_t, "y": y_t, "inlet_vel": v_t})
    heat_out = heat_net({"x": x_t, "y": y_t, "inlet_vel": v_t})

    def reshape(t): return t.squeeze(1).numpy().reshape(ny, nx)

    U = reshape(flow_out["u"])
    V = reshape(flow_out["v"])
    P = reshape(flow_out["p"])
    C = reshape(heat_out["c"])

    for arr in (U, V, P, C):
        arr[~fluid_mask] = np.nan

    return X, Y, U, V, P, C, rects, fluid_mask


# ── Summary ──────────────────────────────────────────────────────────────────
def print_summary(U, V, P, C, inlet_vel):
    print(f"\n=== Sanity Check (inlet_vel={inlet_vel:.2f}) ===")
    print(f"  u_max      = {np.nanmax(U):.4f}   (expect > {inlet_vel:.2f}, gap acceleration)")
    p_in  = np.nanmean(P[:, :8])
    p_out = np.nanmean(P[:, -8:])
    print(f"  p_inlet    = {p_in:.4f}   p_outlet = {p_out:.4f}   (expect inlet > outlet)")
    print(f"  c_max      = {np.nanmax(C):.4f}   (expect ≈ {C_WALL:.4f}, fin wall BC)")
    print(f"  c_inlet    = {np.nanmean(C[:, :8]):.4f}   (expect ≈ 0, inlet BC)")


# ── Plot ─────────────────────────────────────────────────────────────────────
def plot_fields(X, Y, U, V, P, C, rects, inlet_vel):
    speed = np.sqrt(U**2 + V**2)

    fig, axes = plt.subplots(2, 2, figsize=(15, 7))
    fig.suptitle(f"PINN Inference  |  inlet_vel = {inlet_vel:.2f} m/s", fontsize=13)

    panels = [
        (axes[0, 0], U,     "u  (x-velocity [m/s])",    "RdBu_r",  None),
        (axes[0, 1], speed, "Speed |u| [m/s]",           "viridis",  None),
        (axes[1, 0], P,     "p  (pressure)",              "coolwarm", None),
        (axes[1, 1], C,     f"c  (norm. temperature)\nwall BC ≈ {C_WALL:.3f}", "hot", None),
    ]

    for ax, field, title, cmap, clim in panels:
        im = ax.pcolormesh(X, Y, field, cmap=cmap, shading="auto")
        if clim:
            im.set_clim(*clim)
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        add_fin_patches(ax, rects)
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    # streamlines on speed panel
    ax_s = axes[0, 1]
    xs_1d = np.linspace(CHANNEL_X[0], CHANNEL_X[1], X.shape[1])
    ys_1d = np.linspace(CHANNEL_Y[0], CHANNEL_Y[1], X.shape[0])
    try:
        ax_s.streamplot(xs_1d, ys_1d,
                        np.where(np.isnan(U), 0, U),
                        np.where(np.isnan(V), 0, V),
                        color="white", linewidth=0.5, density=1.2,
                        arrowsize=0.6, zorder=4)
    except Exception:
        pass

    plt.tight_layout()
    out_path = f"field_vel{inlet_vel:.1f}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    inlet_vel = float(sys.argv[1]) if len(sys.argv) > 1 else 1.5

    print(f"Loading checkpoints from {CHECKPOINT_DIR} ...")
    flow_net, heat_net = load_networks()
    print("Networks loaded OK.")

    print(f"Running inference at inlet_vel = {inlet_vel} ...")
    X, Y, U, V, P, C, rects, fluid_mask = infer(flow_net, heat_net, inlet_vel)

    print_summary(U, V, P, C, inlet_vel)
    plot_fields(X, Y, U, V, P, C, rects, inlet_vel)
