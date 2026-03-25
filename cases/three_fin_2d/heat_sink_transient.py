# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Transient (time-parameterized) variant: the model learns the full temporal
# evolution from quiescent initial conditions (t=0) to steady state (t=10).
# All equation classes use time=True to include temporal derivative terms.

import os
import warnings

import torch
import numpy as np
from sympy import Symbol, exp

import physicsnemo.sym
from physicsnemo.sym.hydra import to_absolute_path, instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.geometry.primitives_2d import Rectangle, Line, Channel2D
from physicsnemo.sym.utils.sympy.functions import parabola
from physicsnemo.sym.utils.io import csv_to_dict
from physicsnemo.sym.eq.pdes.navier_stokes import NavierStokes, GradNormal
from physicsnemo.sym.eq.pdes.basic import NormalDotVec
from physicsnemo.sym.eq.pdes.turbulence_zero_eq import ZeroEquation
from physicsnemo.sym.eq.pdes.advection_diffusion import AdvectionDiffusion
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint,
)
from physicsnemo.sym.domain.monitor import PointwiseMonitor
from physicsnemo.sym.domain.validator import PointwiseValidator
from physicsnemo.sym.key import Key
from physicsnemo.sym.geometry import Parameterization, Parameter


@physicsnemo.sym.main(config_path="conf_transient", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    # ── Domain parameters ─────────────────────────────────────────────────────
    channel_length = (-2.5, 2.5)
    channel_width = (-0.5, 0.5)
    heat_sink_origin = (-1, -0.3)
    nr_heat_sink_fins = 3
    gap = 0.15 + 0.1
    heat_sink_length = 1.0
    heat_sink_fin_thickness = 0.1
    inlet_vel = 1.5
    heat_sink_temp = 350
    base_temp = 293.498
    nu = 0.01
    diffusivity = 0.01 / 5

    # ── Time parameterization ─────────────────────────────────────────────────
    t_max = 10.0
    t_symbol = Symbol("t")
    time_range = {t_symbol: (0.0, t_max)}

    # Smooth ramp function: r(t) = 1 - exp(-t/tau)
    # r(0) = 0 exactly (compatible with IC), r(3*tau) ≈ 0.95, r(5*tau) ≈ 0.99
    # Default tau=0.4 → ~95% at t=1.2s, ~99.3% at t=2s, fully steady well before t_max
    tau = cfg.custom.get("ramp_tau", 0.4)
    ramp = 1 - exp(-t_symbol / tau)

    # ── Sympy spatial variables ───────────────────────────────────────────────
    _x, y = Symbol("x"), Symbol("y")

    # ── Geometry ─────────────────────────────────────────────────────────────
    channel = Channel2D(
        (channel_length[0], channel_width[0]), (channel_length[1], channel_width[1])
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

    inlet = Line(
        (channel_length[0], channel_width[0]), (channel_length[0], channel_width[1]), -1
    )
    outlet = Line(
        (channel_length[1], channel_width[0]), (channel_length[1], channel_width[1]), 1
    )

    x_pos = Parameter("x_pos")
    integral_line = Line(
        (x_pos, channel_width[0]),
        (x_pos, channel_width[1]),
        1,
        parameterization=Parameterization({x_pos: channel_length}),
    )

    # ── Equations (time=True for transient terms) ────────────────────────────
    ze = ZeroEquation(
        nu=nu, rho=1.0, dim=2, max_distance=(channel_width[1] - channel_width[0]) / 2,
        time=True,
    )
    ns = NavierStokes(nu=ze.equations["nu"], rho=1.0, dim=2, time=True)
    ade = AdvectionDiffusion(T="c", rho=1.0, D=diffusivity, dim=2, time=True)
    gn_c = GradNormal("c", dim=2, time=True)
    normal_dot_vel = NormalDotVec(["u", "v"])

    # Networks receive t as an additional input coordinate
    flow_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("t")],
        output_keys=[Key("u"), Key("v"), Key("p")],
        cfg=cfg.arch.fully_connected,
    )
    heat_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("t")],
        output_keys=[Key("c")],
        cfg=cfg.arch.fully_connected,
    )

    nodes = (
        ns.make_nodes()
        + ze.make_nodes()
        + ade.make_nodes(detach_names=["u", "v"])
        + gn_c.make_nodes()
        + normal_dot_vel.make_nodes()
        + [flow_net.make_node(name="flow_network")]
        + [heat_net.make_node(name="heat_network")]
    )

    # ── Domain ────────────────────────────────────────────────────────────────
    domain = Domain()

    # ── BCs (applied for all t in [0, t_max]) ─────────────────────────────────

    # inlet — parabolic profile with smooth ramp-up: u(y,t) = parabola(y) * ramp(t)
    # At t=0: ramp=0 → u=0, compatible with IC (fluid at rest)
    # As t→∞: ramp→1 → u=parabola(y)*inlet_vel, same as steady-state BC
    inlet_parabola = parabola(
        y, inter_1=channel_width[0], inter_2=channel_width[1], height=inlet_vel
    )
    inlet_constraint = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=inlet,
        outvar={"u": inlet_parabola * ramp, "v": 0, "c": 0},
        batch_size=cfg.batch_size.inlet,
        parameterization=time_range,
    )
    domain.add_constraint(inlet_constraint, "inlet")

    # outlet
    outlet_constraint = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=outlet,
        outvar={"p": 0},
        batch_size=cfg.batch_size.outlet,
        parameterization=time_range,
    )
    domain.add_constraint(outlet_constraint, "outlet")

    # heat_sink wall — temperature ramp-up: c(t) = c_wall * ramp(t)
    # At t=0: ramp=0 → c=0, compatible with IC (ambient temperature)
    # As t→∞: ramp→1 → c=c_wall, same as steady-state BC
    c_wall = (heat_sink_temp - base_temp) / 273.15
    hs_wall = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=heat_sink,
        outvar={"u": 0, "v": 0, "c": c_wall * ramp},
        batch_size=cfg.batch_size.hs_wall,
        parameterization=time_range,
    )
    domain.add_constraint(hs_wall, "heat_sink_wall")

    # channel wall
    channel_wall = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=channel,
        outvar={"u": 0, "v": 0, "normal_gradient_c": 0},
        batch_size=cfg.batch_size.channel_wall,
        parameterization=time_range,
    )
    domain.add_constraint(channel_wall, "channel_wall")

    # ── Interior PDE constraints ──────────────────────────────────────────────

    # interior flow
    interior_flow = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0},
        batch_size=cfg.batch_size.interior_flow,
        compute_sdf_derivatives=True,
        lambda_weighting={
            "continuity": Symbol("sdf"),
            "momentum_x": Symbol("sdf"),
            "momentum_y": Symbol("sdf"),
        },
        parameterization=time_range,
    )
    domain.add_constraint(interior_flow, "interior_flow")

    # interior heat
    interior_heat = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"advection_diffusion_c": 0},
        batch_size=cfg.batch_size.interior_heat,
        lambda_weighting={
            "advection_diffusion_c": 1.0,
        },
        parameterization=time_range,
    )
    domain.add_constraint(interior_heat, "interior_heat")

    # ── Initial conditions (t=0): fluid at rest ───────────────────────────────

    ic_flow = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u": 0, "v": 0, "p": 0},
        batch_size=cfg.batch_size.ic_flow,
        lambda_weighting={
            "u": 10.0,
            "v": 10.0,
            "p": 10.0,
        },
        parameterization={t_symbol: 0.0},
    )
    domain.add_constraint(ic_flow, "ic_flow")

    ic_heat = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"c": 0},
        batch_size=cfg.batch_size.ic_heat,
        lambda_weighting={
            "c": 10.0,
        },
        parameterization={t_symbol: 0.0},
    )
    domain.add_constraint(ic_heat, "ic_heat")

    # ── Integral continuity ───────────────────────────────────────────────────
    def integral_criteria(invar, params):
        sdf = geo.sdf(invar, params)
        return np.greater(sdf["sdf"], 0)

    integral_continuity = IntegralBoundaryConstraint(
        nodes=nodes,
        geometry=integral_line,
        outvar={"normal_dot_vel": 1 * ramp},
        batch_size=cfg.batch_size.num_integral_continuity,
        integral_batch_size=cfg.batch_size.integral_continuity,
        lambda_weighting={"normal_dot_vel": 0.1},
        criteria=integral_criteria,
        parameterization=Parameterization({
            x_pos: channel_length,
            t_symbol: (0.0, t_max),
        }),
    )
    domain.add_constraint(integral_continuity, "integral_continuity")

    # ── Validators (optional, requires NGC data) ──────────────────────────────
    file_path = "openfoam/heat_sink_zeroEq_Pr5_mesh20.csv"
    if os.path.exists(to_absolute_path(file_path)):
        mapping = {
            "Points:0": "x",
            "Points:1": "y",
            "U:0": "u",
            "U:1": "v",
            "p": "p",
            "d": "sdf",
            "nuT": "nu",
            "T": "c",
        }
        openfoam_var = csv_to_dict(to_absolute_path(file_path), mapping)
        openfoam_var["nu"] += nu
        openfoam_var["c"] += -base_temp
        openfoam_var["c"] /= 273.15
        # Inject t=t_max to compare at steady state
        n_pts = openfoam_var["x"].shape[0]
        openfoam_var["t"] = np.full((n_pts, 1), t_max)
        openfoam_invar_numpy = {
            key: value
            for key, value in openfoam_var.items()
            if key in ["x", "y", "sdf", "t"]
        }
        openfoam_outvar_numpy = {
            key: value
            for key, value in openfoam_var.items()
            if key in ["u", "v", "p", "c"]
        }
        openfoam_validator = PointwiseValidator(
            nodes=nodes,
            invar=openfoam_invar_numpy,
            true_outvar=openfoam_outvar_numpy,
        )
        domain.add_validator(openfoam_validator, "openfoam_steady")
    else:
        warnings.warn(
            f"Directory {file_path} does not exist. Will skip adding validators. "
            "Please download the additional files from NGC "
            "https://catalog.ngc.nvidia.com/orgs/nvidia/teams/physicsnemo/resources/"
            "physicsnemo_sym_examples_supplemental_materials"
        )

    # ── Monitors at time snapshots ────────────────────────────────────────────
    monitor_times = [0.0, 2.5, 5.0, t_max]
    interior_monitor_points = 200
    boundary_monitor_points = 200

    def time_tag(t_val: float) -> str:
        return f"t{t_val:.1f}".replace(".", "p")

    def weighted_area_mean(var, key):
        return torch.sum(var["area"] * var[key]) / torch.sum(var["area"])

    for t_val in monitor_times:
        tag = time_tag(t_val)
        fixed_time = Parameterization({t_symbol: t_val})

        global_monitor = PointwiseMonitor(
            geo.sample_interior(interior_monitor_points, parameterization=fixed_time),
            output_names=["u", "v", "c", "continuity", "momentum_x", "momentum_y"],
            metrics={
                f"mass_imbalance_{tag}": lambda var: torch.sum(
                    var["area"] * torch.abs(var["continuity"])
                ),
                f"momentum_imbalance_{tag}": lambda var: torch.sum(
                    var["area"]
                    * (torch.abs(var["momentum_x"]) + torch.abs(var["momentum_y"]))
                ),
                f"speed_max_{tag}": lambda var: torch.max(
                    torch.sqrt(var["u"] ** 2 + var["v"] ** 2)
                ),
                f"c_mean_{tag}": lambda var: weighted_area_mean(var, "c"),
            },
            nodes=nodes,
            requires_grad=True,
        )
        domain.add_monitor(global_monitor, f"global_{tag}")

        heat_sink_monitor = PointwiseMonitor(
            heat_sink.sample_boundary(
                boundary_monitor_points, parameterization=fixed_time
            ),
            output_names=["p", "c"],
            metrics={
                f"force_x_{tag}": lambda var: torch.sum(
                    var["normal_x"] * var["area"] * var["p"]
                ),
                f"force_y_{tag}": lambda var: torch.sum(
                    var["normal_y"] * var["area"] * var["p"]
                ),
                f"peakT_{tag}": lambda var: torch.max(var["c"]),
            },
            nodes=nodes,
        )
        domain.add_monitor(heat_sink_monitor, f"heat_sink_{tag}")

    # ── Solve ─────────────────────────────────────────────────────────────────
    slv = Solver(cfg, domain)
    slv.solve()


if __name__ == "__main__":
    run()
