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
# Parameterized variant: inlet_vel is an extra network input (range 1.0~2.5 m/s).
# A single trained model covers the full inlet-velocity space.

import os
import warnings

import torch
import numpy as np
from sympy import Symbol

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


@physicsnemo.sym.main(config_path="conf_param", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    # ── Domain parameters ─────────────────────────────────────────────────────
    channel_length = (-2.5, 2.5)
    channel_width = (-0.5, 0.5)
    heat_sink_origin = (-1, -0.3)
    nr_heat_sink_fins = 3
    gap = 0.15 + 0.1
    heat_sink_length = 1.0
    heat_sink_fin_thickness = 0.1
    heat_sink_temp = 350
    base_temp = 293.498
    nu = 0.01
    diffusivity = 0.01 / 5

    # inlet_vel parameter range (1.0 ~ 2.5 m/s)
    inlet_vel_sym = Parameter("inlet_vel")
    inlet_vel_range = (1.0, 2.5)
    param_ranges = Parameterization({inlet_vel_sym: inlet_vel_range})

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

    # ── Nodes ─────────────────────────────────────────────────────────────────
    ze = ZeroEquation(
        nu=nu, rho=1.0, dim=2, max_distance=(channel_width[1] - channel_width[0]) / 2
    )
    ns = NavierStokes(nu=ze.equations["nu"], rho=1.0, dim=2, time=False)
    ade = AdvectionDiffusion(T="c", rho=1.0, D=diffusivity, dim=2, time=False)
    gn_c = GradNormal("c", dim=2, time=False)
    normal_dot_vel = NormalDotVec(["u", "v"])

    # Networks receive inlet_vel as an additional input coordinate
    flow_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("inlet_vel")],
        output_keys=[Key("u"), Key("v"), Key("p")],
        cfg=cfg.arch.fully_connected,
    )
    heat_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("inlet_vel")],
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

    # inlet — parabolic profile scaled by inlet_vel_sym
    inlet_parabola = parabola(
        y, inter_1=channel_width[0], inter_2=channel_width[1], height=inlet_vel_sym
    )
    inlet_constraint = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=inlet,
        outvar={"u": inlet_parabola, "v": 0, "c": 0},
        batch_size=cfg.batch_size.inlet,
        parameterization=param_ranges,
    )
    domain.add_constraint(inlet_constraint, "inlet")

    # outlet
    outlet_constraint = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=outlet,
        outvar={"p": 0},
        batch_size=cfg.batch_size.outlet,
        parameterization=param_ranges,
    )
    domain.add_constraint(outlet_constraint, "outlet")

    # heat_sink wall
    hs_wall = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=heat_sink,
        outvar={"u": 0, "v": 0, "c": (heat_sink_temp - base_temp) / 273.15},
        batch_size=cfg.batch_size.hs_wall,
        parameterization=param_ranges,
    )
    domain.add_constraint(hs_wall, "heat_sink_wall")

    # channel wall
    channel_wall = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=channel,
        outvar={"u": 0, "v": 0, "normal_gradient_c": 0},
        batch_size=cfg.batch_size.channel_wall,
        parameterization=param_ranges,
    )
    domain.add_constraint(channel_wall, "channel_wall")

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
        parameterization=param_ranges,
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
        parameterization=param_ranges,
    )
    domain.add_constraint(interior_heat, "interior_heat")

    # integral continuity — combine x_pos and inlet_vel parameters
    # Target flow rate = integral of inlet parabola over channel width:
    #   u(y) = parabola(y, -0.5, 0.5, height=inlet_vel_sym)
    #   integral_{-0.5}^{0.5} u dy = (2/3) * inlet_vel_sym
    # When inlet_vel=1.5 (original fixed case): (2/3)*1.5 = 1.0 (matches original)
    def integral_criteria(invar, params):
        sdf = geo.sdf(invar, params)
        return np.greater(sdf["sdf"], 0)

    integral_continuity = IntegralBoundaryConstraint(
        nodes=nodes,
        geometry=integral_line,
        outvar={"normal_dot_vel": (2 / 3) * inlet_vel_sym},
        batch_size=cfg.batch_size.num_integral_continuity,
        integral_batch_size=cfg.batch_size.integral_continuity,
        lambda_weighting={"normal_dot_vel": 1.0},
        criteria=integral_criteria,
        parameterization=Parameterization({
            x_pos: channel_length,
            inlet_vel_sym: inlet_vel_range,
        }),
    )
    domain.add_constraint(integral_continuity, "integral_continuity")

    representative_velocities = (1.0, 1.5, 2.5)
    boundary_validator_points = 256
    interior_monitor_points = 200
    boundary_monitor_points = 200
    line_monitor_points = 256

    def velocity_tag(value: float) -> str:
        return f"v{value:.1f}".replace(".", "p")

    def weighted_area_mean(var, key):
        return torch.sum(var["area"] * var[key]) / torch.sum(var["area"])

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
        # Add fixed inlet_vel=1.5 to match OpenFOAM reference condition
        n_pts = openfoam_var["x"].shape[0]
        openfoam_var["inlet_vel"] = np.full((n_pts, 1), 1.5)
        openfoam_invar_numpy = {
            key: value
            for key, value in openfoam_var.items()
            if key in ["x", "y", "sdf", "inlet_vel"]
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
        domain.add_validator(openfoam_validator, "openfoam_1p5")
    else:
        warnings.warn(
            f"Directory {file_path} does not exist. Will skip adding validators. "
            "Please download the additional files from NGC "
            "https://catalog.ngc.nvidia.com/orgs/nvidia/teams/physicsnemo/resources/"
            "physicsnemo_sym_examples_supplemental_materials"
        )

    # Additional analytic validators at representative velocities make boundary
    # regressions visible even when OpenFOAM data only exists at inlet_vel=1.5.
    for monitor_vel in representative_velocities:
        vel_tag = velocity_tag(monitor_vel)
        fixed_params = Parameterization({inlet_vel_sym: monitor_vel})

        inlet_samples = inlet.sample_boundary(
            boundary_validator_points, parameterization=fixed_params
        )
        inlet_y = inlet_samples["y"]
        inlet_invar_numpy = {
            "x": inlet_samples["x"],
            "y": inlet_samples["y"],
            "inlet_vel": np.full_like(inlet_samples["x"], monitor_vel),
        }
        inlet_outvar_numpy = {
            "u": monitor_vel * (1.0 - 4.0 * inlet_y**2),
            "v": np.zeros_like(inlet_y),
            "c": np.zeros_like(inlet_y),
        }
        inlet_validator = PointwiseValidator(
            nodes=nodes,
            invar=inlet_invar_numpy,
            true_outvar=inlet_outvar_numpy,
        )
        domain.add_validator(inlet_validator, f"inlet_profile_{vel_tag}")

        outlet_samples = outlet.sample_boundary(
            boundary_validator_points, parameterization=fixed_params
        )
        outlet_invar_numpy = {
            "x": outlet_samples["x"],
            "y": outlet_samples["y"],
            "inlet_vel": np.full_like(outlet_samples["x"], monitor_vel),
        }
        outlet_outvar_numpy = {
            "p": np.zeros_like(outlet_samples["x"]),
        }
        outlet_validator = PointwiseValidator(
            nodes=nodes,
            invar=outlet_invar_numpy,
            true_outvar=outlet_outvar_numpy,
        )
        domain.add_validator(outlet_validator, f"outlet_pressure_{vel_tag}")

        target_flow = (2.0 / 3.0) * monitor_vel

        global_monitor = PointwiseMonitor(
            geo.sample_interior(interior_monitor_points, parameterization=fixed_params),
            output_names=["u", "v", "c", "continuity", "momentum_x", "momentum_y"],
            metrics={
                f"mass_imbalance_{vel_tag}": lambda var: torch.sum(
                    var["area"] * torch.abs(var["continuity"])
                ),
                f"momentum_imbalance_{vel_tag}": lambda var: torch.sum(
                    var["area"]
                    * (torch.abs(var["momentum_x"]) + torch.abs(var["momentum_y"]))
                ),
                f"speed_max_{vel_tag}": lambda var: torch.max(
                    torch.sqrt(var["u"] ** 2 + var["v"] ** 2)
                ),
                f"c_mean_{vel_tag}": lambda var: weighted_area_mean(var, "c"),
            },
            nodes=nodes,
            requires_grad=True,
        )
        domain.add_monitor(global_monitor, f"global_{vel_tag}")

        heat_sink_monitor = PointwiseMonitor(
            heat_sink.sample_boundary(
                boundary_monitor_points, parameterization=fixed_params
            ),
            output_names=["p", "c"],
            metrics={
                f"force_x_{vel_tag}": lambda var: torch.sum(
                    var["normal_x"] * var["area"] * var["p"]
                ),
                f"force_y_{vel_tag}": lambda var: torch.sum(
                    var["normal_y"] * var["area"] * var["p"]
                ),
                f"peakT_{vel_tag}": lambda var: torch.max(var["c"]),
            },
            nodes=nodes,
        )
        domain.add_monitor(heat_sink_monitor, f"heat_sink_{vel_tag}")

        inlet_monitor = PointwiseMonitor(
            inlet.sample_boundary(line_monitor_points, parameterization=fixed_params),
            output_names=["p", "c", "normal_dot_vel"],
            metrics={
                f"inlet_pressure_mean_{vel_tag}": lambda var: weighted_area_mean(
                    var, "p"
                ),
                f"inlet_c_mean_{vel_tag}": lambda var: weighted_area_mean(var, "c"),
                f"inlet_flow_{vel_tag}": lambda var: -torch.sum(
                    var["area"] * var["normal_dot_vel"]
                ),
                f"inlet_flow_error_{vel_tag}": lambda var, target=target_flow: -torch.sum(
                    var["area"] * var["normal_dot_vel"]
                )
                - target,
            },
            nodes=nodes,
        )
        domain.add_monitor(inlet_monitor, f"inlet_{vel_tag}")

        outlet_monitor = PointwiseMonitor(
            outlet.sample_boundary(line_monitor_points, parameterization=fixed_params),
            output_names=["p", "c", "normal_dot_vel"],
            metrics={
                f"outlet_pressure_mean_{vel_tag}": lambda var: weighted_area_mean(
                    var, "p"
                ),
                f"outlet_c_mean_{vel_tag}": lambda var: weighted_area_mean(var, "c"),
                f"outlet_flow_{vel_tag}": lambda var: torch.sum(
                    var["area"] * var["normal_dot_vel"]
                ),
                f"outlet_flow_error_{vel_tag}": lambda var, target=target_flow: torch.sum(
                    var["area"] * var["normal_dot_vel"]
                )
                - target,
            },
            nodes=nodes,
        )
        domain.add_monitor(outlet_monitor, f"outlet_{vel_tag}")

    # ── Solve ─────────────────────────────────────────────────────────────────
    slv = Solver(cfg, domain)
    slv.solve()


if __name__ == "__main__":
    run()
