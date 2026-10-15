"""Reusable plane-strain section defined entirely by support_design.yaml."""

from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


class RadialSection:
    def __init__(self, config_path=ROOT / "configs/support_design.yaml", *, mesh_scale=1.0):
        from dolfinx import mesh as dmesh
        from asbuilt_e_sweep import build_asbuilt_mesh, make_E_field, build_elastic_problem
        from asbuilt_geometry import GROUT_TAG

        config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        g, elastic, mesh = config["geometry"], config["elasticity"], config["mesh"]
        cavity = g["lining_inner_diameter_m"] / 2
        ring = g["lining_outer_diameter_m"] / 2
        grout = g["excavation_diameter_m"] / 2
        axis = g["crown_cover_m"] + grout
        dom, ct, ft, e_by_tag = build_asbuilt_mesh(
            g["domain_width_m"], g["domain_height_m"], axis, cavity, ring, grout,
            [(0., None, elastic["ground_modulus_pa"] / 1e6)], mesh["size_m"],
            mesh["near_annulus_size_m"] * mesh_scale,
            with_lining=True, tag_ground_interface=True)
        self.cells = ct.indices[ct.values == GROUT_TAG]
        mid = dmesh.compute_midpoints(dom, dom.topology.dim, self.cells)
        self.radius = np.hypot(mid[:, 0], mid[:, 1] + axis)
        self.modulus = make_E_field(dom, ct, e_by_tag, elastic["lining_modulus_pa"], 1e9, 1e6)
        self.pressure = elastic["inward_traction_pa"]
        self.problem = build_elastic_problem(
            dom, ft, self.modulus, elastic["poisson_ratio"], self.pressure,
            groundside_load=True, load_center=(0., -axis), degree=1)
        coords = self.problem.u.function_space.tabulate_dof_coordinates()
        self.top = np.isclose(coords[:, 1], 0, atol=1e-6)
        self.geometry = dict(
            **g, grout_thickness_m=grout-ring, lining_thickness_m=ring-cavity,
            axis_depth_m=axis, ground_modulus_pa=elastic["ground_modulus_pa"],
            lining_modulus_pa=elastic["lining_modulus_pa"],
            poisson_ratio=elastic["poisson_ratio"], driver="groundside_interface_dS",
            peak_definition="maximum downward surface displacement per unit inward traction",
            dimension_source="Liu et al. (2021), doi:10.1016/j.tust.2021.103857, section 2.1",
            source_traction_pa=self.pressure, element_degree=1,
            n_cells=dom.topology.index_map(dom.topology.dim).size_local,
            mesh_size_m=mesh["size_m"],
            near_annulus_size_m=mesh["near_annulus_size_m"] * mesh_scale)

    def response(self, radius, modulus):
        """Peak downward surface compliance in micrometres/kPa."""
        self.modulus.x.array[self.cells] = np.interp(self.radius, radius, modulus)
        u = self.problem.solve()
        return float(u.x.array.reshape(-1, 2)[self.top, 1].min() / self.pressure * 1e9)
