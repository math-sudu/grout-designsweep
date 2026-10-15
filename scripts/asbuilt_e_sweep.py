"""Plane-strain elasticity on a bonded soil, lining and annulus mesh.

The support-design workflow uses inward radial loading at the soil-annulus
interface and updates the annulus modulus on a shared elastic problem.
The standalone CLI also accepts a supplied mesh/response sweep payload.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import gmsh
from mpi4py import MPI
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io.gmsh import model_to_mesh
import ufl

from asbuilt_geometry import (       # geometry + meshing, gmsh only
    SOIL_TAG_BASE,
    LINING_TAG,
    GROUT_TAG,
    TOP_TAG,
    BOTTOM_TAG,
    LEFT_TAG,
    RIGHT_TAG,
    CAVITY_TAG,
    INTERFACE_TAG,
    build_asbuilt_gmsh_model,
    stratum_profile,
)


def build_asbuilt_mesh(domain_width: float, domain_height: float,
                       axis_depth: float, r_cavity: float, r_ring_out: float,
                       r_grout_out: float, strata: list,
                       mesh_size: float, mesh_size_near_cavity: float,
                       with_lining: bool = True,
                       tag_ground_interface: bool = False):
    """The gmsh section of ``asbuilt_geometry``, converted to a dolfinx mesh."""
    e_by_tag = build_asbuilt_gmsh_model(
        domain_width, domain_height, axis_depth, r_cavity, r_ring_out,
        r_grout_out, strata, mesh_size, mesh_size_near_cavity,
        with_lining=with_lining, tag_ground_interface=tag_ground_interface)

    md = model_to_mesh(gmsh.model, MPI.COMM_WORLD, rank=0, gdim=2)
    gmsh.finalize()
    return md.mesh, md.cell_tags, md.facet_tags, e_by_tag


def make_E_field(domain, cell_tags, e_by_tag: dict, e_lining_pa: float,
                 e_grout_pa: float, grout_floor_pa: float) -> fem.Function:
    V_E = fem.functionspace(domain, ("DG", 0))
    f = fem.Function(V_E)
    arr = np.zeros_like(f.x.array, dtype=np.float64)
    for tag, e_mpa in e_by_tag.items():
        cells = cell_tags.indices[cell_tags.values == tag]
        arr[cells] = float(e_mpa) * 1e6
    arr[cell_tags.indices[cell_tags.values == LINING_TAG]] = float(e_lining_pa)
    arr[cell_tags.indices[cell_tags.values == GROUT_TAG]] = max(
        float(e_grout_pa), float(grout_floor_pa))
    if not np.all(arr > 0.0):
        raise RuntimeError(f"{int((arr <= 0).sum())} cells carry no modulus")
    f.x.array[:] = arr
    return f


def make_nu_field(domain, cell_tags, e_by_tag: dict, default_nu: float,
                  nu_by_material: dict) -> fem.Function:
    """Piecewise-constant Poisson ratio on the same tags ``make_E_field`` uses.

    ``nu_by_material`` names ``lining``, ``grout`` and ``soil``; ``soil`` is
    either one ratio for the whole ground or a list carrying one ratio per
    stratum in the profile order ``stratum_profile`` returns.  Anything the
    object omits keeps ``default_nu``, so a partial override prices exactly the
    materials it names.
    """
    V_nu = fem.functionspace(domain, ("DG", 0))
    f = fem.Function(V_nu)
    arr = np.full_like(f.x.array, float(default_nu), dtype=np.float64)

    soil = nu_by_material.get("soil")
    if soil is not None:
        soil_tags = sorted(t for t in e_by_tag if t >= SOIL_TAG_BASE)
        values = ([float(soil)] * len(soil_tags)
                  if np.isscalar(soil) else [float(v) for v in soil])
        if len(values) != len(soil_tags):
            raise RuntimeError(
                f"nu_by_material['soil'] carries {len(values)} ratios for "
                f"{len(soil_tags)} meshed strata")
        for tag, value in zip(soil_tags, values):
            arr[cell_tags.indices[cell_tags.values == tag]] = value
    for key, tag in (("lining", LINING_TAG), ("grout", GROUT_TAG)):
        if nu_by_material.get(key) is not None:
            arr[cell_tags.indices[cell_tags.values == tag]] = float(
                nu_by_material[key])

    if not np.all((arr > 0.0) & (arr < 0.5)):
        raise RuntimeError("Poisson ratio outside (0, 0.5) on "
                           f"{int(((arr <= 0.0) | (arr >= 0.5)).sum())} cells")
    f.x.array[:] = arr
    return f


def build_elastic_problem(domain, facet_tags, E_func, nu,
          cavity_net_pressure_pa: float, groundside_load: bool = False,
          load_center: tuple[float, float] | None = None,
          degree: int = 1):
    """Build an elastic problem that can be reused after updating E_func.

    ``nu`` is either a float -- the single-ratio path every shipped table is
    built on, assembled exactly as before -- or the piecewise-constant
    ``fem.Function`` ``make_nu_field`` returns.

    ``degree`` is the Lagrange order of the displacement space.  Every shipped
    table is order 1 and that stays the default, so an existing call assembles
    the same space it always did.  The option exists because a constant-strain
    triangle cannot carry an isochoric deformation: its strain is one tensor per
    cell, so on a fixed mesh the constraint tr(eps) = 0 is imposed once per cell
    and, as a material's ratio approaches 0.5, removes more of the element's
    freedom than the deformation actually costs.  The assembled response then
    stiffens for a reason that belongs to the element and not to the elasticity.
    A fresh cementitious paste is nearly incompressible, so a fluid annulus
    priced at a ratio near 0.5 on order-1 elements could be reporting the
    element.  Order 2 carries the same mesh into a space whose strain varies
    inside the cell and which therefore admits the isochoric modes.  How much
    of the difference the constraint is actually worth on a given section is a
    measurement and not a given -- it depends on how much of the domain is near
    incompressible and on how directly the reported quantity passes through it
    -- so the pair is run and the gap reported, rather than the order-1 number
    being discarded on the mechanism alone.

    """
    gdim = domain.geometry.dim
    V = fem.functionspace(domain, ("Lagrange", int(degree), (gdim,)))
    lam = E_func * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E_func / (2.0 * (1.0 + nu))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    def eps(w):
        return ufl.sym(ufl.grad(w))

    def sig(w):
        return lam * ufl.tr(eps(w)) * ufl.Identity(gdim) + 2.0 * mu * eps(w)

    a = ufl.inner(sig(u), eps(v)) * ufl.dx
    zero = fem.Constant(domain, np.zeros(gdim, dtype=np.float64))
    L = ufl.inner(zero, v) * ufl.dx
    if groundside_load:
        # Ground-side driver: the released in-situ radial stress at the
        # excavation boundary, applied as a uniform radial force density on
        # the soil-grout interface circle.  The facets are interior (soil
        # outside, grout inside, bonded), so the load enters through an
        # interior-facet measure and is carried by the assembled composite;
        # the interface displacement stays an unknown and the ground/support
        # partition of convergence comes out of equilibrium.  Same sign
        # convention as the mesh-hole traction below: the applied traction is
        # cavity_net_pressure_pa * rhat, so a negative value pulls the
        # interface toward the tunnel axis.  No traction on the mesh-hole
        # (ring inner) boundary in this mode, and no Dirichlet condition
        # prescribes the interface displacement.
        if load_center is None:
            raise RuntimeError("groundside_load requires load_center")
        if not np.any(facet_tags.values == INTERFACE_TAG):
            raise RuntimeError("groundside_load requires INTERFACE_TAG facets "
                               "in the mesh")
        domain.topology.create_entity_permutations()
        dS_int = ufl.Measure("dS", domain=domain, subdomain_data=facet_tags,
                             subdomain_id=INTERFACE_TAG)
        xsc = ufl.SpatialCoordinate(domain)
        centre = fem.Constant(domain, np.asarray(load_center,
                                                 dtype=np.float64))
        rvec = xsc - centre
        rhat = rvec / ufl.sqrt(ufl.dot(rvec, rvec))
        traction = cavity_net_pressure_pa * rhat
        L += ufl.inner(ufl.avg(traction), ufl.avg(v)) * dS_int
    else:
        ds_cav = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags,
                             subdomain_id=CAVITY_TAG)
        n = ufl.FacetNormal(domain)
        L += ufl.inner(-cavity_net_pressure_pa * n, v) * ds_cav

    fdim = domain.topology.dim - 1
    bcs = []
    bfac = facet_tags.indices[facet_tags.values == BOTTOM_TAG]
    if bfac.size:
        bcs.append(fem.dirichletbc(np.zeros(gdim, dtype=np.float64),
                                   fem.locate_dofs_topological(V, fdim, bfac), V))
    Vx, _ = V.sub(0).collapse()
    u_zero = fem.Function(Vx)
    u_zero.x.array[:] = 0.0
    for tag in (LEFT_TAG, RIGHT_TAG):
        fac = facet_tags.indices[facet_tags.values == tag]
        if fac.size:
            bcs.append(fem.dirichletbc(
                u_zero, fem.locate_dofs_topological((V.sub(0), Vx), fdim, fac),
                V.sub(0)))
    prob = LinearProblem(a, L, bcs=bcs,
                         petsc_options_prefix=f"asbuilt_{int(time.time()*1e6)}_",
                         petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                                        "pc_factor_mat_solver_type": "mumps"})
    return prob


def solve(domain, facet_tags, E_func, nu,
          cavity_net_pressure_pa: float, groundside_load: bool = False,
          load_center: tuple[float, float] | None = None,
          degree: int = 1, return_diagnostics: bool = False):
    """Solve one state, optionally returning its residual and dof count."""
    prob = build_elastic_problem(domain, facet_tags, E_func, nu,
                                cavity_net_pressure_pa, groundside_load,
                                load_center, degree)
    uh = prob.solve()
    uh = uh[0] if isinstance(uh, tuple) else uh
    if not return_diagnostics:
        return uh
    V = uh.function_space
    diag = {"n_dofs": int(V.dofmap.index_map.size_global
                          * V.dofmap.index_map_bs)}
    try:
        A = getattr(prob, "A", None)
        b = getattr(prob, "b", None)
        if A is None or b is None:
            raise AttributeError("LinearProblem exposes no assembled A / b")
        xv = getattr(uh.x, "petsc_vec", None)
        if xv is None:
            xv = uh.vector
        r = b.duplicate()
        A.mult(xv, r)                     # r <- A x
        r.axpby(1.0, -1.0, b)             # r <- b - A x
        diag["rhs_norm"] = float(b.norm())
        diag["residual_norm"] = float(r.norm())
        diag["relative_residual"] = (diag["residual_norm"] / diag["rhs_norm"]
                                     if diag["rhs_norm"] > 0.0 else float("nan"))
        r.destroy()
    except Exception as exc:                                   # noqa: BLE001
        diag["residual_error"] = f"{type(exc).__name__}: {exc}"
    return uh, diag


def top_uy(domain, u, degree: int = 1):
    """Vertical displacement on the y = 0 surface, sorted by x.

    ``degree`` is the order of the scalar space the component is interpolated
    into, and it wants to match the order of ``u``.  An order-2 displacement
    carries surface values at the facet midpoints, and an order-1 readout space
    has no dof there: reading an order-2 solve at order 1 subsamples the surface
    back to its vertices without saying so, and the peak taken over that reading
    is a minimum over half the nodes the solve actually resolved.  Both readings
    are wanted whenever two orders are compared -- the matched one for the
    profile the solve produced, the vertex one for a peak taken over exactly the
    points the order-1 table takes its own minimum over, which is what separates
    a change of discretization from a change of sampling -- so the order is an
    argument here rather than being read off ``u``.
    """
    V_s = fem.functionspace(domain, ("Lagrange", int(degree)))
    uy = fem.Function(V_s)
    pts = V_s.element.interpolation_points
    if callable(pts):
        pts = pts()
    uy.interpolate(fem.Expression(u[1], pts))
    co = V_s.tabulate_dof_coordinates()
    m = np.isclose(co[:, 1], 0.0, atol=1e-6)
    x, v = co[m, 0], uy.x.array[m]
    o = np.argsort(x)
    return x[o], v[o]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-npz", required=True)
    ap.add_argument("--output-npz", required=True)
    args = ap.parse_args()

    d = np.load(args.input_npz)
    E_grid = np.asarray(d["E_grid"], dtype=np.float64)
    f = lambda k: float(d[k])                                    # noqa: E731

    uniform_e = float(d["uniform_e_mpa"]) if "uniform_e_mpa" in d.files else 0.0
    strata = ([(0.0, None, uniform_e)] if uniform_e > 0.0
              else stratum_profile(f("cover_depth"), f("invert_depth"),
                                   f("sub_invert_e_mpa")))
    print(f"[asbuilt] strata: {[(round(a,2), None if b is None else round(b,2), e) for a, b, e in strata]}",
          flush=True)

    with_lining = bool(d["with_lining"]) if "with_lining" in d.files else True
    groundside = (bool(d["groundside_load"])
                  if "groundside_load" in d.files else False)
    print(f"[asbuilt] with_lining={with_lining}", flush=True)
    print("[asbuilt] load path: "
          + ("ground-side radial force density on the soil-grout interface "
             "(r_grout_out), no mesh-hole traction" if groundside
             else "net traction on the mesh-hole boundary"), flush=True)

    t0 = time.time()
    dom, ct, ft, e_by_tag = build_asbuilt_mesh(
        f("domain_width"), f("domain_height"), f("axis_depth"),
        f("r_cavity"), f("r_ring_out"), f("r_grout_out"), strata,
        f("mesh_size"), f("mesh_size_near_cavity"), with_lining=with_lining,
        tag_ground_interface=groundside)
    n_cells = dom.topology.index_map(dom.topology.dim).size_local
    print(f"[asbuilt] mesh built in {time.time()-t0:.1f} s, {n_cells} cells",
          flush=True)

    nu_spec = (json.loads(str(d["nu_by_material"]))
               if "nu_by_material" in d.files else None)
    nu_arg = f("nu")
    if nu_spec:
        nu_arg = make_nu_field(dom, ct, e_by_tag, f("nu"), nu_spec)
        print(f"[asbuilt] Poisson ratio by material: {nu_spec} "
              f"(default {f('nu')})", flush=True)

    degree = int(d["degree"]) if "degree" in d.files else 1
    if degree != 1:
        print(f"[asbuilt] displacement space: Lagrange {degree} "
              "(the shipped tables are Lagrange 1)", flush=True)

    uy_list, peak = [], []
    x_ref = None
    ts = time.time()
    for k, E in enumerate(E_grid):
        Ef = make_E_field(dom, ct, e_by_tag, f("e_lining_pa"), float(E),
                          f("grout_floor_pa"))
        u = solve(dom, ft, Ef, nu_arg, f("cavity_net_pressure_pa"),
                  groundside_load=groundside,
                  load_center=(0.0, -f("axis_depth")), degree=degree)
        x, v = top_uy(dom, u, degree=degree)
        if x_ref is None:
            x_ref = x.copy()
        elif x_ref.shape != x.shape:
            raise RuntimeError("top-surface node count changed across solves")
        uy_list.append(v)
        peak.append(float(v.min()))
        if (k + 1) % 16 == 0 or k == E_grid.size - 1:
            print(f"[asbuilt] solve {k+1:3d}/{E_grid.size}  E={E/1e6:8.3f} MPa  "
                  f"peak={1e3*v.min():+.4f} mm", flush=True)
    elapsed = time.time() - ts
    print(f"[asbuilt] {E_grid.size} solves in {elapsed:.1f} s", flush=True)
    print(f"[asbuilt] fluid anchor {1e3*peak[0]:+.4f} mm   "
          f"cured anchor {1e3*peak[-1]:+.4f} mm", flush=True)

    out = Path(args.output_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, x_top=x_ref, E_grid=E_grid, uy_grid=np.stack(uy_list, axis=0),
        E_test=np.array([]), uy_test=np.zeros((0, x_ref.size)),
        cavity_net_pressure_pa=np.array(f("cavity_net_pressure_pa")),
        grout_floor_pa=np.array(f("grout_floor_pa")),
        domain_width=np.array(f("domain_width")),
        nu=np.array(f("nu")),
        nu_by_material=np.array(json.dumps(nu_spec) if nu_spec else ""),
        degree=np.array(degree),
        n_cells=np.array(n_cells), solve_seconds=np.array(elapsed))
    print(f"[asbuilt] saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
