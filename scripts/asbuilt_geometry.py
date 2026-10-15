"""Geometry and mesh construction for the as-built anchor section, gmsh only.

Split out of ``asbuilt_e_sweep.py`` so that the mesh the response tables were
solved on can be rebuilt without an FE stack.  ``asbuilt_e_sweep`` imports this
module and adds the dolfinx conversion on top; the figure that draws the mesh
imports it directly.  Nothing here touches dolfinx, mpi4py or ufl, so it runs on
the host as well as inside the container.

The single copy matters: a figure that redrew the section from its dimensions
would be a drawing of what the text says, not of what was solved.
"""

from __future__ import annotations

import numpy as np
import gmsh


SOIL_TAG_BASE = 100          # stratum k gets SOIL_TAG_BASE + k
LINING_TAG = 3
GROUT_TAG = 2
TOP_TAG, BOTTOM_TAG, LEFT_TAG, RIGHT_TAG, CAVITY_TAG = 11, 12, 13, 14, 15
INTERFACE_TAG = 16           # soil-grout interface circle at r_grout_out


def _curve_lies_on_circle(tag: int, cx: float, cy: float, radius: float,
                          tol: float = 1e-6) -> bool:
    """Whether every point of curve ``tag`` sits at ``radius`` from (cx, cy).

    The interface curves are arcs of the ``r_grout_out`` circle, split by the
    stratum lines, so a centroid test cannot separate them from arcs of the
    other circles; sampling the parametrization can.
    """
    lo, hi = gmsh.model.getParametrizationBounds(1, tag)
    for t in np.linspace(float(lo[0]), float(hi[0]), 7)[1:-1]:
        x, y, _ = gmsh.model.getValue(1, tag, [float(t)])
        if abs(float(np.hypot(x - cx, y - cy)) - radius) > tol:
            return False
    return True


def stratum_profile(cover_depth: float, invert_depth: float,
                    sub_invert_e_mpa: float) -> list[tuple[float, float, float]]:
    """(top_depth, bottom_depth, E in MPa) for the reported six-layer profile.

    The fines zone spans the surface to the crown and the coarse zone the crown
    to the invert, each zone at equal thickness across its three layers, which
    is the weighting the shipped single modulus is built from.  Below the invert
    the reported profile stops, so the deepest listed layer is carried on and
    the choice is reported as its own variant.
    """
    fines = [10.0, 30.0, 15.0]
    coarse = [20.0, 300.0, 270.0]
    out: list[tuple[float, float, float]] = []
    h = cover_depth / len(fines)
    for k, e in enumerate(fines):
        out.append((k * h, (k + 1) * h, e))
    h = (invert_depth - cover_depth) / len(coarse)
    for k, e in enumerate(coarse):
        out.append((cover_depth + k * h, cover_depth + (k + 1) * h, e))
    out.append((invert_depth, None, sub_invert_e_mpa))
    return out


def build_asbuilt_gmsh_model(domain_width: float, domain_height: float,
                             axis_depth: float, r_cavity: float,
                             r_ring_out: float, r_grout_out: float,
                             strata: list, mesh_size: float,
                             mesh_size_near_cavity: float,
                             with_lining: bool = True,
                             tag_ground_interface: bool = False) -> dict:
    """Initialize gmsh, build the section, tag it, and generate the 2-D mesh.

    Leaves the gmsh model live and returns ``e_by_tag``; the caller converts it
    (``asbuilt_e_sweep.build_asbuilt_mesh``) or reads the raw triangulation
    (the F-MESH figure) and is responsible for ``gmsh.finalize()``.

    ``with_lining=False`` puts the mesh hole at ``r_ring_out`` and leaves the
    ring out of the series, which is the shipped construction carried to a new
    annulus radius; the traction then lands on the annulus inner face exactly as
    it does in the shipped model.

    ``tag_ground_interface=True`` additionally tags the interior facets on the
    soil--grout interface circle at ``r_grout_out`` with ``INTERFACE_TAG`` so
    the ground-side driver can load them; the geometry, the mesh fields and the
    recorded boundary classification are untouched.
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("bjucr_asbuilt")

    Lx, Ly = float(domain_width), float(domain_height)
    cx, cy = 0.0, -float(axis_depth)
    r_hole = float(r_cavity) if with_lining else float(r_ring_out)

    # Whole annuli, built before the ground so nothing splits them.
    d_out = gmsh.model.occ.addDisk(cx, cy, 0.0, r_grout_out, r_grout_out)
    d_ring = gmsh.model.occ.addDisk(cx, cy, 0.0, r_ring_out, r_ring_out)
    # keep d_ring when a lining is wanted -- it is the object the cavity disk is
    # cut out of; discard it otherwise, where it is only the tool
    grout, _ = gmsh.model.occ.cut([(2, d_out)], [(2, d_ring)],
                                  removeObject=True, removeTool=not with_lining)
    lining = []
    if with_lining:
        d_cav = gmsh.model.occ.addDisk(cx, cy, 0.0, r_cavity, r_cavity)
        # removeTool=True: the cavity is a hole, so the disk must not survive as
        # a surface -- if it does it is fragmented back in as material and the
        # cavity face becomes an internal interface the exterior traction misses.
        lining, _ = gmsh.model.occ.cut([(2, d_ring)], [(2, d_cav)],
                                       removeObject=True, removeTool=True)
    if with_lining and not lining:
        raise RuntimeError("the lining cut produced no surface")

    # Ground strata, each already holed by the grout outer disk so that the
    # annuli survive the fragment whole and stay identifiable by area.
    soil_pieces: list[tuple[list, float]] = []
    for top, bot, e_mpa in strata:
        y_hi = -float(top)
        y_lo = -Ly if bot is None else -float(bot)
        if y_lo >= y_hi:
            continue
        rect = gmsh.model.occ.addRectangle(-Lx / 2.0, y_lo, 0.0, Lx, y_hi - y_lo)
        hole = gmsh.model.occ.addDisk(cx, cy, 0.0, r_grout_out, r_grout_out)
        piece, _ = gmsh.model.occ.cut([(2, rect)], [(2, hole)],
                                      removeObject=True, removeTool=True)
        if piece:
            soil_pieces.append((piece, float(e_mpa)))

    all_dimtags = [dt for p, _ in soil_pieces for dt in p] + grout + lining
    gmsh.model.occ.fragment(all_dimtags, [])
    gmsh.model.occ.synchronize()

    area_lining = np.pi * (r_ring_out ** 2 - r_cavity ** 2)
    area_grout = np.pi * (r_grout_out ** 2 - r_ring_out ** 2)
    e_by_tag: dict[int, float] = {}
    lining_tags, grout_tags = [], []
    soil_by_stratum: dict[int, list[int]] = {}

    for dim, tag in gmsh.model.getEntities(dim=2):
        area = gmsh.model.occ.getMass(dim, tag)
        if with_lining and abs(area - area_lining) < 1e-6 * max(area_lining, 1.0):
            lining_tags.append(tag)
            continue
        if abs(area - area_grout) < 1e-6 * max(area_grout, 1.0):
            grout_tags.append(tag)
            continue
        x_com, y_com, _ = gmsh.model.occ.getCenterOfMass(dim, tag)
        # A leftover cavity disk is a surface whose whole bounding box fits
        # inside the hole.  Testing the box rather than the centroid keeps the
        # check from firing on a single un-split ground block, whose centroid
        # can sit near the axis while the block spans the domain.
        bx0, by0, _, bx1, by1, _ = gmsh.model.occ.getBoundingBox(dim, tag)
        if (bx0 >= cx - r_hole - 1e-6 and bx1 <= cx + r_hole + 1e-6
                and by0 >= cy - r_hole - 1e-6 and by1 <= cy + r_hole + 1e-6):
            raise RuntimeError(
                f"surface {tag} (area {area:.3f}) sits inside the cavity; the "
                "cavity must be a hole, not a meshed region")
        depth = -y_com
        k_hit = None
        for k, (top, bot, _e) in enumerate(strata):
            hi = float(top)
            lo = Ly if bot is None else float(bot)
            if hi - 1e-9 <= depth <= lo + 1e-9:
                k_hit = k
                break
        if k_hit is None:
            raise RuntimeError(f"surface {tag} at depth {depth:.3f} m "
                               "matches no stratum")
        soil_by_stratum.setdefault(k_hit, []).append(tag)
        e_by_tag[SOIL_TAG_BASE + k_hit] = strata[k_hit][2]

    if len(grout_tags) != 1 or len(lining_tags) != (1 if with_lining else 0):
        raise RuntimeError(f"expected {1 if with_lining else 0} lining and one "
                           f"grout surface, got {len(lining_tags)} and "
                           f"{len(grout_tags)}")
    if lining_tags:
        gmsh.model.addPhysicalGroup(2, lining_tags, LINING_TAG)
    gmsh.model.addPhysicalGroup(2, grout_tags, GROUT_TAG)
    for k, tags in soil_by_stratum.items():
        gmsh.model.addPhysicalGroup(2, tags, SOIL_TAG_BASE + k)

    # Boundary curves: the outer box by position, the cavity by radius; the
    # soil-grout interface, when asked for, by lying on the r_grout_out circle.
    top_b, bottom_b, left_b, right_b, cav_b, iface_b = [], [], [], [], [], []
    for dim, tag in gmsh.model.getEntities(dim=1):
        if tag_ground_interface and _curve_lies_on_circle(
                tag, cx, cy, float(r_grout_out)):
            iface_b.append(tag)
            continue
        x_com, y_com, _ = gmsh.model.occ.getCenterOfMass(dim, tag)
        r = float(np.hypot(x_com - cx, y_com - cy))
        if abs(y_com) < 1e-6:
            top_b.append(tag)
        elif abs(y_com + Ly) < 1e-6:
            bottom_b.append(tag)
        elif abs(x_com + Lx / 2.0) < 1e-6:
            left_b.append(tag)
        elif abs(x_com - Lx / 2.0) < 1e-6:
            right_b.append(tag)
        elif r < r_hole * 0.5:
            cav_b.append(tag)
    for tags, tag_id in ((top_b, TOP_TAG), (bottom_b, BOTTOM_TAG),
                         (left_b, LEFT_TAG), (right_b, RIGHT_TAG),
                         (cav_b, CAVITY_TAG), (iface_b, INTERFACE_TAG)):
        if tags:
            gmsh.model.addPhysicalGroup(1, tags, tag_id)
    if not cav_b:
        raise RuntimeError("no cavity boundary curve found")
    if tag_ground_interface and not iface_b:
        raise RuntimeError("no soil-grout interface curve found on the "
                           "r_grout_out circle")
    print(f"[asbuilt] lining={lining_tags} grout={grout_tags} "
          f"strata={ {k: len(v) for k, v in soil_by_stratum.items()} } "
          f"cavity_curves={len(cav_b)} interface_curves={len(iface_b)}",
          flush=True)

    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin",
                          mesh_size_near_cavity / 2.0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1)
    fld = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(fld, "CurvesList", cav_b)
    thr = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(thr, "InField", fld)
    gmsh.model.mesh.field.setNumber(thr, "SizeMin", mesh_size_near_cavity)
    gmsh.model.mesh.field.setNumber(thr, "SizeMax", mesh_size)
    gmsh.model.mesh.field.setNumber(thr, "DistMin", 2.0 * r_grout_out)
    gmsh.model.mesh.field.setNumber(thr, "DistMax", 6.0 * r_grout_out)
    gmsh.model.mesh.field.setAsBackgroundMesh(thr)
    gmsh.model.mesh.generate(2)

    return e_by_tag
