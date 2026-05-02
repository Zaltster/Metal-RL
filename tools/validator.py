"""humanoid_v1 spec — parser and validator.

Reads a JSON spec, runs the §9 invariants from humanoid_robot_spec.md, and
returns a Report listing errors and warnings. Also derives the dof_layout
(action-vector offset table) per §4.6.

Usage:
    python validator.py <spec.json>

Returns exit code 0 if no errors (warnings allowed), 1 otherwise.
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# -- Report -------------------------------------------------------------------

@dataclass
class Issue:
    severity: str  # "error" | "warning"
    code: str
    message: str
    path: str = ""


@dataclass
class Report:
    issues: List[Issue] = field(default_factory=list)
    dof_layout: Optional[Dict[str, Any]] = None

    def err(self, code: str, msg: str, path: str = "") -> None:
        self.issues.append(Issue("error", code, msg, path))

    def warn(self, code: str, msg: str, path: str = "") -> None:
        self.issues.append(Issue("warning", code, msg, path))

    @property
    def errors(self) -> List[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        if not self.issues:
            return "OK  all invariants pass."
        out = []
        for i in self.issues:
            tag = "ERROR" if i.severity == "error" else " warn"
            path = f"  [{i.path}]" if i.path else ""
            out.append(f"{tag}  {i.code}: {i.message}{path}")
        n_err = len(self.errors)
        n_warn = len(self.issues) - n_err
        out.append("")
        out.append(f"{n_err} error(s), {n_warn} warning(s)")
        return "\n".join(out)


# -- DoF layout (spec §4.6) ---------------------------------------------------

DOF_SIZE = {"free": 0, "fixed": 0, "revolute": 1, "prismatic": 1, "spherical": 3}


def derive_dof_layout(spec: dict) -> Dict[str, Any]:
    rows = []
    offset = 0
    for j in spec.get("joints", []):
        size = DOF_SIZE.get(j.get("type", ""), 0)
        rows.append({"name": j.get("name"), "offset": offset, "size": size})
        offset += size
    return {"total_dofs": offset, "joints": rows}


# -- Math helpers -------------------------------------------------------------

def _quat_norm(q: List[float]) -> float:
    return math.sqrt(sum(x * x for x in q))


def _is_pos_def_3x3(I: List[List[float]]) -> bool:
    """Sylvester's criterion: all leading principal minors > 0."""
    a, b, c = I[0][0], I[1][1], I[2][2]
    d, e, f = I[0][1], I[0][2], I[1][2]
    if a <= 0:
        return False
    if a * b - d * d <= 0:
        return False
    det3 = a * (b * c - f * f) - d * (d * c - f * e) + e * (d * f - b * e)
    return det3 > 0


# -- Invariant checks (spec §9) -----------------------------------------------

def _v_top(spec: dict, r: Report) -> None:
    for k in ("schema_version", "name", "units", "links", "joints", "default_pose"):
        if k not in spec:
            r.err("MISSING_FIELD", f"top-level '{k}' missing", "$")
    if spec.get("schema_version") not in (None, "1.0"):
        r.warn("VERSION", f"unexpected schema_version {spec.get('schema_version')!r}",
               "$.schema_version")


def _v_quaternions(spec: dict, r: Report) -> None:
    """Invariant 6: all quaternions normalized within 1e-6."""
    def check(q, path):
        if not isinstance(q, list) or len(q) != 4:
            n = len(q) if isinstance(q, list) else "?"
            r.err("QUAT_SHAPE", f"quaternion has {n} elements (expected 4)", path)
            return
        n = _quat_norm(q)
        if abs(n - 1.0) > 1e-6:
            r.err("QUAT_NORM", f"quaternion not normalized (|q|={n:.6f})", path)

    for li, link in enumerate(spec.get("links", [])):
        for vi, v in enumerate(link.get("visual", [])):
            t = v.get("transform", {})
            if "rotation" in t:
                check(t["rotation"], f"$.links[{li}].visual[{vi}].transform.rotation")
        for ci, c in enumerate(link.get("collision", [])):
            t = c.get("transform", {})
            if "rotation" in t:
                check(t["rotation"], f"$.links[{li}].collision[{ci}].transform.rotation")

    for ji, j in enumerate(spec.get("joints", [])):
        for fld in ("frame_in_parent", "frame_in_child"):
            if fld in j and j[fld] is not None:
                check(j[fld], f"$.joints[{ji}].{fld}")

    dp = spec.get("default_pose", {})
    if isinstance(dp, dict) and "root_rotation" in dp:
        check(dp["root_rotation"], "$.default_pose.root_rotation")


def _v_links(spec: dict, r: Report) -> None:
    """Invariant 1: positive mass and positive-definite inertia per link."""
    seen = set()
    for li, link in enumerate(spec.get("links", [])):
        path = f"$.links[{li}]"
        name = link.get("name")
        if not name:
            r.err("LINK_NAME", "link missing 'name'", path); continue
        if name in seen:
            r.err("LINK_DUPE", f"duplicate link name {name!r}", path)
        seen.add(name)
        inertial = link.get("inertial", {}) or {}
        mass = inertial.get("mass", 0)
        if not isinstance(mass, (int, float)) or mass <= 0:
            r.err("MASS_POS", f"link {name!r} mass must be > 0 (got {mass!r})",
                  f"{path}.inertial.mass")
        I = inertial.get("inertia", [])
        if not (isinstance(I, list) and len(I) == 6):
            n = len(I) if isinstance(I, list) else "?"
            r.err("INERTIA_SHAPE", f"link {name!r} inertia must be 6 numbers (got {n})",
                  f"{path}.inertial.inertia")
        else:
            Ixx, Iyy, Izz, Ixy, Ixz, Iyz = I
            mat = [[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]]
            if not _is_pos_def_3x3(mat):
                r.err("INERTIA_PD", f"link {name!r} inertia tensor not positive-definite",
                      f"{path}.inertial.inertia")


def _v_topology(spec: dict, r: Report) -> None:
    """Invariants 2, 3, 7: topology and ordering."""
    links = spec.get("links", [])
    link_names = [l.get("name") for l in links]
    link_set = set(link_names)
    link_index = {n: i for i, n in enumerate(link_names) if n}

    free_count = 0
    parent_count: Dict[str, int] = {}
    seen = set()

    for ji, j in enumerate(spec.get("joints", [])):
        path = f"$.joints[{ji}]"
        name = j.get("name")
        if not name:
            r.err("JOINT_NAME", "joint missing 'name'", path); continue
        if name in seen:
            r.err("JOINT_DUPE", f"duplicate joint name {name!r}", path)
        seen.add(name)

        jtype = j.get("type")
        if jtype not in ("free", "fixed", "revolute", "prismatic", "spherical"):
            r.err("JOINT_TYPE", f"unknown joint type {jtype!r}", f"{path}.type")
            continue

        if jtype == "free":
            free_count += 1

        parent = j.get("parent_link")
        child = j.get("child_link")
        if jtype != "free" and parent != "world" and parent not in link_set:
            r.err("JOINT_PARENT_MISSING",
                  f"joint {name!r} parent_link {parent!r} not in links[]",
                  f"{path}.parent_link")
        if child not in link_set:
            r.err("JOINT_CHILD_MISSING",
                  f"joint {name!r} child_link {child!r} not in links[]",
                  f"{path}.child_link")

        parent_count[child] = parent_count.get(child, 0) + 1

        # Invariant 7: parent before child in links[]
        if jtype != "free" and parent in link_index and child in link_index:
            if link_index[parent] >= link_index[child]:
                r.err("LINK_ORDER",
                      f"joint {name!r}: parent {parent!r} (idx {link_index[parent]}) "
                      f"must come before child {child!r} (idx {link_index[child]}) in links[]",
                      path)

    # Invariant 3: exactly 1 free root
    if free_count != 1:
        r.err("FREE_COUNT", f"expected exactly 1 free root joint; found {free_count}",
              "$.joints")

    # Invariant 2: each non-root link has exactly one parent joint
    for n in link_names:
        if not n:
            continue
        c = parent_count.get(n, 0)
        if c == 0:
            r.err("LINK_ORPHAN", f"link {n!r} has no parent joint",
                  f"$.links[*]({n})")
        elif c > 1:
            r.err("LINK_MULTIPARENT",
                  f"link {n!r} has {c} parent joints (expected 1)",
                  f"$.links[*]({n})")


def _v_default_pose(spec: dict, r: Report) -> None:
    """Invariant 5: default_pose values inside limits."""
    dp = spec.get("default_pose")
    if not isinstance(dp, dict):
        return
    rp = dp.get("root_position")
    if not (isinstance(rp, list) and len(rp) == 3):
        r.err("MISSING_FIELD",
              "default_pose.root_position must be [x, y, z]",
              "$.default_pose.root_position")

    joints_by_name = {j["name"]: j for j in spec.get("joints", []) if "name" in j}
    jp = dp.get("joint_positions") or {}

    for jname, val in jp.items():
        if jname not in joints_by_name:
            r.warn("POSE_UNKNOWN_JOINT",
                   f"default_pose has joint {jname!r} not in joints[]",
                   f"$.default_pose.joint_positions.{jname}")
            continue
        j = joints_by_name[jname]
        jt = j.get("type")
        path = f"$.default_pose.joint_positions.{jname}"

        if jt in ("revolute", "prismatic"):
            if not isinstance(val, (int, float)):
                r.err("POSE_TYPE", f"{jname}: scalar expected for {jt}", path); continue
            limits = (j.get("limits") or {}).get("position")
            if limits and not (limits[0] <= val <= limits[1]):
                r.err("POSE_LIMIT", f"{jname}={val} outside limits {limits}", path)
        elif jt == "spherical":
            if not (isinstance(val, list) and len(val) == 3):
                r.err("POSE_TYPE",
                      f"{jname}: [swing_x, swing_y, twist_z] expected for spherical",
                      path); continue
            limits = j.get("limits") or {}
            for k, axis in enumerate(("swing_x", "swing_y", "twist_z")):
                lim = limits.get(axis)
                if lim and not (lim[0] <= val[k] <= lim[1]):
                    r.err("POSE_LIMIT",
                          f"{jname}.{axis}={val[k]} outside limits {lim}", path)
        elif jt in ("free", "fixed"):
            r.warn("POSE_UNUSED",
                   f"{jname} is {jt} (no DoF) but appears in default_pose", path)


def _v_collision(spec: dict, r: Report) -> None:
    """Invariant 9: every collision shape has valid type and positive dimensions."""
    valid = ("box", "sphere", "capsule", "cylinder", "convex_hull")
    for li, link in enumerate(spec.get("links", [])):
        for ci, c in enumerate(link.get("collision", [])):
            path = f"$.links[{li}].collision[{ci}]"
            ct = c.get("type")
            if ct not in valid:
                r.err("COLLISION_TYPE", f"unknown collision type {ct!r}", path); continue
            p = c.get("params", {}) or {}
            if ct == "box":
                he = p.get("half_extents")
                if not (isinstance(he, list) and len(he) == 3
                        and all(isinstance(x, (int, float)) and x > 0 for x in he)):
                    r.err("COLLISION_DIM",
                          "box.half_extents must be 3 positive numbers", path)
            elif ct == "sphere":
                if not (isinstance(p.get("radius"), (int, float)) and p["radius"] > 0):
                    r.err("COLLISION_DIM", "sphere.radius must be > 0", path)
            elif ct in ("capsule", "cylinder"):
                if not (isinstance(p.get("radius"), (int, float)) and p["radius"] > 0):
                    r.err("COLLISION_DIM", f"{ct}.radius must be > 0", path)
                if not (isinstance(p.get("half_length"), (int, float))
                        and p["half_length"] > 0):
                    r.err("COLLISION_DIM", f"{ct}.half_length must be > 0", path)
            elif ct == "convex_hull":
                if not p.get("vertices_file"):
                    r.err("COLLISION_DIM",
                          "convex_hull.vertices_file required", path)


def _v_total_mass(spec: dict, r: Report,
                  target: float = 75.0, tol: float = 0.20) -> None:
    """Invariant 8: total mass within ±tol of target. Severity: warning."""
    total = sum(l.get("inertial", {}).get("mass", 0)
                for l in spec.get("links", []))
    lo, hi = target * (1 - tol), target * (1 + tol)
    if not (lo <= total <= hi):
        r.warn("TOTAL_MASS",
               f"total mass {total:.2f} kg outside ±{tol*100:.0f}% of target {target} kg",
               "$.links[*].inertial.mass")


# -- Public API ---------------------------------------------------------------

def validate(spec: dict, *,
             target_mass: float = 75.0,
             mass_tolerance: float = 0.20) -> Report:
    """Run all §9 invariants. Returns a Report; report.dof_layout has §4.6 layout."""
    r = Report()
    _v_top(spec, r)
    _v_quaternions(spec, r)
    _v_links(spec, r)
    _v_topology(spec, r)
    _v_default_pose(spec, r)
    _v_collision(spec, r)
    _v_total_mass(spec, r, target_mass, mass_tolerance)
    r.dof_layout = derive_dof_layout(spec)
    return r


# -- CLI ----------------------------------------------------------------------

def _main(argv: List[str]) -> int:
    if len(argv) != 2:
        print("Usage: python validator.py <spec.json>", file=sys.stderr)
        return 2
    with open(argv[1]) as f:
        spec = json.load(f)
    report = validate(spec)
    print(report.summary())
    if report.dof_layout:
        layout = report.dof_layout
        actuated = [j for j in layout["joints"] if j["size"] > 0]
        print(f"\nDoF layout: total_dofs={layout['total_dofs']}, "
              f"actuated_joints={len(actuated)}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(_main(sys.argv))