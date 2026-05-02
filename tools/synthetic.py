"""Synthetic anthropometric humanoid_v1 generator.

Emits a valid humanoid_v1 spec JSON without Blender, derived from approximate
anthropometric defaults (~75 kg / 1.75 m male reference). This is the engine's
v1 baseline asset and the test fixture the Blender exporter is measured
against.

Run:
    python synthetic.py                    # writes to stdout
    python synthetic.py humanoid_v1.json   # writes to file
"""
from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Tuple

# -- Anthropometric defaults --------------------------------------------------
# (mass kg, length m, radius m). Approximations for RL — physically plausible,
# not biomechanically exact.

SEG: Dict[str, Tuple[float, float, float]] = {
    "pelvis":     (9.0,  0.18, 0.12),
    "lumbar":     (6.0,  0.12, 0.10),
    "chest":      (12.0, 0.22, 0.14),
    "head":       (5.0,  0.18, 0.09),
    "upper_arm":  (2.5,  0.30, 0.05),
    "lower_arm":  (1.5,  0.27, 0.04),
    "hand":       (0.5,  0.10, 0.04),
    "thigh":      (8.5,  0.40, 0.07),
    "shin":       (4.0,  0.40, 0.05),
    "foot":       (1.0,  0.22, 0.05),
}

INTERMEDIATE_MASS = 0.05
INTERMEDIATE_INERTIA = [1e-5, 1e-5, 1e-5, 0.0, 0.0, 0.0]


# -- Inertia formulas ---------------------------------------------------------

def capsule_inertia(mass: float, radius: float, half_length: float) -> List[float]:
    """Solid capsule, axis = local +Z. Returns [Ixx, Iyy, Izz, 0, 0, 0]."""
    L = 2.0 * half_length
    R = radius
    # Split mass between cylinder and 2 hemispheres by volume
    Vc = math.pi * R * R * L
    Vs = (4.0 / 3.0) * math.pi * R ** 3
    mc = mass * Vc / (Vc + Vs)
    ms = mass - mc
    # Cylinder
    Iz_c = 0.5 * mc * R * R
    Ix_c = (1.0 / 12.0) * mc * (3 * R * R + L * L)
    # 2 hemispheres at z = ±L/2
    Iz_s = (2.0 / 5.0) * ms * R * R
    Ix_s = Iz_s + ms * (L / 2.0) ** 2
    return [Ix_c + Ix_s, Ix_c + Ix_s, Iz_c + Iz_s, 0.0, 0.0, 0.0]


def box_inertia(mass: float, hx: float, hy: float, hz: float) -> List[float]:
    """Solid box, half-extents [hx, hy, hz]."""
    Ixx = (mass / 3.0) * (hy * hy + hz * hz)
    Iyy = (mass / 3.0) * (hx * hx + hz * hz)
    Izz = (mass / 3.0) * (hx * hx + hy * hy)
    return [Ixx, Iyy, Izz, 0.0, 0.0, 0.0]


import math  # noqa: E402  (used by capsule_inertia)


# -- Link / joint constructors ------------------------------------------------

def link_capsule(name: str, mass: float, radius: float, half_length: float,
                 com_z: float = 0.0) -> Dict[str, Any]:
    return {
        "name": name,
        "inertial": {
            "mass": mass,
            "com": [0.0, 0.0, com_z],
            "inertia": capsule_inertia(mass, radius, half_length),
        },
        "visual": [{
            "type": "capsule",
            "params": {"radius": radius, "half_length": half_length},
            "transform": {"translation": [0.0, 0.0, com_z],
                          "rotation": [0.0, 0.0, 0.0, 1.0]},
        }],
        "collision": [{
            "type": "capsule",
            "params": {"radius": radius, "half_length": half_length},
            "transform": {"translation": [0.0, 0.0, com_z],
                          "rotation": [0.0, 0.0, 0.0, 1.0]},
            "material": {"friction": 0.8, "restitution": 0.0},
        }],
    }


def link_box(name: str, mass: float, hx: float, hy: float, hz: float,
             friction: float = 0.8) -> Dict[str, Any]:
    return {
        "name": name,
        "inertial": {
            "mass": mass,
            "com": [0.0, 0.0, 0.0],
            "inertia": box_inertia(mass, hx, hy, hz),
        },
        "visual": [{
            "type": "box",
            "params": {"half_extents": [hx, hy, hz]},
            "transform": {"translation": [0.0, 0.0, 0.0],
                          "rotation": [0.0, 0.0, 0.0, 1.0]},
        }],
        "collision": [{
            "type": "box",
            "params": {"half_extents": [hx, hy, hz]},
            "transform": {"translation": [0.0, 0.0, 0.0],
                          "rotation": [0.0, 0.0, 0.0, 1.0]},
            "material": {"friction": friction, "restitution": 0.0},
        }],
    }


def link_intermediate(name: str) -> Dict[str, Any]:
    return {
        "name": name,
        "inertial": {
            "mass": INTERMEDIATE_MASS,
            "com": [0.0, 0.0, 0.0],
            "inertia": list(INTERMEDIATE_INERTIA),
        },
        "visual": [],
        "collision": [],
    }


def joint_free() -> Dict[str, Any]:
    return {
        "name": "root", "type": "free",
        "parent_link": "world", "child_link": "pelvis",
        "anchor_in_parent": [0.0, 0.0, 0.0],
        "anchor_in_child":  [0.0, 0.0, 0.0],
        "frame_in_parent":  [0.0, 0.0, 0.0, 1.0],
        "frame_in_child":   [0.0, 0.0, 0.0, 1.0],
        "limits": None, "dynamics": None, "actuator": None,
    }


def joint_revolute(name: str, parent: str, child: str,
                   anchor_p: List[float], anchor_c: List[float],
                   limits: Tuple[float, float],
                   max_torque: float, max_vel: float = 12.0,
                   damping: float = 0.5, armature: float = 0.01) -> Dict[str, Any]:
    return {
        "name": name, "type": "revolute",
        "parent_link": parent, "child_link": child,
        "anchor_in_parent": list(anchor_p),
        "anchor_in_child":  list(anchor_c),
        "frame_in_parent":  [0.0, 0.0, 0.0, 1.0],
        "frame_in_child":   [0.0, 0.0, 0.0, 1.0],
        "limits":   {"position": list(limits)},
        "dynamics": {"damping": damping, "armature": armature,
                     "friction": 0.0, "stiffness": 0.0},
        "actuator": {"type": "torque",
                     "max_force": max_torque, "max_velocity": max_vel},
    }


def joint_spherical(name: str, parent: str, child: str,
                    anchor_p: List[float], anchor_c: List[float],
                    swing_x: Tuple[float, float],
                    swing_y: Tuple[float, float],
                    twist_z: Tuple[float, float],
                    max_torque: Tuple[float, float, float],
                    max_vel: float = 10.0,
                    damping: float = 1.0,
                    armature: float = 0.02) -> Dict[str, Any]:
    return {
        "name": name, "type": "spherical",
        "parent_link": parent, "child_link": child,
        "anchor_in_parent": list(anchor_p),
        "anchor_in_child":  list(anchor_c),
        "frame_in_parent":  [0.0, 0.0, 0.0, 1.0],
        "frame_in_child":   [0.0, 0.0, 0.0, 1.0],
        "limits": {"swing_x": list(swing_x),
                   "swing_y": list(swing_y),
                   "twist_z": list(twist_z)},
        "dynamics": {"damping": damping, "armature": armature,
                     "friction": 0.0, "stiffness": 0.0},
        "actuator": {"type": "torque",
                     "max_force": list(max_torque), "max_velocity": max_vel},
    }


# -- Build --------------------------------------------------------------------

def build_humanoid() -> Dict[str, Any]:
    L_lumbar = SEG["lumbar"][1]
    L_chest  = SEG["chest"][1]
    L_uarm   = SEG["upper_arm"][1]
    L_larm   = SEG["lower_arm"][1]
    L_thigh  = SEG["thigh"][1]
    L_shin   = SEG["shin"][1]

    links: List[Dict[str, Any]] = []
    joints: List[Dict[str, Any]] = []

    # ----- Links (parent-before-child order) -----
    pmass = SEG["pelvis"][0]
    links.append(link_box("pelvis", pmass, 0.12, 0.10, 0.06))

    lmass, llen, lrad = SEG["lumbar"]
    links.append(link_capsule("lumbar", lmass, lrad, llen / 2, com_z=llen / 2))

    cmass, clen, _ = SEG["chest"]
    links.append(link_box("chest", cmass, 0.18, 0.12, clen / 2))

    hmass, hlen, hrad = SEG["head"]
    links.append(link_capsule("head", hmass, hrad, hlen / 2, com_z=hlen / 2))

    for side in ("left", "right"):
        umass, ulen, urad = SEG["upper_arm"]
        lamass, lalen, larad = SEG["lower_arm"]
        hmass2, _, hrad2 = SEG["hand"]
        links.append(link_capsule(f"{side}_upper_arm", umass, urad, ulen / 2,
                                  com_z=-ulen / 2))
        links.append(link_capsule(f"{side}_lower_arm", lamass, larad, lalen / 2,
                                  com_z=-lalen / 2))
        links.append(link_intermediate(f"{side}_wrist_twist"))
        links.append(link_capsule(f"{side}_hand", hmass2, hrad2, 0.05,
                                  com_z=-0.05))

    for side in ("left", "right"):
        tmass, tlen, trad = SEG["thigh"]
        smass, slen, srad = SEG["shin"]
        fmass, flen, _ = SEG["foot"]
        links.append(link_capsule(f"{side}_thigh", tmass, trad, tlen / 2,
                                  com_z=-tlen / 2))
        links.append(link_capsule(f"{side}_shin", smass, srad, slen / 2,
                                  com_z=-slen / 2))
        links.append(link_intermediate(f"{side}_ankle_tilt"))
        links.append(link_box(f"{side}_foot", fmass, flen / 2, 0.05, 0.03,
                              friction=1.0))

    # ----- Joints -----
    joints.append(joint_free())

    # Spine
    joints.append(joint_spherical("j_lumbar", "pelvis", "lumbar",
        anchor_p=[0, 0, 0.06], anchor_c=[0, 0, 0],
        swing_x=(-0.3, 0.3), swing_y=(-0.3, 0.3), twist_z=(-0.4, 0.4),
        max_torque=(150, 150, 80)))
    joints.append(joint_spherical("j_chest", "lumbar", "chest",
        anchor_p=[0, 0, L_lumbar], anchor_c=[0, 0, -L_chest / 2],
        swing_x=(-0.3, 0.3), swing_y=(-0.3, 0.3), twist_z=(-0.5, 0.5),
        max_torque=(120, 120, 60)))
    joints.append(joint_spherical("j_neck", "chest", "head",
        anchor_p=[0, 0, L_chest / 2], anchor_c=[0, 0, 0],
        swing_x=(-0.6, 0.6), swing_y=(-0.6, 0.6), twist_z=(-0.6, 0.6),
        max_torque=(20, 20, 15)))

    # Arms (left then right) — must come after chest in joint order, since chest
    # is the parent. links[]: chest is index 2, so all arm joints reference an
    # earlier link index. Same for legs vs pelvis.
    for side, sx in (("left", +1.0), ("right", -1.0)):
        joints.append(joint_spherical(f"j_{side}_shoulder",
            "chest", f"{side}_upper_arm",
            anchor_p=[0.18 * sx, 0, L_chest / 2 - 0.02], anchor_c=[0, 0, 0],
            swing_x=(-2.0, 2.0), swing_y=(-1.5, 1.5), twist_z=(-1.5, 1.5),
            max_torque=(80, 80, 40)))
        joints.append(joint_revolute(f"j_{side}_elbow",
            f"{side}_upper_arm", f"{side}_lower_arm",
            anchor_p=[0, 0, -L_uarm], anchor_c=[0, 0, 0],
            limits=(-2.5, 0.0), max_torque=80.0))
        joints.append(joint_revolute(f"j_{side}_pronation",
            f"{side}_lower_arm", f"{side}_wrist_twist",
            anchor_p=[0, 0, -L_larm], anchor_c=[0, 0, 0],
            limits=(-1.5, 1.5), max_torque=15.0,
            damping=0.2, armature=0.001))
        joints.append(joint_revolute(f"j_{side}_wrist_flex",
            f"{side}_wrist_twist", f"{side}_hand",
            anchor_p=[0, 0, 0], anchor_c=[0, 0, 0],
            limits=(-1.0, 1.0), max_torque=10.0,
            damping=0.2, armature=0.001))

    # Legs
    for side, sx in (("left", +1.0), ("right", -1.0)):
        joints.append(joint_spherical(f"j_{side}_hip",
            "pelvis", f"{side}_thigh",
            anchor_p=[0.10 * sx, 0, -0.04], anchor_c=[0, 0, 0],
            swing_x=(-0.7, 0.7), swing_y=(-0.5, 1.7), twist_z=(-0.4, 0.4),
            max_torque=(150, 200, 60)))
        joints.append(joint_revolute(f"j_{side}_knee",
            f"{side}_thigh", f"{side}_shin",
            anchor_p=[0, 0, -L_thigh], anchor_c=[0, 0, 0],
            limits=(0.0, 2.4), max_torque=200.0))
        joints.append(joint_revolute(f"j_{side}_ankle_pitch",
            f"{side}_shin", f"{side}_ankle_tilt",
            anchor_p=[0, 0, -L_shin], anchor_c=[0, 0, 0],
            limits=(-0.7, 0.7), max_torque=80.0,
            damping=0.5, armature=0.005))
        joints.append(joint_revolute(f"j_{side}_ankle_roll",
            f"{side}_ankle_tilt", f"{side}_foot",
            anchor_p=[0, 0, 0], anchor_c=[0, 0, 0.03],
            limits=(-0.4, 0.4), max_torque=60.0,
            damping=0.5, armature=0.005))

    # Default pose: neutral standing, slight knee/elbow flex
    jp: Dict[str, Any] = {
        "j_lumbar": [0.0, 0.0, 0.0],
        "j_chest":  [0.0, 0.0, 0.0],
        "j_neck":   [0.0, 0.0, 0.0],
    }
    for side in ("left", "right"):
        jp[f"j_{side}_shoulder"]    = [0.0, 0.0, 0.0]
        jp[f"j_{side}_elbow"]       = -0.10
        jp[f"j_{side}_pronation"]   = 0.0
        jp[f"j_{side}_wrist_flex"]  = 0.0
        jp[f"j_{side}_hip"]         = [0.0, 0.0, 0.0]
        jp[f"j_{side}_knee"]        = 0.05
        jp[f"j_{side}_ankle_pitch"] = 0.0
        jp[f"j_{side}_ankle_roll"]  = 0.0

    return {
        "schema_version": "1.0",
        "name": "humanoid_v1",
        "units": "SI",
        "links": links,
        "joints": joints,
        "default_pose": {
            "root_position": [0.0, 0.0, 1.0],
            "root_rotation": [0.0, 0.0, 0.0, 1.0],
            "joint_positions": jp,
        },
    }


# -- CLI ----------------------------------------------------------------------

def _main(argv: List[str]) -> int:
    spec = build_humanoid()
    if len(argv) >= 2:
        with open(argv[1], "w") as f:
            json.dump(spec, f, indent=2)
        print(f"Wrote {argv[1]}", file=sys.stderr)
    else:
        json.dump(spec, sys.stdout, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))