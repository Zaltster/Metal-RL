# Humanoid Robot Spec — v1.0

Target: RL-ready humanoid for a custom Metal-based physics engine.
Format: JSON describing links and joints, with mesh and collision-hull files referenced alongside.
Status: **Locked.** All decisions resolved; ready for Blender build.

---

## 1. Conventions

| Item | Value |
|---|---|
| Units | SI: meters, kilograms, seconds, radians |
| Coordinate system | Right-handed, **+Z up, +X forward, +Y left**. This is the *engine's* convention. Blender shares Z-up but asset/bone forward axes vary, and Blender bone local axes are not engine joint frames — the exporter explicitly converts Blender object and bone transforms into engine coordinates. |
| Quaternion order | `[qx, qy, qz, qw]` (scalar-last). Matches glTF / Bullet / Unity / Swift+Metal. |
| Inertia tensor | 6 numbers in the body frame at the link's COM: `[Ixx, Iyy, Izz, Ixy, Ixz, Iyz]` |
| Angles | Radians; ranges expressed as `[min, max]` |
| Array ordering | `links[]` order = link index. `joints[]` order = joint index / action-vector order. Order is part of the contract — GPU buffer layout depends on it, so the exporter is deterministic and the JSON should not be reordered by hand. |

---

## 2. Top-level JSON schema

```json
{
  "schema_version": "1.0",
  "name": "humanoid_v1",
  "units": "SI",
  "links":   [ /* see §3 */ ],
  "joints":  [ /* see §4 */ ],
  "default_pose": { /* see §5 */ }
}
```

---

## 3. Link object

```json
{
  "name": "left_thigh",
  "inertial": {
    "mass": 8.5,
    "com": [0.0, 0.0, -0.18],
    "inertia": [0.18, 0.18, 0.025, 0.0, 0.0, 0.0]
  },
  "visual": [
    {
      "type": "mesh",
      "mesh_file": "meshes/left_thigh.obj",
      "transform": { "translation": [0, 0, 0], "rotation": [0, 0, 0, 1] }
    }
  ],
  "collision": [
    {
      "type": "capsule",
      "params":   { "radius": 0.07, "half_length": 0.18 },
      "transform":{ "translation": [0, 0, -0.18], "rotation": [0, 0, 0, 1] },
      "material": { "friction": 0.8, "restitution": 0.0 }
    }
  ]
}
```

**Collision shape types** and their `params`:

- `box` → `{ "half_extents": [hx, hy, hz] }`
- `sphere` → `{ "radius": r }`
- `capsule` → `{ "radius": r, "half_length": h }`, axis along local +Z
- `cylinder` → `{ "radius": r, "half_length": h }`, axis along local +Z
- `convex_hull` → `{ "vertices_file": "collision/<name>.obj" }` (vertices used; faces ignored)

**Material defaults** (used if `material` is omitted): `{ "friction": 0.8, "restitution": 0.0 }`. Specify per-shape so feet, hands, and torso can carry distinct contact behavior.

**Tiny-mass intermediate links** used for compound joints (wrist, ankle): `mass: 0.05 kg`, `com: [0, 0, 0]`, `inertia: [1e-5, 1e-5, 1e-5, 0, 0, 0]`. No visual or collision geometry. The small but nonzero inertia keeps the constraint solver well-conditioned — fully massless intermediates cause singular mass matrices.

---

## 4. Joint object

Joint types: `free` (6 DoF, root only), `revolute` (1 DoF hinge), `prismatic` (1 DoF slider), `spherical` (3 DoF), `fixed` (0 DoF, rigid weld).

### 4.1 Common fields (every joint type)

```json
{
  "name": "left_knee",
  "type": "revolute",
  "parent_link": "left_thigh",
  "child_link":  "left_shin",
  "anchor_in_parent": [0.0, 0.0, -0.36],
  "anchor_in_child":  [0.0, 0.0,  0.00],
  "frame_in_parent":  [0, 0, 0, 1],
  "frame_in_child":   [0, 0, 0, 1]
}
```

`frame_in_parent` and `frame_in_child` are **required on every joint, including revolute and prismatic**. They define the joint's local coordinate basis relative to each link, and joint axes are interpreted in this joint frame:

- `revolute` rotates about joint frame **+X**
- `prismatic` translates along joint frame **+X**
- `spherical` twists about joint frame **+Z**, with swing about **+X** and **+Y**
- `fixed` and `free` ignore axis convention

### 4.2 Revolute additions

```json
"limits":   { "position": [-2.4, 0.0] },
"dynamics": { "damping": 0.5, "armature": 0.01, "friction": 0.0, "stiffness": 0.0 },
"actuator": { "type": "torque", "max_force": 200.0, "max_velocity": 12.0 }
```

### 4.3 Prismatic additions

```json
"limits":   { "position": [-0.10, 0.10] },
"dynamics": { "damping": 5.0, "armature": 0.0, "friction": 0.0, "stiffness": 0.0 },
"actuator": { "type": "force", "max_force": 500.0, "max_velocity": 1.0 }
```

`max_force` is linear force (N); `max_velocity` is m/s. Not used by humanoid v1 but present in the schema so the engine implements it once.

### 4.4 Spherical additions

```json
"limits": {
  "swing_x": [-0.7, 0.7],
  "swing_y": [-0.5, 1.7],
  "twist_z": [-0.3, 0.3]
},
"dynamics": { "damping": 1.0, "armature": 0.02, "friction": 0.0, "stiffness": 0.0 },
"actuator": { "type": "torque", "max_force": [120, 200, 60], "max_velocity": 10.0 }
```

Spherical limits use a **swing–twist decomposition**: twist about the joint frame's local +Z, swing about +X and +Y. `max_force` is a 3-vector in the same swing/swing/twist order.

### 4.5 Free additions (root only)

```json
"limits":   null,
"dynamics": null,
"actuator": null
```

### 4.6 Action / DoF flattening

The action vector and joint-state vector are built by concatenating actuated DoFs in `joints[]` array order. Per joint type:

| Joint type | DoFs | Order within joint |
|---|---|---|
| `free`      | 0 | (root state lives in `default_pose.root_position` / `root_rotation`, never in the action vector) |
| `fixed`     | 0 | — |
| `revolute`  | 1 | scalar |
| `prismatic` | 1 | scalar |
| `spherical` | 3 | `[swing_x, swing_y, twist_z]` |

Joint-array index ≠ action-vector offset in general — spherical joints make this a DoF-offset problem, not a joint-index problem. The parser computes a `dof_layout` table on load:

```text
{ "total_dofs": 33,
  "joints": [
    {"name": "root",     "offset": 0, "size": 0},
    {"name": "j_lumbar", "offset": 0, "size": 3},
    {"name": "j_chest",  "offset": 3, "size": 3},
    ...
  ] }
```

This is **derived**, not authored — it's a function of `joints[]` plus the table above. The validator emits it as part of its load result so engine bindings can read it without re-implementing the rule.

---

## 5. Default pose

Initial state used when the engine instantiates the robot in the world. Required.

```json
"default_pose": {
  "root_position": [0.0, 0.0, 1.0],
  "root_rotation": [0.0, 0.0, 0.0, 1.0],
  "joint_positions": {
    "j_lumbar":           [0.0, 0.0, 0.0],
    "j_chest":            [0.0, 0.0, 0.0],
    "j_neck":             [0.0, 0.0, 0.0],
    "j_left_shoulder":    [0.0, 0.0, 0.0],
    "j_left_elbow":       -0.10,
    "j_left_pronation":    0.0,
    "j_left_wrist_flex":   0.0,
    "j_left_hip":         [0.0, 0.0, 0.0],
    "j_left_knee":        -0.05,
    "j_left_ankle_pitch":  0.0,
    "j_left_ankle_roll":   0.0
  }
}
```

Encoding by joint type:

- `revolute` / `prismatic` → scalar
- `spherical` → 3-vector `[swing_x, swing_y, twist_z]` in radians, matching the limit field order
- `fixed` → omitted (no DoF)
- `free` → represented as `root_position` and `root_rotation` at the top level

(Right-side joints mirror the left and are omitted from the example for brevity. The exported file lists all of them explicitly.)

---

## 6. Kinematic tree

| # | Link | Parent joint | Joint type | Notes |
|---|---|---|---|---|
| 1 | `pelvis` | `root` | free 6-DoF | Floating base |
| 2 | `lumbar` | `j_lumbar` | spherical | Lower spine |
| 3 | `chest` | `j_chest` | spherical | Upper spine |
| 4 | `head` | `j_neck` | spherical | |
| 5 | `left_upper_arm` | `j_left_shoulder` | spherical | |
| 6 | `left_lower_arm` | `j_left_elbow` | revolute | Flexion only |
| 7 | `left_wrist_twist` | `j_left_pronation` | revolute | Tiny-mass intermediate (0.05 kg, ε-diagonal inertia) |
| 8 | `left_hand` | `j_left_wrist_flex` | revolute | |
| 9–12 | mirror right arm | | | |
| 13 | `left_thigh` | `j_left_hip` | spherical | |
| 14 | `left_shin` | `j_left_knee` | revolute | |
| 15 | `left_ankle_tilt` | `j_left_ankle_pitch` | revolute | Tiny-mass intermediate |
| 16 | `left_foot` | `j_left_ankle_roll` | revolute | |
| 17–20 | mirror right leg | | | |

**Totals:** 20 links (16 substantive + 4 tiny-mass intermediates), 19 joints + free root. No fingers or toes in v1 — can be appended later as revolute chains off `left_hand` / `right_hand` / `left_foot` / `right_foot`.

---

## 7. Resolved decisions

| # | Decision | Resolution |
|---|---|---|
| 1 | Quaternion order | Scalar-last `[x, y, z, w]` |
| 2 | Spherical joint limits | Swing–twist decomposition |
| 3 | Wrist / ankle compound joints | Stacked revolutes with tiny-mass (0.05 kg) intermediate links |
| 4 | Spine | Two spherical joints (lumbar + thoracic) |
| 5 | Mesh delivery | Separate `.obj` files referenced by relative path |
| 6 | Mass defaults | Anthropometric, ~75 kg total |
| 7 | Visual mesh style | Geometric primitives first; sculpted mesh swap-in optional later |

---

## 8. Build plan

1. **Build the parser/validator + synthetic baseline first** (`validator.py`, `synthetic.py`). This gives the engine a working v1 humanoid before Blender is wired up, and is the test harness the Blender exporter is measured against.
2. Build armature + link empties in Blender for the full kinematic tree, with explicit joint frames (Empty axes set to engine convention) — not Blender bone local axes.
3. Place visual meshes — capsules and boxes sized to anthropometric defaults; swappable later.
4. Add collision primitives per link: capsules for limbs, boxes for torso/pelvis, convex hulls for hands and feet. Attach `material` to each.
5. Compute inertials per link from collision primitive density; sanity-check totals against the ~75 kg anthropometric distribution.
6. Set per-joint limits, damping, armature, friction, and actuator torque/velocity envelopes. Apply tiny-mass inertials to wrist/ankle intermediates.
7. Set the default pose (root height ~1.0 m, slight knee/elbow flex for solver stability at t=0).
8. Run the Blender Python exporter: walk the rig, convert Blender → engine coordinates, emit JSON + mesh + collision-hull files in deterministic link/joint order.
9. Round-trip sanity check: re-parse the JSON via the validator, verify every joint frame and inertial matches, render a debug skeleton view, and diff against the synthetic baseline.

---

## 9. Validator invariants

The parser/validator runs these checks on every spec it loads. Severity is `error` unless noted.

| # | Invariant | Notes |
|---|---|---|
| 1 | Every link has positive `mass` and a positive-definite inertia tensor. | Sylvester's criterion on the reconstructed 3×3 matrix. |
| 2 | Every non-root link has **exactly one** parent joint. | Catches orphan links and accidental duplicate parents. |
| 3 | Exactly **one** free root joint. | More than one ⇒ multi-rooted tree; zero ⇒ unanchored. |
| 4 | Joint anchors match in world space under the default pose. | Computed from FK over the tree at write-time by the exporter; the JSON-side validator emits a debug visualization rather than re-checking, since the JSON does not carry independent world-space link poses. |
| 5 | All `default_pose` values are inside the corresponding joint's limits. | Per-axis for spherical. |
| 6 | All quaternions are normalized within `1e-6`. | All rotation fields: visual / collision transforms, `frame_in_parent` / `frame_in_child`, `root_rotation`. |
| 7 | Link and joint array order is deterministic (parent before child). | Each joint's `child_link` index in `links[]` is greater than the parent's, ensuring topologically valid array order. |
| 8 | Total mass is within ±20% of the configured target (default 75 kg). | Severity: warning. |
| 9 | Every collision shape has valid type and positive dimensions. | Type ∈ {`box`, `sphere`, `capsule`, `cylinder`, `convex_hull`}; per-type params positive and well-shaped. |

The validator is the single source of truth for "is this asset loadable." Any new invariant added later (joint-limit reasonableness, mesh-file existence, actuator power vs. body mass sanity) goes through this same pipeline.
