# Humanoid Rigid-Body Contact Physics Plan

Target: a GPU-first 3D rigid-body robotics physics engine for many humanoid robots in parallel, with a visual renderer good enough to debug motion and export video.

This plan extends the current GPU elastic-joint humanoid environment. It does not replace the locked humanoid v1 asset spec.

## Ground Rules

- The product environment is Metal-only.
- Do not add a CPU humanoid simulator or CPU fallback.
- CPU code may parse assets, validate static invariants, and compute tiny hand-checkable expected values for tests.
- If the GPU path fails, fix the GPU path.
- Every change must be tested one step at a time.
- Every failure must be reproduced before it is fixed. Do not patch based on guesses.
- For each bug, keep or add a validation case that fails before the fix and passes after the fix.
- Documentation must describe what the repo actually does, not what we intend it to do later.

## Current Starting Point

Already present:

- `docs/humanoid_robot_spec.md`
- `docs/humanoid_v1_baseline.json`
- `tools/synthetic.py`
- `tools/validator.py`
- Swift humanoid JSON loader/validator
- GPU humanoid reset
- GPU elastic joint stepping
- GPU joint-limit clamp
- GPU forward kinematics
- GPU observation/reward/done export
- GPU rigid-body link constants
- GPU link linear/angular velocity buffers initialized on reset
- isolated GPU free-body integration for controlled repro cases
- GPU joint anchor constraint solver for controlled repro cases
- GPU joint anchor error measurement
- GPU joint motor/limit/damping/stiffness validation
- GPU ground-plane contact detection for box, sphere, capsule, and cylinder shapes
- GPU self-collision candidate detection for sphere-sphere and capsule-capsule pairs
- GPU ground contact solver with penetration correction, normal velocity projection, and tangent friction
- GPU standing-step path for many humanoid lanes with gravity, joint anchors, ground contact, and output export
- HTML replay renderer for selected env lanes

Current limitation:

- The humanoid is not yet a full rigid-body simulator.
- The full standing path is an early GPU integration path, not a production-quality humanoid dynamics solver.
- Joint anchors, fixed/revolute/prismatic angular constraints, motor rows, and joint-limit rows are solved with effective-mass rows and accumulated warm-started impulses. Spherical joints still permit three-axis rotation, with swing-cone and twist/per-axis limits instead of an orientation lock.
- Ground contact is implemented for one collision primitive per link against a plane.
- Self-collision candidate detection is implemented for sphere-sphere and capsule-capsule only. Self-contact solving, convex hull contact, and terrain meshes are not implemented.
- Contact solving uses a simple split position correction plus normal/tangent velocity projection for ground contacts. It is not yet a full iterative impulse solver with accumulated impulses and warm starting.

## Engine State Needed

Per environment lane:

```text
root pose
root velocity
root angular velocity
joint positions
joint velocities
link world positions
link world rotations
link linear velocities
link angular velocities
contact points
contact normals
contact impulses
actions
observations
rewards
dones
reset counts
```

Static robot constants:

```text
link masses
link inverse masses
link local inertias
link inverse inertias
joint parent/child indices
joint anchors
joint frames
joint limits
joint damping/stiffness/friction
actuator force limits
collision primitive metadata
collision material metadata
DoF layout
```

GPU layout should stay structure-of-arrays where possible:

```text
linkPositions[env, link, xyz]
linkRotations[env, link, xyzw]
linkLinearVelocities[env, link, xyz]
linkAngularVelocities[env, link, xyz]
jointPositions[env, dof]
jointVelocities[env, dof]
actions[env, dof]
```

## Physics Pipeline

A full step should eventually look like:

```text
1. apply actions / motors
2. apply passive spring-damper joint forces
3. apply gravity
4. predict unconstrained body velocities
5. solve joint constraints
6. detect contacts
7. solve contact normal impulses
8. solve friction impulses
9. integrate positions and rotations
10. project/clamp joint limits if needed
11. update FK/debug link transforms
12. export observations/rewards/dones
```

This should be split into small kernels, not one giant kernel.

## Milestone 1: GPU Rigid Body State

Goal: add real per-link velocities and inertial constants without changing behavior yet.

Status: implemented. The GPU environment now uploads link mass/inertia constants, keeps link linear/angular velocity buffers, initializes them on reset, and validates the buffers in the smoke harness. This milestone intentionally does not change motion behavior.

Work:

- Add link mass, inverse mass, inertia, inverse inertia buffers.
- Add link linear/angular velocity buffers.
- Add reset kernel initialization for all link velocities.
- Add output/readback helpers for selected lanes.

Validation:

- Reset gives finite poses and zero velocities.
- Buffer shapes match `envCount * linkCount`.
- Inertias are positive and inverse inertias are finite.
- Existing humanoid replay still works.

Failure discipline:

- If reset output is wrong, save the exact JSON + env count + buffer index that reproduces it.
- Fix only after the failure is reproduced in the smoke harness.

## Milestone 2: Free-Body Integration

Goal: integrate independent rigid bodies under gravity with no joints/contact.

Status: implemented. The smoke harness validates zero-gravity invariance, gravity-driven height decrease, quaternion normalization under angular velocity, and repeated finite integration. The same GPU integration kernel is also used by the early `stepStanding` path.

Work:

- Add velocity integration:

```text
v += dt * gravity
x += dt * v
q += dt * 0.5 * omega_quat * q
normalize(q)
```

- Add a mode/kernel for isolated free-body integration.

Validation:

- With zero gravity, poses remain unchanged.
- With gravity, height decreases monotonically.
- Quaternion norm stays near 1.
- No NaN/Inf values after many steps.

Failure discipline:

- Reproduce quaternion drift or NaN with a minimal env count before fixing.

## Milestone 3: Joint Constraint Solver

Goal: connect links as rigid bodies using joint constraints.

Status: implemented for the current sequential-impulse scope. The GPU path measures joint anchor error, deliberately reproduces broken joint states in the smoke harness, solves anchor attachment constraints with iterative effective-mass rows, and validates that more iterations do not increase error. It also uploads joint frame quaternions, solves fixed/revolute/prismatic angular rows with effective mass, and stores accumulated anchor/angular row impulses for warm starting. Spherical joints intentionally preserve anchors while allowing three-axis rotation; their motion bounds are handled by Milestone 4 swing/twist limits.

Work:

- Implemented: anchor constraint rows for fixed, revolute, prismatic, and spherical joints.
- Implemented: effective-mass anchor rows include linear mass, anchor offsets, and inverse inertia.
- Implemented: accumulated/warm-started anchor-row impulses.
- Implemented: fixed joint orientation rows use effective mass and accumulated/warm-started angular impulses.
- Implemented: revolute hinge-axis rows use effective mass and accumulated/warm-started angular impulses for off-axis rotation.
- Implemented: prismatic slider semantics: perpendicular anchor projection, allowed slider-axis offset, and effective-mass orientation rows.
- Implemented: free root remains unconstrained.
- Implemented: long zero-action `stepStanding` stress rollout keeps state, contact penetration, and accumulated impulses finite.
- Implemented: adversarial one-joint repro covers high mass ratio, tiny child inertia, off-center anchor displacement, high initial velocity, and a low iteration count.
- Implemented: nonzero-action chain/contact standing stress keeps exported state, contact penetration, solver diagnostics, and accumulated motor/limit impulses finite.
- Remaining: learned-control stress before claiming production-grade humanoid behavior.

Recommended solver:

```text
sequential impulse / projected Gauss-Seidel
fixed small iteration count
warm starting for stable rows
```

Validation:

- Implemented: joint anchor error decreases from an intentionally broken GPU state.
- Implemented: extra solver iterations do not increase measured anchor error.
- Implemented: off-center anchor row produces angular velocity through inverse inertia.
- Implemented: anchor-row impulses accumulate and reset clears them.
- Implemented: angular-row impulses accumulate and reset clears them.
- Implemented: fixed joint reduces deliberately injected child orientation error in a generated one-joint repro.
- Implemented: revolute joint reduces deliberately injected hinge-axis error and removes off-axis relative angular velocity in a generated repro.
- Implemented: prismatic joint reduces perpendicular anchor error, preserves slider-axis offset, and reduces orientation error in a generated repro.
- Implemented: spherical joint preserves anchor position and permits 3-axis rotation.
- Implemented: longer contact-coupled zero-action standing stress keeps state and impulses finite.
- Implemented: adversarial one-joint mass/inertia/anchor/velocity repro reduces error and reports clean solver diagnostics.
- Implemented: nonzero-action multi-link standing stress with contact and gravity keeps state finite and contacts resolved within tolerance.
- Remaining: learned-control stress before claiming production-grade humanoid joint behavior.

Failure discipline:

- For every joint bug, produce a one-joint repro scene first.
- Do not debug the full humanoid until the one-joint repro passes.

## Milestone 4: Joint Limits And Motors

Goal: make humanoid-style limits and actuators physically meaningful.

Status: implemented for the current sequential-impulse scope. Actions are converted to bounded motor effort, joint velocity is capped by actuator `max_velocity`, hard limits clamp joint position, damping reduces velocity, and stiffness pulls displaced joints toward defaults. The GPU rigid-body path solves revolute/spherical motor rows against parent/child angular velocities and prismatic motor rows against parent/child linear velocities with effective mass, accumulated impulses, and warm starting. Joint-limit rows are one-sided effective-mass impulse rows with accumulated impulses. Spherical joints use swing-cone and twist/per-axis limits rather than an orientation lock.

Work:

- Implemented: limit constraints for revolute/prismatic/spherical DoFs.
- Implemented: actions convert to torque/force through actuator `max_force`.
- Implemented: velocity clamp through actuator `max_velocity`.
- Implemented: passive stiffness/damping.
- Implemented: effective-mass rigid-body motor rows for revolute, spherical, and prismatic joints.
- Implemented: accumulated/warm-started motor impulses.
- Implemented: effective-mass one-sided joint-limit rows for revolute, spherical, and prismatic DoFs.
- Implemented: accumulated/warm-started limit impulses.
- Implemented: spherical swing-cone clamp plus twist/per-axis limits.
- Implemented: GPU solver diagnostics for max constraint error, max impulse magnitude, non-finite row count, and failed-row count.

Validation:

- Implemented: a joint cannot pass its limit under large action.
- Implemented: a damped joint loses velocity over time.
- Implemented: a stiff joint returns toward its default pose using a generated repro asset.
- Implemented: max force changes acceleration in the expected direction.
- Implemented: max velocity clamps saturated motor output.
- Implemented: revolute motor creates parent/child relative angular velocity in the expected direction.
- Implemented: limit correction creates opposite-direction rigid-body angular velocity when pushed past a limit.
- Implemented: spherical motor creates axis-specific relative angular velocity.
- Implemented: spherical swing cone clamps combined swing instead of only independent axes.
- Implemented: spherical limit row creates opposite-direction rigid-body angular velocity when pushed past a swing axis limit.
- Implemented: generated prismatic repro creates parent/child relative linear velocity in the expected direction.
- Implemented: reset clears accumulated motor/limit impulses and solver diagnostics.
- Implemented: solver diagnostics remain finite and report no failed/non-finite rows in focused motor/limit, adversarial one-joint, and nonzero chain/contact stress tests.

Failure discipline:

- Reproduce limit violations with one joint before testing the whole humanoid.

## Milestone 5: Collision Detection

Goal: generate contact candidates for robot-ground and robot-self collisions.

Status: implemented for the initial shape set. The GPU path uploads one collision primitive per link, detects ground-plane contacts for box, sphere, capsule, and cylinder primitives, and detects self-collision candidates for sphere-sphere and capsule-capsule pairs. The smoke harness verifies penetrating/separated cases and checks generated two-link self-collision scenes produce exactly one active pair.

Start with:

- Implemented: sphere-plane
- Implemented: capsule-plane
- Implemented: box-plane
- Implemented: cylinder-plane through the same axial primitive path
- Implemented: sphere-sphere
- Implemented: capsule-capsule

Later:

- box-box
- convex hull-plane
- convex hull-convex hull
- static triangle mesh terrain

Do not use visual triangle meshes for dynamic robot collision.

Validation:

- Implemented: contact normal points from ground into body.
- Implemented: contact penetration is positive only when overlapping.
- Implemented: contact point is finite for simple primitive-plane cases.
- Implemented: contact normal points from the first self-collision body toward the second.
- Implemented: self-collision penetration is positive only when overlapping.
- Implemented: contact point is finite for sphere-sphere and capsule-capsule.
- Implemented: generated two-link self-collision repros produce exactly one active pair.

Failure discipline:

- Store failing shape pair, transforms, and expected relation.
- Add the shape-pair repro to smoke checks before fixing.

## Milestone 6: Contact Solver

Goal: prevent bodies from penetrating the ground and support friction.

Status: implemented for ground-plane contacts. The GPU solver consumes the contact buffers, corrects penetration, removes inward normal velocity, and applies bounded tangent friction. It is intentionally simple and does not yet accumulate impulses or warm start.

Work:

- Implemented: build contact rows from ground-plane contact buffers.
- Implemented: solve normal velocity with non-negative projection.
- Implemented: split position correction for penetration.
- Implemented: bounded friction tangent velocity reduction.
- Remaining: accumulated impulses and warm starting.
- Remaining: restitution; default remains 0.

Validation:

- Implemented: a falling box lands on the plane and does not retain penetration at small dt.
- Implemented: resting contacts do not gain energy.
- Implemented: friction slows horizontal sliding.
- Remaining: sphere/capsule falling loops.
- Remaining: foot boxes supporting articulated vertical load as a stable standing policy/control test.

Failure discipline:

- If contact explodes, reproduce with a single primitive and one env lane.
- Keep the failing case as a regression test.

## Milestone 7: Full Humanoid Standing Environment

Goal: run many humanoids in parallel with ground contact.

Status: implemented as an early GPU standing-step path. `stepStanding` runs joint motors, FK, gravity, joint anchor constraints, ground contact detection/solve, pelvis-to-root sync, and observation/reward/done export. The smoke harness runs 1024 lanes for repeated zero-action standing steps and checks finite outputs, contact resolution, and feet reaching the ground. This is not yet a stable learned standing task or a full humanoid controller.

Environment:

```text
HumanoidStandEnv
gravity
ground plane
humanoid reset pose
torque or target-action inputs
upright reward
height reward
energy penalty
fall termination
```

Validation:

- Implemented: reset starts above ground.
- Implemented: feet reach the ground under gravity/contact stepping.
- Implemented: pelvis/link heights remain finite.
- Implemented: rewards/dones are finite.
- Implemented: 1024 env lanes step without CPU fallback.
- Implemented: deterministic replay test specific to `stepStanding` compares two reset/spec/action-matched GPU runs, including outputs, link state, contact penetrations, and joint state.

Failure discipline:

- If full humanoid fails, reduce to the smallest subsystem:
  - one body
  - one joint
  - one limb
  - one contact pair
  - then full humanoid

## Visual Renderer

The renderer is required, not optional.

Current renderer:

- HTML/Canvas replay from selected env lanes.
- Shows ground plane.
- Shows link positions.
- Shows parent-child skeleton.
- Shows frame, reward, done, reset count, and env index.

Current renderer output:

```text
humanoid_replay.html
```

Remaining renderer work:

- Show primitive collision shapes.
- Show contact points and normals.
- Show joint limit violations in color.
- Export replay JSON alongside the HTML replay.

Future video path:

1. Generate replay JSON/HTML from GPU readback of selected lanes.
2. Open replay in browser for manual inspection.
3. Add a Playwright or browser automation script to capture frames.
4. Encode frames to MP4 with `ffmpeg`.

Example future command:

```bash
./scripts/run_humanoid_demo.sh
./scripts/render_humanoid_video.sh humanoid-demo/.build/humanoid_replay.json /tmp/humanoid.mp4
```

Validation:

- Implemented: replay HTML is written by the humanoid smoke check.
- Implemented: replay frame writer rejects empty input and shape mismatches.
- Remaining: replay has the expected number of frames.
- Remaining: no frame has NaN/Inf positions.
- Remaining: camera bounds include the humanoid.
- Remaining: contact markers line up with collision shapes.
- Remaining: video generation fails loudly if browser/ffmpeg is missing.

## Smoke Harness Requirements

Every milestone must add at least one smoke check.

Required checks:

- compile Metal kernels
- load humanoid JSON
- validate DoF layout
- reset GPU buffers
- run one step
- run repeated steps
- verify finite state
- verify shape-specific contact cases
- write replay artifact

The smoke harness prints explicit success lines, currently including:

```text
humanoid gpu rigid-body state, free-body integration, joint anchor constraints, motors/limits, ground/self contacts, contact solver, standing env, FK, and replay passed
```

## Implementation Order

1. Rigid body state buffers.
2. Free-body gravity integration.
3. One-joint constraint repro scenes.
4. Joint limits and motors.
5. Primitive contact detection.
6. Contact solver.
7. Humanoid ground contact.
8. Better renderer with collision/contact overlays.
9. Video export script.
10. RL environment integration.

Do not skip directly to the full humanoid. The full humanoid is the integration test, not the debugging surface.
