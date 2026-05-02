# Production Physics Goal

One-shot spec to take the Metal-first humanoid physics engine to production-grade.

## Ground Rules

- Metal-first. No MLX, PyTorch, CUDA, or hidden tensor frameworks. No CPU humanoid simulator. No CPU fallback. CPU is allowed only for asset parsing, static validation, hand-checkable expected values for tests, and tiny readback comparisons.
- Repo root: `/Users/smile/pufer/Environment`.
- Do not revert any current dirty-worktree changes. Build forward.
- Every change must be reproduced as a focused smoke test before claiming a feature works. Add negative tests where practical (toggle a knob, confirm behavior changes).
- Do not overclaim in docs. If something is partial, say partial.
- At the end, `docs/HUMANOID_RIGID_BODY_PHYSICS_PLAN.md` must match actual repo state. No aspirational status text.
- Keep kernels structure-of-arrays and Metal-friendly. If `humanoid_kernels.metal` grows past ~2500 lines, split into per-stage files (integrate / joint_solver / contact_solver / collision_detect / output) with shared headers, and update both build scripts.
- Additive changes only for risky shell actions. Never `--no-verify`, never force-push, never `rm -rf` the worktree.
- Each milestone ends with all four validation commands passing:
  - `python3 tools/validator.py docs/humanoid_v1_baseline.json`
  - `./scripts/run_humanoid_demo.sh`
  - `./metal-smoke-check/scripts/build_and_run.sh`
  - `git diff --check`
- If any fail, fix the GPU path. Do not move on with failing tests.

## Execution

Do milestones A through G in order. Do not skip ahead. After each milestone, run the four validation commands. If a milestone item is genuinely impractical, leave the code as-is, document the limitation honestly in the plan doc, and continue. Do not silently drop work.

Bias toward correctness over breadth. Ship fewer milestones fully done over more milestones half-done. Half-done physics silently destroys RL training and is worse than missing features.

## Milestone A — Solver correctness foundation

1. Replace the per-row scalar joint solver with a proper effective-mass block solver: build Jacobian rows for anchor + angular constraints, solve sequentially with PGS over the row set, but with correct block effective-mass (parent+child anchor offsets, world-space inverse inertia tensors). Keep accumulated impulses, but make warm-starting work across frames. Do not clear impulse buffers each frame — clear only on reset. Add cross-frame contact-ID matching so warm-start applies.
2. Add split-impulse / pseudo-velocity position correction. Position errors correct a separate pseudo-velocity, never the real velocity. Remove the `linearSlop` hack from contact normal target velocity.
3. Per-island detection + sleeping. Resting islands cost ~0 GPU time. Wake on action input, contact change, or gravity > threshold.

Tests
- Single revolute, prismatic, fixed, spherical one-joint repros remain passing AND solver converges in fewer iterations than before.
- Cross-frame warm-start test: stack two boxes, verify normal impulse magnitudes carry over and total iterations to converge each frame drops after frame 1.
- Sleeping test: 256 humanoids settle, then per-step kernel time drops measurably (record into solver diagnostics).

## Milestone B — Multi-contact, manifold persistence, friction cone

4. Replace single-contact-per-link/pair with a contact manifold (up to 4 points per pair). For box-plane and box-box, generate clipped polygon contact points. For capsule-plane, two endpoints if both touch. Persist each contact's feature ID across frames so impulses warm-start.
5. Replace two independent clamped tangent rows with a friction cone: project the combined tangent impulse magnitude onto the friction disk `|f_t| <= mu * |f_n|`.

Tests
- Box resting flat on plane stays planted under sideways nudge below `mu*m*g`; slides above it. Quantify with a sweep.
- Stack 3 boxes, verify final heights settle without slow penetration over 600 steps.
- Friction direction is independent of axis order (rotate world 45°, same trajectory).

## Milestone C — Real convex collision

6. Replace the AABB-proxy `convex_hull` narrowphase with GJK distance + EPA penetration depth + Sutherland–Hodgman polygon clipping for manifold generation. GPU-side. Hulls come from the existing OBJ loader.
7. Real box-box (SAT) and convex-convex.
8. Mixed primitives: sphere-box, capsule-box, capsule-cylinder.
9. Sphere-mesh and capsule-mesh against a static triangle-mesh terrain asset (BVH built CPU-side at load, traversed on GPU).

Tests
- Two convex hulls (asymmetric tetrahedra at arbitrary rotation) produce correct contact normal vs a CPU reference computed at fixture time.
- Capsule rolling on a tilted box stays on top under gravity + friction.
- Triangle-mesh terrain: drop a humanoid on a 2-triangle ramp, verify it slides at the expected angle under given mu.

## Milestone D — Spatial broadphase + colored self-contact

10. Dynamic AABB tree on GPU (or sweep-and-prune) for contact pair generation. Replace the O(n²) self-pair list.
11. Replace the per-pair serial self-contact dispatch in `stepStanding` with graph-coloring batched dispatch: pairs that share no link run in parallel.
12. Wire self-contact through the same warm-started effective-mass solver used for ground contacts.

Tests
- 1024 humanoids step with self-collision on; broadphase scales sub-quadratically in active pair count.
- Colored self-contact dispatch produces bit-exact same state as the serial dispatch for at least one fixed seed.

## Milestone E — Determinism, robustness, soak

13. Replace any nondeterministic reduction (`atomicAdd<float>`) with tree reductions. Pin traversal order. Bit-exact replay across runs on the same device.
14. Substepping: `stepStanding` takes a `substeps` parameter; physics runs at `substeps * dt` with one observation export.
15. Quaternion exponential map for angular integration.
16. NaN/Inf trap kernel: any non-finite value sets a per-env error flag that propagates to a host-readable diagnostic.
17. Robustness suite (must pass):
    - Mass ratio 1000:1 with stiff joint.
    - Stacked-bodies no-penetration over 600 steps.
    - Rolling sphere on incline with mu = 0.3.
    - Fast spinning body, |ω| = 50 rad/s, no quaternion blowup.
    - dt = 1/30 runs cleanly on the standing scene.
    - 30-second soak (proxy for 8h): 256 humanoids × 30k steps random nonzero actions, zero NaN, no monotonic penetration growth, no energy growth in islands flagged as resting.

## Milestone F — Humanoid env + RL completion

18. Single canonical `step()` on `HumanoidMetalEnvironment` that replaces the special-case `stepStanding`. Substeps, fixed timestep, full physics pipeline. Keep `stepStanding` as a thin back-compat wrapper.
19. Promote root linear and angular velocity to first-class state buffers. Stop deriving them from pelvis position diffs.
20. Action-space scaling per DoF mapped to actuator `max_force`.
21. GPU reset randomization: per-env seed buffer, joint angle perturbations, optional mass/friction domain randomization, opt-in flag.
22. Reward primitives library: upright, height, COM velocity tracking, energy penalty, foot clearance, alive bonus. Combine via config in the spec.
23. Termination logic: fall (root height < threshold), NaN, joint-limit excursion, time limit.
24. Observation normalization (running mean/var on GPU).
25. Train a humanoid standing PPO policy end-to-end through `HumanoidStandingVectorEnvDriver`. Save checkpoint to `.build/humanoid_standing_ppo.bin`. Smoke check loads the checkpoint, runs 200 steps, and verifies mean episode return is at least 2× a zero-action baseline collected in the same harness. This is the integration test that proves the env actually trains.

## Milestone G — Debugging surface + tooling

26. Renderer: joint-limit violation coloring (red on links/joints over limit). Constraint error heat overlay per link. Joint frames + axes color-coded by joint type.
27. Headless capture: `scripts/render_humanoid_video.sh` drives a headless browser over the replay JSON and pipes to ffmpeg for MP4 output. Fail loudly if browser or ffmpeg is missing.
28. Per-kernel profiler hook printing slowest N kernels per step.
29. Asset `schema_version` field. Loader rejects mismatch with a clear error pointing to the expected version.
30. Update README and `docs/HUMANOID_RIGID_BODY_PHYSICS_PLAN.md` to reflect final state. Move now-implemented items out of "Remaining" into "Implemented" only when there is a real test asserting the behavior.

## Final report

Produce a single report:
- Per-milestone: what landed, what was deferred, why.
- Final list of files changed.
- Last-run tail of all four validation commands.
- "Remaining limitations" list with concrete next steps.
- Any perf numbers from the profiler hook.
