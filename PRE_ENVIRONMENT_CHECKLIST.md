# Pre-Environment Checklist

This document defines what needs to be true before we start building actual RL environments, neural networks, or PPO logic.

The goal is to remove ambiguity from the stack-up order. If these prerequisites are not solid, later RL bugs will be hard to localize.

## Objective

Before environment and ML work, we need a minimal but trustworthy Metal compute foundation:

- host-side Metal setup works reliably
- shader compilation works reliably
- CPU and GPU agree on memory layout
- command submission and synchronization are understood
- simple kernels can be validated against CPU reference code
- debugging workflow is in place before complexity increases

This is the phase where we prove the system can execute correct GPU work, not just plausible GPU work.

## Global Test And Debug Rule

For every stage below, we will follow the same rule:

- first reproduce the error consistently
- only after reproduction is reliable do we attempt a fix
- after the fix, rerun the exact failing case to confirm the cause was understood

This matters because "fixing" an error without a reliable reproduction usually means we do not actually know why it happened. That is especially dangerous in GPU work, where layout bugs, synchronization bugs, and math bugs can all look similar from the outside.

Our debugging standard is:

- create the smallest failing case
- record the expected behavior
- record the actual behavior
- make one targeted change
- rerun the same test
- only move on once the failure is both fixed and explained

## What Comes Later

These are explicitly out of scope for this phase:

- full environment library design
- learned rollout storage for PPO
- learned policy/value networks
- autodiff or manual backprop
- PPO loss and optimization
- performance tuning for large-scale training

Those come after the foundational checks below are passing.

Clarification:

- fixed-weight policy inference used only for interface and parity testing is allowed in this phase
- learned models, gradient flow, and PPO optimization are still out of scope

## Stage 0: Machine And Toolchain Sanity

We first need to know the machine can support the workflow.

Requirements:

- Apple Silicon Mac with Metal-capable GPU
- Swift compiler available
- Metal framework available
- a repeatable command to build and run a tiny host program

What we need to verify:

- the host can create `MTLCreateSystemDefaultDevice()`
- the host can report the real GPU name
- the host can compile and run a minimal Metal program from the terminal

Why this matters:

- if the host cannot reliably create a Metal device, everything after this is blocked
- if the build path is inconsistent, later failures will be confounded with environment problems

Exit criterion:

- one minimal program prints the Metal device name successfully

How we will test this stage:

- run a tiny host-only Metal program from the terminal
- print the device name and basic device properties
- rerun the same command at least twice to confirm the setup is stable and not a one-off success
- if it fails, capture the exact command, exact stderr, and whether the failure is reproducible before changing anything

## Stage 1: Minimal Compute Pipeline

This is the first real GPU check.

Requirements:

- create a `MTLDevice`
- create a `MTLCommandQueue`
- compile an MSL compute kernel
- allocate a `MTLBuffer`
- dispatch `N` threads
- wait for completion
- read results back on the CPU

Recommended first kernel:

- `output[i] = i`

Why this matters:

- it tests the entire end-to-end compute path with almost no math
- if this fails, the issue is in dispatch, compilation, binding, or synchronization, not RL logic

Exit criterion:

- GPU writes a simple known pattern
- CPU reads back exactly the expected values

How we will test this stage:

- allocate a small buffer first, such as `N = 16`
- run a kernel that writes `output[i] = i`
- read back every element and compare exactly, not approximately
- print a short sample like the first 8 values plus the first mismatching index if any
- rerun with a larger `N`, such as `256` or `4096`, after the small case passes
- if it fails, rerun the exact same input and dispatch settings before editing code so we know whether the bug is deterministic

## Stage 2: Shared Struct Layout

After scalar buffers work, we need proof that typed state buffers are laid out identically on host and GPU.

Requirements:

- define a small struct in Swift
- define the matching struct in MSL
- verify `size`, `stride`, and `alignment`
- write struct values on GPU
- read them back on CPU
- validate every field

Why this matters:

- most early GPU bugs in custom systems are layout and binding bugs
- RL environments are state-struct-heavy, so this is a hard prerequisite

What to verify:

- mixed scalar fields
- vector fields where relevant
- predictable array-of-struct behavior

Exit criterion:

- host and GPU agree exactly on struct layout and field values

How we will test this stage:

- print `size`, `stride`, and `alignment` on the Swift side
- use a GPU kernel to populate every field of the struct with lane-dependent values
- read back all structs on the CPU and validate every field exactly
- choose values that make layout bugs obvious, such as alternating integers and distinct float/vector patterns
- if there is a mismatch, keep the same struct definition and same lane inputs while reproducing the failure before changing the layout

## Stage 3: One-Step Environment Dynamics With CPU Reference

Only after the previous stages pass should we step a real environment.

Recommended first environment:

- cartpole

Why cartpole:

- small state
- small action space
- known equations
- easy to cross-check on CPU

Requirements:

- one state struct per lane
- one action per lane
- one GPU kernel that advances all lanes one step
- CPU reference implementation of the same equations
- validation of every output field against the CPU reference

What to verify:

- state updates
- reward calculation
- done flag calculation

Exit criterion:

- GPU one-step dynamics match CPU reference within a small float tolerance

How we will test this stage:

- use deterministic initial states and deterministic actions
- compute one step on CPU and GPU from the exact same inputs
- compare every lane and every output field
- use a fixed float tolerance and report the largest observed difference
- print a few representative lanes, especially one near the center and one near a threshold
- if a mismatch appears, rerun the same states and actions before changing the kernel so the mismatch is reproducible and attributable

## Stage 4: Multi-Step Rollout Validation

One step is not enough. Many bugs only appear after repeated stepping.

Requirements:

- run several consecutive GPU steps
- keep actions deterministic
- compare full trajectories against CPU reference

What to verify:

- drift does not accumulate unexpectedly
- `done` handling remains correct over time
- terminated lanes do not continue evolving incorrectly

Exit criterion:

- short GPU rollouts match CPU reference over multiple steps

How we will test this stage:

- run a short deterministic rollout, such as 8 to 32 steps
- store CPU and GPU trajectories for the same small set of lanes
- compare state, reward, and done at every time step
- report the first step and first field where divergence appears
- if divergence shows up, rerun the same rollout with the same initial states and actions before trying to fix it

## Stage 5: Reset Semantics

Before training, we need a trustworthy reset path.

Requirements:

- detect terminated environments
- reset only the terminated lanes
- preserve active lanes
- produce deterministic resets for testing

Why this matters:

- asynchronous episode termination is a core source of bugs in vectorized RL systems
- reset logic interacts with reward, done flags, and rollout storage

What to verify:

- only terminated lanes reset
- reset values are valid
- post-reset states do not carry stale data

Exit criterion:

- multi-step tests with resets behave exactly as expected

How we will test this stage:

- create deterministic states that force some lanes to terminate while others stay active
- run one or more steps, then apply reset logic
- verify that only done lanes reset
- verify that non-done lanes preserve their state exactly
- verify that reset lanes receive known reset values or known seeded random values
- if reset behavior is wrong, preserve the same initial lanes and termination pattern until the bug is reproduced consistently

## Stage 6: Random Number Strategy

Even before learning, we need a deliberate RNG plan.

Questions to settle:

- do we use stateless per-lane RNG or persistent RNG state?
- do we generate random numbers on GPU, CPU, or both?
- how do we make rollouts reproducible for debugging?

Recommendation for early phases:

- deterministic initialization and deterministic action patterns first
- introduce RNG only after deterministic checks are solid

Exit criterion:

- reproducible test runs with controlled seeds

How we will test this stage:

- run the same seeded test twice and confirm byte-for-byte identical outputs where expected
- run with a different seed and confirm outputs change in the intended places
- verify one lane at a time first before scaling to many lanes
- if reproducibility fails, rerun the same seed and test harness before changing RNG code so we know whether the bug is in seeding, state update, or dispatch order

## Stage 7: Buffer Ownership And Dataflow Plan

Before the codebase grows, we need a clear model of what data lives where.

Core buffers we will eventually need:

- states
- actions
- observations
- rewards
- done flags
- rollout storage
- model weights
- activations
- gradients

What to decide early:

- which buffers are long-lived
- which buffers are scratch/intermediate
- what gets read back to CPU and when
- what must remain GPU-resident across rollout and training phases

Why this matters:

- unnecessary readbacks will kill throughput
- unclear ownership leads to stale-data bugs

Exit criterion:

- written dataflow plan for state/action/result buffers in the environment loop

How we will test this stage:

- write down buffer ownership for one minimal rollout path
- trace one environment lane through the buffers step by step
- identify which buffers are read-only, write-only, or read-write in each phase
- confirm that the documented buffer flow matches what the code actually binds
- if code and documentation differ, reproduce the active code path first and then update whichever side is wrong

## Stage 8: Debugging Discipline

We need a debugging method before we scale anything up.

Rules:

- start with tiny batch sizes like `N = 4`
- use deterministic initial states
- use deterministic actions
- compare against CPU reference often
- print a few specific lanes, not the whole buffer
- only scale up after small cases pass

What will likely break first:

- host and MSL struct mismatch
- wrong buffer index binding
- wrong thread-grid assumptions
- silent out-of-bounds access
- reward/done logic errors
- reset logic errors

Exit criterion:

- every new kernel has a small deterministic validation mode

How we will test this stage:

- every new kernel gets a minimal validation harness before it is used in the main path
- each harness includes a tiny deterministic case and a larger sanity case
- each harness reports the first mismatch instead of dumping the full buffer
- each harness is rerunnable from one command
- if a kernel breaks later, first reproduce it inside the smallest existing harness before attempting a broader system-level fix

## Stage 9: Minimal Project Structure

Before environments and ML code spread out, we need a simple folder layout.

Suggested layout:

- `metal-smoke-check/`
  - lowest-level validation programs
- `src/metal_core/`
  - device setup
  - pipeline creation
  - buffer helpers
- `src/envs/cartpole/`
  - state structs
  - kernels
  - CPU reference implementation
- `src/tests/`
  - deterministic validation harnesses
- `docs/`
  - architecture notes
  - memory layout notes
  - kernel contracts

The principle is simple:

- smoke checks stay separate from real system code
- CPU reference code lives alongside the GPU implementation it validates

## Stage 10: Success Criteria Before Environment Expansion

Before building more environments or touching policy networks, the following should all be true:

- Metal device creation works reliably
- shader compilation works reliably
- scalar buffer smoke test passes
- shared struct layout test passes
- one-step cartpole GPU step matches CPU reference
- short multi-step cartpole rollout matches CPU reference
- reset logic is correct and deterministic
- small-scale debugging workflow is comfortable and repeatable

If any item here is still shaky, we should not move on yet.

How we will test this stage:

- rerun the full prerequisite checklist from top to bottom after any major refactor
- confirm that each stage still has a passing validation harness
- if a previously passing stage regresses, reproduce that exact regression in isolation before fixing it

## Stage 11: Exported Environment Interface

Once raw environment stepping is correct, we need a trainer-facing interface that does not depend on raw state layout.

Requirements:

- export observations separately from internal state
- export rewards separately from internal state
- export done flags separately from internal state
- keep reset counts available for deterministic debugging

Why this matters:

- trainer code should not know about `CartPoleState` memory layout
- internal environment layout should be free to evolve without breaking the trainer path

Exit criterion:

- consumers can read observations, rewards, dones, and reset counts without touching raw state structs

How we will test this stage:

- compare exported observations against CPU-derived observations from the reference state
- compare exported rewards and dones against CPU reference values after both step and reset phases
- rerun the same deterministic rollout after any output-interface refactor and confirm no regression

## Stage 12: Stable Vector-Environment Driver

After the exported interface exists, we need a stable trainer-facing loop shape.

Requirements:

- a `reset()` entrypoint
- a `step(actions:)` entrypoint
- a `resetDone()` entrypoint
- a single batch type carrying observations, rewards, dones, and reset counts

Why this matters:

- the trainer should depend on one environment-driver interface, not on cartpole-specific internals
- later environments should be able to conform to the same loop shape

Exit criterion:

- the validation harness can run through the vector-driver interface instead of the raw cartpole environment object

How we will test this stage:

- reroute the existing deterministic validation harness through the driver abstraction
- confirm identical rollout results before and after the abstraction boundary is introduced
- if the refactor breaks behavior, reproduce it through the driver path before changing lower-level code

## Stage 13: Trainer-Side Rollout Collection Without Learned Policies

Before any learned model exists, we need proof that rollout collection works against the trainer-facing environment interface.

Requirements:

- collect rollouts using only the vector-driver interface
- support at least one random-policy collector
- support at least one fixed deterministic policy collector
- keep rollout data deterministic under fixed seeds or fixed weights

Why this matters:

- rollout collection logic is a trainer responsibility and should be validated separately from learning
- a lot of trainer bugs have nothing to do with PPO or backprop

Exit criterion:

- rollout collectors can reset, step, reset-done, and record batches without using environment internals

How we will test this stage:

- run same-seed random-policy rollouts twice and require exact replay
- run different-seed random-policy rollouts and require action differences
- run fixed-policy rollouts twice and require exact replay
- validate action bounds on every collected rollout

## Stage 14: Fixed-Weight CPU/GPU Policy Parity

Before moving to learned networks, we need one minimal policy inference path on GPU that matches a CPU reference.

Requirements:

- implement a tiny fixed-weight policy on CPU
- implement the same fixed-weight policy on GPU
- use the same observation interface and action bounds for both
- compare both direct inference and full rollout behavior

Why this matters:

- it proves the policy side of the stack can move onto Metal without immediately introducing MLP complexity
- it isolates policy-kernel issues from optimizer and loss issues

Exit criterion:

- CPU and GPU fixed-weight policy outputs match within tolerance
- CPU and GPU fixed-weight rollouts match within tolerance under identical seeds and initial conditions

How we will test this stage:

- compare one direct CPU-vs-GPU policy forward pass on the same observation batch
- compare full rollouts collected with CPU and GPU versions of the same fixed policy
- rerun identical seeds after any policy-kernel change to confirm parity is preserved

## Stage 15: Host-Side Rollout Storage

Before PPO-specific math exists, we need one explicit trajectory storage layer on the trainer side.

Requirements:

- store `T x N` rollout data with explicit shapes
- store observations, actions, rewards, dones, and reset counts
- store next observations or post-reset observations for each step
- provide indexing helpers so trainer code does not hand-roll flat indexing everywhere

Why this matters:

- PPO and GAE both depend on correct trajectory indexing
- rollout storage bugs are trainer bugs, not optimizer bugs, and should be isolated early
- explicit storage lets us verify determinism at the data-structure level, not just at the rollout API level

Exit criterion:

- rollout storage can be built from the trainer-facing rollout path
- same-seed runs produce identical stored contents
- storage indexing helpers return the expected per-step, per-env data

How we will test this stage:

- build storage from at least one random-policy rollout and one fixed-policy rollout
- compare stored contents against the original rollout batches field by field
- rerun the same seeded rollout and require identical stored arrays
- validate a few direct indexing calls like `observation(step:env:)` and `action(step:env:)` against known rollout slices

## Stage 16: Non-ML Completion Gate

This is the point where the non-learning infrastructure should be considered complete enough to start learned-model work.

Before we move on, all of the following should be true:

- environment stepping is validated against CPU reference
- reset behavior is deterministic and correct
- exported observations, rewards, dones, and reset counts are validated
- the trainer-facing vector-driver interface is stable
- trainer-side rollout collection works without environment internals
- fixed-weight CPU and GPU policy inference match
- fixed-weight CPU and GPU rollouts match
- host-side rollout storage is deterministic and correctly indexed
- deterministic replay checks are routine and reliable

If any of those are still shaky, we are not ready for learned models.

How we will test this stage:

- rerun the full harness after every major refactor
- confirm environment, rollout, and fixed-policy parity checks still pass
- if a regression appears, reproduce it in the smallest layer that shows the mismatch before fixing anything

## Recommended Build Order

This is the exact order we should follow:

1. host-only Metal device check
2. scalar buffer compute smoke test
3. shared struct layout test
4. one-step cartpole GPU step plus CPU reference
5. multi-step rollout validation
6. reset handling
7. deterministic RNG plan
8. exported environment interface
9. vector-environment driver abstraction
10. trainer-side rollout collection with random and fixed policies
11. fixed-weight CPU/GPU policy parity
12. host-side rollout storage
13. only after that add learned neural network forward pass
14. only after that add backward pass and PPO

## Why This Order Is Important

The temptation is to jump into environments or learning quickly. That usually creates a debugging pile where:

- physics bugs
- layout bugs
- dispatch bugs
- reward bugs
- reset bugs
- optimizer bugs

all appear at once.

This checklist prevents that. Each stage isolates one class of failure.

## Next Immediate Steps

The next steps after this document are:

1. keep the current validation harness runnable as layers are added
2. preserve CPU reference parity while policy inference moves from CPU to Metal
3. treat learned-model work as a new phase after the non-ML completion gate

Only after those pass should we start learned-model and PPO work.
