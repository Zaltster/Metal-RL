# Environment Buffer Dataflow

This document is the concrete Stage 7 buffer/dataflow plan for the current cartpole validation harness.

It describes what data lives in which buffer, who owns it, and how it moves through one rollout step before we build larger environment abstractions.

The current consumer-facing abstraction above these buffers is now:

- `MetalVectorEnvDriver`
- `CartPoleVectorEnvDriver`
- `VectorEnvBatch`
- `collectRandomPolicyRollout(...)`
- `collectPolicyRollout(...)`

## Current Scope

This plan covers the deterministic validation harness only:

- multi-step cartpole rollout
- GPU step kernel
- GPU done-reset kernel
- GPU output extraction kernel
- CPU reference validation

It does not yet include policy/value outputs, PPO storage, or gradients.

## Abstraction Boundary

The raw buffer graph remains inside `CartPoleMetalEnvironment`, but the intended trainer-facing interface is no longer the raw environment object.

The intended boundary is:

- trainer calls `reset()`
- trainer calls `step(actions:)`
- trainer reads a `VectorEnvBatch`

The current trainer-side rollout layer then does:

- rollout collector samples actions
- rollout collector calls `step(actions:)`
- rollout collector records `rewards`, `dones`, and post-reset observations
- rollout collector calls `resetDone()` between steps

For the CPU linear-policy path, the sequence is:

- policy reads flattened observations from `VectorEnvBatch`
- policy computes actions on CPU
- rollout collector calls `step(actions:)`
- rollout collector records returned signals and post-reset observations

This matters because it lets the environment implementation keep evolving internally while the trainer depends only on:

- flattened observations
- rewards
- dones
- reset counts

## Core Buffers

### `stateBuffer`

Type:

- array of `CartPoleState`

Ownership:

- persistent across the entire rollout

Written by:

- host at initialization
- `step_cartpole` kernel
- `reset_done_cartpoles` kernel

Read by:

- `step_cartpole` kernel
- `reset_done_cartpoles` kernel
- host validation code after each stage

Role:

- canonical GPU-resident environment state

### `actionBuffer`

Type:

- array of `Float`

Ownership:

- persistent allocation, contents updated every step

Written by:

- host before each rollout step

Read by:

- `step_cartpole` kernel

Role:

- current action for each environment lane

### `stepParamsBuffer`

Type:

- one `CartPoleParams`

Ownership:

- long-lived, effectively read-only during a run

Written by:

- host before rollout starts

Read by:

- `step_cartpole` kernel

Role:

- physics constants and environment count

### `observationBuffer`

Type:

- flat array of `Float`
- shape is `envCount * 4`

Ownership:

- persistent across the rollout

Written by:

- `write_cartpole_outputs` kernel

Read by:

- host validation code
- future training loop code

Role:

- exported observation surface for consumers that should not depend on `CartPoleState` layout

### `rewardBuffer`

Type:

- array of `Float`

Ownership:

- persistent across the rollout

Written by:

- `write_cartpole_outputs` kernel

Read by:

- host validation code
- future training loop code

Role:

- exported reward surface separate from raw state

### `doneBuffer`

Type:

- array of `UInt32`

Ownership:

- persistent across the rollout

Written by:

- `write_cartpole_outputs` kernel

Read by:

- host validation code
- future training loop code

Role:

- exported termination surface separate from raw state

### `resetCountBuffer`

Type:

- array of `UInt32`

Ownership:

- persistent across the rollout

Written by:

- host initializes to zero
- `reset_done_cartpoles` increments done lanes

Read by:

- `reset_done_cartpoles` kernel
- host validation code

Role:

- per-lane episode/reset counter
- deterministic input to seeded reset generation

### `resetParamsBuffer`

Type:

- one `ResetParams`

Ownership:

- long-lived, effectively read-only during a run

Written by:

- host before rollout starts

Read by:

- `reset_done_cartpoles` kernel

Role:

- reset seed and environment count

## One Rollout Step

The current deterministic validation flow for one logical environment step is:

1. Host writes the current action vector into `actionBuffer`.
2. GPU `step_cartpole` reads `stateBuffer`, `actionBuffer`, and `stepParamsBuffer`.
3. GPU `step_cartpole` writes stepped states back into `stateBuffer`.
4. GPU `write_cartpole_outputs` reads `stateBuffer` and writes `observationBuffer`, `rewardBuffer`, and `doneBuffer`.
5. Host reads `stateBuffer`, `observationBuffer`, `rewardBuffer`, and `doneBuffer` and compares them against the CPU step reference.
6. GPU `reset_done_cartpoles` reads `stateBuffer`, `resetCountBuffer`, and `resetParamsBuffer`.
7. GPU `reset_done_cartpoles` increments reset counts for done lanes only.
8. GPU `reset_done_cartpoles` writes reset states back into `stateBuffer` for done lanes only.
9. GPU `write_cartpole_outputs` runs again to refresh `observationBuffer`, `rewardBuffer`, and `doneBuffer`.
10. Host reads `stateBuffer`, `observationBuffer`, `rewardBuffer`, `doneBuffer`, and `resetCountBuffer` and compares them against the CPU reset reference.

This is intentionally validation-first, not throughput-first.

## Why The Buffers Are Split This Way

### States Are Persistent

`stateBuffer` stays alive for the whole rollout because environment stepping is inherently stateful.

That lets us model the real long-term design:

- one canonical environment state array on GPU
- kernels mutate it in place
- host only reads it for validation or debugging

### Actions Are Rewritten Every Step

`actionBuffer` is separate because actions change each step and will eventually come from the policy network.

That separation makes the future interface clear:

- environment kernel consumes actions
- policy path produces actions

### Outputs Are Materialized Separately

`observationBuffer`, `rewardBuffer`, and `doneBuffer` exist so downstream code can consume the environment through a stable interface rather than through the internal state struct.

That matters because:

- the internal state layout may evolve
- the training loop should rely on exported observations and signals
- state-specific debugging can stay separate from trainer-facing data

### Reset Counts Are Explicit

`resetCountBuffer` exists because deterministic seeded resets need a stable per-lane episode counter.

That gives us:

- reproducible resets for debugging
- a clean place to track per-lane episode progression

## Current Readback Policy

For the validation harness, the host reads back:

- `stateBuffer` after the step kernel
- `observationBuffer` after output extraction
- `rewardBuffer` after output extraction
- `doneBuffer` after output extraction
- `stateBuffer` after the reset kernel
- `observationBuffer` after reset output extraction
- `rewardBuffer` after reset output extraction
- `doneBuffer` after reset output extraction
- `resetCountBuffer` after the reset kernel

This is intentionally heavy on readbacks because correctness is the priority.

For the future training path, the target is different:

- keep `stateBuffer` GPU-resident
- minimize readbacks to debugging, logging, and periodic checks
- eventually keep rollout storage GPU-resident too

## Dataflow Rules

The rules we are following now are:

- `stateBuffer` is the source of truth for GPU environment state
- `actionBuffer` is overwritten by the host each step
- `observationBuffer`, `rewardBuffer`, and `doneBuffer` are derived views of state
- the rollout collector depends only on vector-env outputs, not raw state layout
- the CPU linear policy depends only on flattened observations, not on raw state structs
- step and reset are separate phases with separate validation points
- reset only touches lanes whose `done` flag is set
- seeded reset generation depends only on `baseSeed`, lane id, and reset count

These rules are chosen so failures are easy to isolate.

## What Changes Later

When we move beyond the harness, the next likely buffer additions are:

- rollout storage buffers for PPO
- model parameter and activation buffers
- action logits or action-distribution buffers
- value prediction buffers

But the current buffer plan is enough to support:

- deterministic multi-step validation
- reset semantics validation
- seeded reproducibility checks
- a trainer-facing observation/reward/done interface

## Immediate Constraint

Until we have stronger confidence in the environment loop, we should not hide this flow behind abstractions that make buffer ownership less obvious.

At this stage, explicit is better than elegant.
