# CartPole Module Overview

The project now has two separate concerns:

- reusable environment code in `src/`
- deterministic validation harness code in `metal-smoke-check/`

## Reusable Module

The reusable cartpole code lives in:

- [MetalSupport.swift](/Users/smile/pufer/Environment/src/metal_core/MetalSupport.swift)
- [VectorEnv.swift](/Users/smile/pufer/Environment/src/envs/common/VectorEnv.swift)
- [LinearPolicy.swift](/Users/smile/pufer/Environment/src/rl/policy/LinearPolicy.swift)
- [MetalLinearPolicy.swift](/Users/smile/pufer/Environment/src/rl/policy/MetalLinearPolicy.swift)
- [PolicyRollout.swift](/Users/smile/pufer/Environment/src/rl/policy/PolicyRollout.swift)
- [RandomPolicyRollout.swift](/Users/smile/pufer/Environment/src/rl/random/RandomPolicyRollout.swift)
- [RolloutStorage.swift](/Users/smile/pufer/Environment/src/rl/storage/RolloutStorage.swift)
- [CartPoleTypes.swift](/Users/smile/pufer/Environment/src/envs/cartpole/CartPoleTypes.swift)
- [CartPoleReference.swift](/Users/smile/pufer/Environment/src/envs/cartpole/CartPoleReference.swift)
- [CartPoleMetalEnvironment.swift](/Users/smile/pufer/Environment/src/envs/cartpole/CartPoleMetalEnvironment.swift)
- [CartPoleVectorEnvDriver.swift](/Users/smile/pufer/Environment/src/envs/cartpole/CartPoleVectorEnvDriver.swift)
- [cartpole_kernels.metal](/Users/smile/pufer/Environment/src/envs/cartpole/Shaders/cartpole_kernels.metal)

`CartPoleMetalEnvironment` is the current reusable host-side API. It owns:

- Metal pipelines
- persistent environment buffers
- action uploads
- step dispatch
- done-reset dispatch
- output extraction dispatch
- state readback for validation
- observation, reward, done, and reset-count readback for consumers

The current public interface is intentionally small:

- `load(initialStates:)`
- `setResetSeed(_:)`
- `step(actions:)`
- `resetDone()`
- `readObservations()`
- `readRewards()`
- `readDones()`
- `readResetCounts()`
- `readBatch()`

That means future training code can interact with:

- action bounds through `actionSpec`
- observation shape through `observationSpec`
- observation/reward/done outputs through the interface methods

without depending on raw `CartPoleState` layout.

## Vector Driver Layer

The trainer-facing boundary now sits one layer above the raw environment:

- `MetalVectorEnvDriver`
- `CartPoleVectorEnvDriver`

This layer is responsible for exposing the stable loop shape a trainer wants:

- `reset() -> VectorEnvBatch`
- `step(actions:) -> VectorEnvBatch`
- `resetDone() -> VectorEnvBatch`

`VectorEnvBatch` carries:

- flattened observations
- rewards
- dones
- reset counts

The validation harness still uses debug-only reads of raw state through the cartpole driver so we can keep the deterministic CPU/GPU checks strong, but the intended trainer dependency is now the vector-driver interface rather than `CartPoleMetalEnvironment` directly.

## Random-Policy Rollout Layer

There is now a minimal trainer-side rollout utility above the vector driver:

- `RandomPolicyConfig`
- `VectorRollout`
- `VectorRolloutStep`
- `collectRandomPolicyRollout(driver:config:)`

This layer depends only on `MetalVectorEnvDriver`.

Its purpose is to prove that:

- rollout collection can be written against the trainer-facing interface
- action generation can be deterministic under a fixed seed
- same-seed replays can be checked before any neural network is introduced

## CPU Linear-Policy Layer

There is now also a tiny hand-written policy layer above the vector driver:

- `VectorPolicy`
- `LinearPolicy`
- `MetalLinearPolicy`
- `collectPolicyRollout(driver:policy:config:)`

This layer now exists in both CPU and Metal forms, and it is important because it proves:

- a policy can consume flattened observations from `VectorEnvBatch`
- a policy can emit actions through the stable trainer-facing interface
- rollout collection works without depending on random actions
- CPU and GPU fixed-weight policies can be compared directly before MLP work starts

## Host-Side Rollout Storage

There is now also an explicit trainer-side storage layer:

- `VectorRolloutStorage`
- `makeRolloutStorage(from:driver:)`
- `collectRandomPolicyRolloutStorage(...)`
- `collectPolicyRolloutStorage(...)`

This layer is still strictly non-learning infrastructure. Its purpose is to make trajectory layout explicit before PPO-specific logic is added.

It currently stores:

- observations
- actions
- rewards
- dones
- reset counts
- next observations

## Validation Harness

The validation entrypoint stays here:

- [main.swift](/Users/smile/pufer/Environment/metal-smoke-check/Sources/main.swift)

Its job is only to:

- create the reusable environment
- create the reusable vector driver
- run the random-policy rollout collector through that driver
- run the CPU linear-policy rollout collector through that driver
- verify stored rollout contents and replay determinism
- generate deterministic initial states and actions
- compare GPU results against the CPU reference
- compare exported observations, rewards, and dones against CPU expectations
- verify seeded replay behavior

## Why This Split Matters

This keeps the environment implementation usable by future training code without mixing it with test-only logic, while preserving a deterministic harness that can catch regressions before we layer on policies or PPO.
