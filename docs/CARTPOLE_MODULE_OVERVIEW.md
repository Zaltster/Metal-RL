# CartPole Module Overview

The project now has two separate concerns:

- reusable environment code in `src/`
- deterministic validation harness code in `metal-smoke-check/`

## Reusable Module

The reusable cartpole code lives in:

- [MetalSupport.swift](/Users/smile/pufer/Environment/src/metal_core/MetalSupport.swift)
- [VectorEnv.swift](/Users/smile/pufer/Environment/src/envs/common/VectorEnv.swift)
- [LinearPolicy.swift](/Users/smile/pufer/Environment/src/rl/policy/LinearPolicy.swift)
- [MLPPolicy.swift](/Users/smile/pufer/Environment/src/rl/policy/MLPPolicy.swift)
- [MetalLinearPolicy.swift](/Users/smile/pufer/Environment/src/rl/policy/MetalLinearPolicy.swift)
- [MetalMLPPolicy.swift](/Users/smile/pufer/Environment/src/rl/policy/MetalMLPPolicy.swift)
- [PolicyRollout.swift](/Users/smile/pufer/Environment/src/rl/policy/PolicyRollout.swift)
- [RandomPolicyRollout.swift](/Users/smile/pufer/Environment/src/rl/random/RandomPolicyRollout.swift)
- [RolloutStorage.swift](/Users/smile/pufer/Environment/src/rl/storage/RolloutStorage.swift)
- [GAE.swift](/Users/smile/pufer/Environment/src/rl/postprocess/GAE.swift)
- [PPOLoss.swift](/Users/smile/pufer/Environment/src/rl/losses/PPOLoss.swift)
- [CPUActorCriticUpdate.swift](/Users/smile/pufer/Environment/src/rl/train/CPUActorCriticUpdate.swift)
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
- `MLPPolicy`
- `MetalLinearPolicy`
- `MetalMLPPolicy`
- `collectPolicyRollout(driver:policy:config:)`

This layer now exists in both CPU and Metal forms for both a linear policy and a tiny one-hidden-layer MLP, and it is important because it proves:

- a policy can consume flattened observations from `VectorEnvBatch`
- a policy can emit actions through the stable trainer-facing interface
- rollout collection works without depending on random actions
- CPU and GPU fixed-weight policies can be compared directly before learned-model work starts

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
- optional value predictions
- optional final bootstrap values

## Host-Side Postprocessing

There is now also a first trajectory postprocessing module:

- `GAEConfig`
- `AdvantageEstimates`
- `computeGAE(storage:config:)`

This remains host-side on purpose for now. Its job is to turn stored rollout data into:

- advantages
- returns

before any PPO loss or update code is introduced.

## Host-Side PPO Math

There is now also a first PPO-loss module:

- `PPOConfig`
- `PPOLossBreakdown`
- `computePPOLoss(...)`

This is still host-side and validation-first. Its current job is to prove that:

- stored old log-probabilities are shaped correctly
- current-policy re-evaluation on stored observations/actions is coherent
- clipped PPO objective math is correct
- CPU and GPU fixed policies agree on the resulting loss values

## CPU Backward Step

There is now also a first manual training-step module:

- `PPOBatch`
- `TrainableMLPActorCritic`
- `applySGDStep(...)`
- `makePPOBatch(...)`

This is intentionally narrow. It exists to prove that:

- the current actor-critic forward path is differentiable in practice
- hand-derived gradients are coherent
- a simple SGD step changes parameters
- a simple Adam step changes parameters
- the loss can be reduced on both a synthetic PPO batch and a real stored rollout batch

## Metal Gradient Parity

There is now a first narrow Metal gradient module:

- `MetalMLPGradientComputer`
- `MetalTrainableMLPActorCritic`
- `runMetalSGDTrainingStep(...)`
- `runPersistentMetalTrainingLoop(...)`
- `PolicySamplingMode`
- `mlp_ppo_per_sample_gradients`
- `mlp_reduce_per_sample_gradients`
- `mlp_sgd_update`
- `mlp_adam_update`

This now backs the default standalone CartPole demo path. It computes per-sample PPO gradients for the current one-hidden-layer actor-critic MLP, reduces those gradients with a second Metal kernel, applies Metal SGD or Adam updates, and can keep trainable parameter and Adam state buffers resident across repeated optimizer steps. A tiny persistent-GPU optimizer loop is validated against the matching CPU SGD and Adam loops. Rollout policy sync now copies directly between Metal buffers; host readback is still used for parity validation and final model summaries.

Its current job is to prove that:

- the hand-derived PPO/MLP gradient math can be reproduced in Metal
- policy and value gradients both match the CPU path after GPU-side reduction on a synthetic PPO batch
- a simple Metal SGD update produces the same updated weights as the CPU SGD path
- persistent Metal Adam state produces the same updated weights as the CPU Adam path
- a single Metal SGD training step matches CPU loss and updated weights on both synthetic and real rollout PPO batches
- repeated persistent-buffer Metal SGD updates match repeated CPU SGD updates on both synthetic and real rollout PPO batches
- repeated persistent-buffer Metal Adam updates match repeated CPU Adam updates on both synthetic and real rollout PPO batches
- tiny persistent Metal SGD and Adam loops match the corresponding CPU training loops
- persistent trainable buffers can be copied directly into rollout policy buffers without reconstructing the model on the host
- persistent trainable buffers and Adam state can be checkpointed and restored without changing the next Adam update
- the standalone demo can use persistent GPU Adam while keeping the hybrid CPU-Adam path selectable
- actor-critic rollouts explicitly declare `deterministic-mean` or seeded `stochastic-gaussian` sampling
- seeded stochastic Gaussian rollout replay and CPU/GPU rollout-storage parity are validated

## CPU Training Loop

There is now also a first repeated training-loop module:

- `CPUTrainingLoopConfig`
- `CPUTrainingIterationSummary`
- `CPUTrainingRunSummary`
- `runCPUTrainingLoop(...)`

This remains intentionally narrow. Its current job is to prove that:

- rollout collection, GAE, PPO loss, and SGD updates can be chained repeatedly
- rollout collection, GAE, PPO loss, and Adam updates can be chained repeatedly
- the training loop is deterministic under a fixed reset-seed schedule
- deterministic minibatch shuffling can be introduced without breaking replay
- a changed seed produces a different training trace
- repeated updates move parameters without breaking the validated environment path

## Hybrid GPU Rollout Loop

There is now also a first hybrid training path:

- `runHybridTrainingLoop(...)`
- `MetalMLPPolicy.load(model:)`
- `MLPActorCriticCheckpoint`
- `MLPActorCriticTrainingStateCheckpoint`
- `saveCheckpoint(...)`
- `saveTrainingStateCheckpoint(...)`
- `loadMLPActorCriticCheckpoint(...)`
- `loadMLPActorCriticTrainingStateCheckpoint(...)`
- standalone demo entrypoint in `train-cartpole-demo/`
- `TRAIN_BACKEND=persistent-gpu-adam`
- `TRAIN_BACKEND=hybrid-cpu-adam`
- `TRAIN_POLICY_SAMPLING=deterministic-mean`
- `TRAIN_POLICY_SAMPLING=stochastic-gaussian`
- `TRAIN_POLICY_SEED`
- `TRAIN_LOG_EVERY`
- `TRAIN_REPLAY_HORIZON`
- `TRAIN_REPLAY_ENV`
- `TRAIN_REPLAY_PATH`

The hybrid path keeps:

- environment stepping on the GPU
- policy/value rollout inference on the GPU
- PPO/GAE/update math on the CPU
- trainable model checkpointing on the host side
- persistent Metal trainable model and Adam-state checkpointing for validation/restart tests

The default standalone demo now uses the persistent GPU-Adam path instead:

- environment stepping on the GPU
- policy/value rollout inference on the GPU
- PPO gradients, gradient reduction, and Adam updates on the GPU
- GAE and summary/loss readback still on the host
- deterministic mean or seeded stochastic Gaussian rollout actions; stochastic actions are clamped to environment bounds before stepping
- live progress logging from host-side iteration summaries, including elapsed time, ETA, env steps/sec, reward, loss, and parameter delta
- post-training replay export to a self-contained HTML/Canvas animation generated from saved CartPole observations

Its current job is to prove that:

- trainable CPU weights can be synchronized into the Metal policy
- rollout collection can stay GPU-first during training
- the hybrid loop is still deterministic under fixed seeds
- a restored checkpoint can be loaded back into the Metal policy and preserve CPU/GPU forward parity
- a restored persistent Metal training-state checkpoint preserves the next Adam optimizer update

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
