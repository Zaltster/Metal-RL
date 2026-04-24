# Metal-RL

Reinforcement learning from scratch on Apple Silicon using Metal compute kernels.

## What This Is

This project is building an RL stack directly on top of Metal, without MLX, PyTorch, or CUDA.

Current goals:

- learn Apple GPU programming deeply
- run many environment instances in parallel on a local Apple GPU
- build the RL stack piece by piece with explicit validation at each stage

## Current Status

The repository currently contains a validated hybrid training foundation.

What is implemented:

- GPU cartpole environment stepping in Metal
- GPU done-aware reset kernel in Metal
- GPU export path for observations, rewards, and dones
- Swift host-side environment wrapper with explicit buffers, pipelines, and dispatch
- CPU reference implementation for parity checks
- trainer-facing vector-environment driver
- random-policy rollout collection
- fixed-weight CPU linear policy rollout collection
- fixed-weight GPU linear policy rollout collection
- CPU/GPU parity checks for the fixed linear policy
- fixed-weight CPU MLP policy rollout collection
- fixed-weight GPU MLP policy rollout collection
- CPU/GPU parity checks for the fixed MLP policy
- fixed-weight CPU/GPU MLP value prediction parity
- actor-critic-style rollout storage with value predictions
- host-side GAE computation with synthetic and rollout-backed validation
- host-side PPO loss computation with synthetic and rollout-backed validation
- hand-derived CPU backward/update step for the fixed actor-critic MLP
- first Metal PPO/MLP gradient path with per-sample gradient computation, GPU-side reduction, and CPU parity on a synthetic PPO batch
- first Metal SGD update kernel with CPU/GPU updated-weight parity on a synthetic PPO batch
- single-step Metal SGD training helper with CPU/GPU loss and weight parity on synthetic and real rollout PPO batches
- persistent GPU trainable MLP parameter buffers with repeated SGD parity against CPU updates
- persistent GPU Adam first/second-moment state with repeated Adam parity against CPU updates
- tiny persistent-GPU optimizer training loop with CPU SGD and CPU Adam loop parity
- direct Metal-buffer sync from persistent trainable parameters into GPU rollout policy inference
- standalone CartPole training demo using persistent GPU Adam by default, with hybrid CPU Adam still selectable
- deterministic repeated CPU-side PPO training loop over the GPU cartpole environment with seeded minibatch shuffling and Adam
- deterministic hybrid training loop with GPU rollouts and CPU Adam updates
- JSON checkpoint save/load for the trainable MLP actor-critic, with restored CPU/GPU forward parity validation
- JSON checkpoint/restart validation for persistent GPU trainable parameters plus Adam optimizer state
- host-side rollout storage with explicit indexing helpers

What is not implemented yet:

- policy sampling with a learned stochastic policy
- full GPU-side backpropagation beyond the current one-hidden-layer MLP PPO path
- production resume flow using persistent GPU optimizer-state checkpoints
- fully device-resident checkpointing without host serialization
- fully GPU-native end-to-end training

## Project Structure

- `src/metal_core/`
  Native Metal helpers for pipelines, buffers, and dispatch.
- `src/envs/cartpole/`
  Cartpole state types, CPU reference logic, Metal environment wrapper, and Metal kernels.
- `src/envs/common/`
  Shared vector-environment interfaces.
- `src/rl/random/`
  Random-policy rollout collection.
- `src/rl/policy/`
  Fixed-weight CPU and GPU linear/MLP policy inference, plus fixed MLP value prediction.
- `src/rl/storage/`
  Host-side rollout storage.
- `src/rl/postprocess/`
  Host-side trajectory postprocessing such as GAE.
- `src/rl/losses/`
  Host-side PPO loss computation.
- `src/rl/train/`
  CPU-side hand-derived backward/update step, first Metal gradient/update/training-step parity helpers, persistent GPU trainable parameter buffers and Adam state, direct GPU-buffer rollout policy sync, tiny persistent-GPU optimizer loop parity, repeated epoch/minibatch training loop, hybrid GPU-rollout training loop, and trainable MLP actor-critic/training-state checkpointing.
- `train-cartpole-demo/`
  Small standalone end-to-end training entrypoint. It uses persistent GPU Adam by default; set `TRAIN_BACKEND=hybrid-cpu-adam` to run the previous hybrid GPU-rollout/CPU-Adam path.
- `scripts/train_cartpole_demo.sh`
  Builds and runs the training demo executable.
- `metal-smoke-check/`
  Deterministic validation harness used to verify the stack end to end.
- `docs/`
  Architecture and module notes.

## Validation

The project is built validation-first.

The current harness checks:

- Metal device access
- struct layout assumptions
- one-step and multi-step CPU/GPU environment parity
- reset determinism
- seeded replay for environment rollouts
- seeded replay for random-policy rollouts
- seeded replay for fixed-policy rollouts
- CPU/GPU fixed-policy parity
- CPU/GPU fixed-MLP parity
- CPU/GPU fixed-value parity
- synthetic GAE correctness
- CPU/GPU GAE parity on stored actor-critic rollouts
- synthetic PPO loss correctness
- CPU/GPU PPO loss parity on stored actor-critic rollouts
- synthetic backward-step loss reduction
- CPU/GPU MLP gradient parity on a synthetic PPO batch
- CPU/GPU MLP SGD update parity on a synthetic PPO batch
- CPU/GPU single-step SGD training parity on synthetic and real rollout PPO batches
- repeated persistent GPU SGD update parity on synthetic and real rollout PPO batches
- repeated persistent GPU Adam update parity on synthetic and real rollout PPO batches
- persistent GPU Adam checkpoint/restart parity against uninterrupted GPU Adam updates
- persistent GPU SGD training-loop parity against CPU SGD
- persistent GPU Adam training-loop parity against CPU Adam
- direct persistent-trainable-buffer sync into GPU rollout policy inference
- real stored-rollout backward-step loss reduction
- deterministic CPU training-loop replay
- deterministic hybrid GPU-rollout training-loop replay
- trainable actor-critic checkpoint round-trip and restored GPU policy parity
- rollout storage replay

## Run

```bash
./metal-smoke-check/scripts/build_and_run.sh
```

Expected output includes lines like:

- `CartPole validation harness passed`
- `same-seed replay matched exactly`
- `cpu and gpu linear-policy outputs matched`
- `cpu and gpu mlp gradients matched after gpu reduction on synthetic PPO batch`
- `cpu and gpu mlp sgd updates matched on synthetic PPO batch`
- `cpu and gpu mlp sgd training steps matched on synthetic and real PPO batches`
- `persistent gpu mlp sgd updates matched repeated cpu sgd on synthetic and real PPO batches`
- `persistent gpu mlp adam updates matched repeated cpu adam on synthetic and real PPO batches`
- `persistent gpu adam checkpoint restart matched uninterrupted gpu adam updates`
- `persistent gpu sgd training loop matched cpu sgd training loop`
- `persistent gpu adam training loop matched cpu adam training loop`
- `persistent gpu trainable buffers synced directly into rollout policy buffers`
- `rollout storage replay matched exactly`

Run the standalone CartPole trainer:

```bash
./scripts/train_cartpole_demo.sh
```

The default backend is `persistent-gpu-adam`. The previous hybrid path remains available:

```bash
TRAIN_BACKEND=hybrid-cpu-adam ./scripts/train_cartpole_demo.sh
```

## Design Principles

- explicit buffers over hidden tensor abstractions
- deterministic validation before optimization
- reproduce failures before fixing them
- keep the trainer-facing interface stable while internals evolve

## Next Steps

The next major phase is turning the persistent-GPU optimizer demo into a more complete training product:

1. add production run directories, metrics logs, best/latest checkpoints, and resume
2. reduce CPU readback in the persistent GPU training summaries
3. run longer learning-quality tests and define CartPole solve criteria
4. move more rollout postprocessing and loss bookkeeping onto Metal

## Notes

- `PufferLib/` is included as inspiration/reference and is not part of the current Metal implementation path.
- The project currently targets Apple Silicon with Metal through Swift host code and MSL kernels.
