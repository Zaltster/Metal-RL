# Metal-RL

Reinforcement learning from scratch on Apple Silicon using Metal compute kernels.

## What This Is

This project is building an RL stack directly on top of Metal, without MLX, PyTorch, or CUDA.

Current goals:

- learn Apple GPU programming deeply
- run many environment instances in parallel on a local Apple GPU
- build the RL stack piece by piece with explicit validation at each stage

## Current Status

The repository currently contains a validated non-learning foundation.

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
- host-side rollout storage with explicit indexing helpers

What is not implemented yet:

- learned neural network training
- policy sampling with a learned stochastic policy
- backpropagation
- optimizer updates
- end-to-end training

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
- rollout storage replay

## Run

```bash
./metal-smoke-check/scripts/build_and_run.sh
```

Expected output includes lines like:

- `CartPole validation harness passed`
- `same-seed replay matched exactly`
- `cpu and gpu linear-policy outputs matched`
- `rollout storage replay matched exactly`

## Design Principles

- explicit buffers over hidden tensor abstractions
- deterministic validation before optimization
- reproduce failures before fixing them
- keep the trainer-facing interface stable while internals evolve

## Next Steps

The next major phase is learned-model work:

1. learned policy forward pass on GPU
2. optimization and backward/update logic
3. end-to-end training
4. broader environment/model scaling

## Notes

- `PufferLib/` is included as inspiration/reference and is not part of the current Metal implementation path.
- The project currently targets Apple Silicon with Metal through Swift host code and MSL kernels.
