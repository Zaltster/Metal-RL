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
- live training-demo progress logs with elapsed time, ETA, env steps/sec, reward, loss, and update size
- post-training CartPole replay export as a self-contained HTML/Canvas animation
- explicit policy sampling mode plumbing for `deterministic-mean` and seeded `stochastic-gaussian`
- stochastic Gaussian action sampling for actor-critic rollout exploration, with CPU/GPU storage parity validation
- deterministic repeated CPU-side PPO training loop over the GPU cartpole environment with seeded minibatch shuffling and Adam
- deterministic hybrid training loop with GPU rollouts and CPU Adam updates
- JSON checkpoint save/load for the trainable MLP actor-critic, with restored CPU/GPU forward parity validation
- JSON checkpoint/restart validation for persistent GPU trainable parameters plus Adam optimizer state
- host-side rollout storage with explicit indexing helpers
- locked humanoid v1 robot spec and synthetic baseline JSON
- GPU-first humanoid elastic-joint environment with JSON loading, rigid-body state buffers, Metal reset/step/FK/output kernels, and HTML replay export

What is not implemented yet:

- learned policy log-std
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
  Start with `docs/README.md`. See `docs/HUMANOID_RIGID_BODY_PHYSICS_PLAN.md` for the GPU rigid-body/contact physics and renderer roadmap.

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
- seeded stochastic Gaussian actor-critic rollout replay and CPU/GPU parity
- humanoid GPU rigid-body state, elastic-joint step, forward kinematics, and replay export
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
- `stochastic gaussian actor-critic rollout replay and cpu/gpu parity matched`
- `humanoid gpu rigid-body state, free-body integration, joint anchor constraints, motors/limits, ground/self contacts, contact solver, standing env, FK, and replay passed`
- `rollout storage replay matched exactly`

Run the standalone CartPole trainer:

```bash
./scripts/train_cartpole_demo.sh
```

The demo prints progress during training:

```text
progress iter=1/200 elapsed=00:01 eta=03:12 envSteps=32768/6553600 stepsPerSec=345000 meanReward=...
```

The default backend is `persistent-gpu-adam`. The previous hybrid path remains available:

```bash
TRAIN_BACKEND=hybrid-cpu-adam ./scripts/train_cartpole_demo.sh
```

The rollout sampling mode is explicit and defaults to deterministic action means:

```bash
TRAIN_POLICY_SAMPLING=deterministic-mean ./scripts/train_cartpole_demo.sh
```

Stochastic Gaussian exploration is also available. It samples from the fixed diagonal Gaussian policy, clamps actions to the environment action bounds, and stores the executed action log-probability for PPO:

```bash
TRAIN_POLICY_SAMPLING=stochastic-gaussian ./scripts/train_cartpole_demo.sh
TRAIN_POLICY_SEED=0xC0FFEE11 TRAIN_POLICY_SAMPLING=stochastic-gaussian ./scripts/train_cartpole_demo.sh
```

Progress logs print every iteration by default. Change or disable that with `TRAIN_LOG_EVERY`:

```bash
TRAIN_LOG_EVERY=10 ./scripts/train_cartpole_demo.sh
TRAIN_LOG_EVERY=0 ./scripts/train_cartpole_demo.sh
```

The demo also writes a post-training replay animation by default:

```text
replayPath: /Users/smile/pufer/Environment/train-cartpole-demo/.build/cartpole_replay.html
```

Open that HTML file in a browser to see the cart and pole move. Replay controls:

```bash
TRAIN_REPLAY_HORIZON=480 ./scripts/train_cartpole_demo.sh
TRAIN_REPLAY_ENV=3 ./scripts/train_cartpole_demo.sh
TRAIN_REPLAY_PATH=/tmp/cartpole_replay.html ./scripts/train_cartpole_demo.sh
TRAIN_REPLAY_HORIZON=0 ./scripts/train_cartpole_demo.sh
```

Run the GPU humanoid elastic-joint demo:

```bash
./scripts/run_humanoid_demo.sh
```

It loads `docs/humanoid_v1_baseline.json`, runs the batched Metal humanoid environment, and writes:

```text
humanoid-demo/.build/humanoid_replay.html
```

Useful controls:

```bash
HUMANOID_ENV_COUNT=4096 ./scripts/run_humanoid_demo.sh
HUMANOID_STEPS=600 ./scripts/run_humanoid_demo.sh
HUMANOID_REPLAY_ENV=3 ./scripts/run_humanoid_demo.sh
```

## Design Principles

- explicit buffers over hidden tensor abstractions
- deterministic validation before optimization
- reproduce failures before fixing them
- keep the trainer-facing interface stable while internals evolve

## Next Steps

The next major phase is turning the persistent-GPU optimizer demo into a more complete training product:

1. add production run directories, metrics logs, best/latest checkpoints, and resume
2. add eval mode with fixed checkpoint, mean return, mean episode length, and reset counts
3. reduce CPU readback in the persistent GPU training summaries
4. run throughput scaling tests and longer learning-quality tests with CartPole solve criteria

## Notes

- `PufferLib/` is included as inspiration/reference and is not part of the current Metal implementation path.
- The project currently targets Apple Silicon with Metal through Swift host code and MSL kernels.
