# ProjectG1 Reference Map

`ProjectG1` is useful as a systems reference, but it is not a direct implementation template for this repository.

The short version:

- borrow its separation of responsibilities
- redesign its trainer/runtime assumptions
- ignore its framework-specific machinery

## What ProjectG1 Is

The sibling project at `/Users/smile/pufer/ProjectG1` is a Python MuJoCo PPO codebase built around:

- Gym-style environments
- Stable-Baselines3 vector-environment utilities
- PyTorch policies and PPO updates
- CPU or subprocess environment parallelism

Relevant files:

- [env.py](/Users/smile/pufer/ProjectG1/env.py)
- [env_navigate.py](/Users/smile/pufer/ProjectG1/env_navigate.py)
- [main.py](/Users/smile/pufer/ProjectG1/main.py)
- [multipleEnvs/main.py](/Users/smile/pufer/ProjectG1/multipleEnvs/main.py)
- [lstm/main.py](/Users/smile/pufer/ProjectG1/lstm/main.py)

This repository is building something fundamentally different:

- Swift host code instead of Python
- Metal kernels instead of MuJoCo stepping on CPU
- explicit GPU buffers instead of hidden tensor/env abstractions
- GPU-first vectorization instead of `DummyVecEnv` or `SubprocVecEnv`

## Borrow Directly

These ideas transfer well from `ProjectG1` and are worth copying conceptually.

### 1. Keep Environment Logic Separate From Training Logic

`ProjectG1` keeps environment code and training code in different files:

- env wrappers in `env.py` and `env_navigate.py`
- training loops in `main.py`, `multipleEnvs/main.py`, and `lstm/main.py`

That separation is correct and we should preserve it here.

Current equivalent in this repo:

- environment implementation in `src/envs/cartpole/`
- trainer-facing abstraction in `src/envs/common/`
- rollout logic in `src/rl/random/`, `src/rl/policy/`, and `src/rl/storage/`

### 2. Keep a Stable Trainer-Facing Environment Boundary

`ProjectG1` trains against a simple env loop shape:

- `reset`
- `step(action)`
- consume `obs`, `reward`, `done`, and `info`

That is still the right external contract, even though the internals are completely different.

Current equivalent in this repo:

- `MetalVectorEnvDriver`
- `VectorEnvBatch`
- `CartPoleVectorEnvDriver`

The trainer should depend on that stable boundary, not on raw Metal buffers.

### 3. Keep Logging, Evaluation, and Training Distinct

`ProjectG1` keeps side concerns from bleeding into the core PPO math:

- training loop in `main.py`
- logging in `utils.py` or `VectorG1Logger`
- rendering/evaluation in separate functions and scripts

That is the right design instinct.

What to carry forward here:

- trainer code should not own rendering concerns
- environment code should not own experiment logging
- validation harnesses should stay separate from production trainer code

### 4. Keep Rollout Storage Explicit

`ProjectG1/main.py` and `multipleEnvs/main.py` collect batches in explicit arrays before PPO updates.

That idea transfers directly.

Current equivalent in this repo:

- [RolloutStorage.swift](/Users/smile/pufer/Environment/src/rl/storage/RolloutStorage.swift)

The exact storage format will evolve, but explicit trajectory ownership is the correct pattern.

### 5. Keep Configurable Environment Selection and Policy Layers

`ProjectG1` supports swapping env variants like `MyG1Env` vs `NavigateG1Env`.

That is conceptually useful.

What to borrow:

- keep environment-specific code behind a small driver boundary
- let the trainer depend on interfaces rather than one concrete env type
- allow future environments to plug into the same rollout pipeline

## Redesign Completely

These are the areas where `ProjectG1` is informative, but the implementation model must change.

### 1. Vectorization Strategy

`ProjectG1` uses:

- `DummyVecEnv`
- `SubprocVecEnv`
- one Python env object per lane

That model does not fit this repository.

Here, vectorization should mean:

- one GPU kernel launch stepping many lanes
- one buffer holding many environment states
- one action buffer for the whole batch
- minimal CPU participation in per-lane work

So we should borrow the idea of batched rollouts, but not the process-based implementation.

### 2. Observation Normalization

`ProjectG1` relies on `VecNormalize`, which hides running statistics and couples training to wrapper state.

That is not a good default for this repository.

For this project, normalization should be treated as explicit state if we add it at all:

- clear buffer ownership
- deterministic update rules
- explicit save/load semantics
- validation against known reference behavior

Do not inherit `VecNormalize` as a design pattern.

### 3. Policy and Value Network Implementation

`ProjectG1` defines policies in PyTorch modules:

- `ActorCritic`
- `nn.Linear`
- `th.distributions.Normal`

That is only useful as math reference.

For this repository, the real implementation needs to be redesigned around:

- Metal buffers for weights
- Metal kernels for forward pass
- explicit action-parameter output layout
- eventually explicit gradients or minimal autodiff

Borrow the model boundary, not the framework implementation.

### 4. PPO Update Path

`ProjectG1` has a standard PyTorch PPO training loop:

- advantage calculation on CPU tensors
- minibatch shuffling in Python
- autograd-based optimization

We should reuse the PPO algorithmic structure, but not the runtime structure.

The equivalent future design here should make these choices explicit:

- what stays on CPU versus GPU
- how minibatches are indexed
- how gradient buffers are stored
- how optimizer state is represented

### 5. Environment State Representation

In `ProjectG1`, environment state mostly lives behind Python objects and MuJoCo internals.

In this repository, environment state must be a deliberate data layout:

- structs shared between Swift and MSL
- arrays of state in GPU-visible buffers
- explicit reset, reward, and done paths

So the conceptual role is similar, but the implementation discipline is much stricter here.

## Mostly Ignore

These parts of `ProjectG1` are either project-specific or actively misleading for this repository.

### 1. Stable-Baselines3 Wrapper Stack

Ignore:

- `make_vec_env`
- `DummyVecEnv`
- `SubprocVecEnv`
- `VecNormalize`
- `VecVideoRecorder`

Those are convenience layers for the Python stack, not building blocks for a native Metal runtime.

### 2. MuJoCo Rendering and Viewer Plumbing

Ignore most of the rendering-specific logic in:

- `env.py`
- `record_dual_camera.py`
- video helper paths in `main.py`

That code is useful for the robot project, but it does not help with the core Metal RL systems work.

### 3. Project-Specific Reward and Navigation Details

Ignore the details that are specific to the G1 robot and waypoint tasks unless we later choose to port a similar task:

- `NavigateG1Env`
- waypoint controllers
- target spawning heuristics
- per-task logging metrics

Those are domain logic, not infrastructure patterns.

### 4. Recurrent-Policy Scaffolding

Ignore the `lstm/` branch for now.

Recurrent policies add:

- hidden-state storage
- sequence masking
- recurrent batching concerns

That is not the next problem to solve in this repository.

## File-To-File Mental Mapping

This is the simplest translation table.

- `ProjectG1/env.py`
  Equivalent idea here: `src/envs/cartpole/CartPoleMetalEnvironment.swift`

- `ProjectG1/env_navigate.py`
  Equivalent idea here: a future second environment driver layered on the same `MetalVectorEnvDriver` interface

- `ProjectG1/main.py`
  Equivalent idea here: future single-policy training loop above `VectorEnv.swift`, `RolloutStorage.swift`, and policy/value code

- `ProjectG1/multipleEnvs/main.py`
  Equivalent idea here: the current default trainer architecture, except vectorization must happen inside Metal kernels rather than Python env wrappers

- `ProjectG1/lstm/main.py`
  Equivalent idea here: a much later extension, after feedforward PPO works

## What ProjectG1 Should Influence Next

`ProjectG1` should influence structure, not implementation.

The useful next things to carry forward conceptually are:

- keep the trainer loop separate from env internals
- keep rollout collection and rollout storage explicit
- keep logging/evaluation outside core training math
- keep environment variants behind a stable driver interface

## What ProjectG1 Should Not Pressure Us To Do

`ProjectG1` should not push this repository toward:

- Python-style env wrappers around per-lane state
- framework-managed normalization state
- hidden training state inside helper wrappers
- CPU subprocess vectorization
- premature recurrent-policy complexity

## Bottom Line

Treat `ProjectG1` as an architectural reference for how to partition an RL codebase.

Do not treat it as an implementation template for:

- vectorization
- memory layout
- policy execution
- PPO updates

Those parts must be designed natively for Metal.
