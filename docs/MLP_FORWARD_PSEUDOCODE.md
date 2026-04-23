# MLP Forward Pseudocode

This document describes the first actor-critic-shaped model step in this repository:

- fixed weights
- forward pass only
- no training
- CPU and GPU implementations that can be compared directly

The goal is not to build the final neural-network stack yet. The goal is to prove that a small multi-layer actor-critic forward pass can run correctly on Metal and match a CPU reference exactly enough to trust the path.

## Scope

Start with the smallest useful actor-critic MLP:

- input dimension = observation dimension
- one hidden layer
- hidden activation = ReLU
- policy head output dimension = action dimension
- value head output dimension = 1
- final policy output clamped to the action bounds

For cartpole today, that means:

- observation dim = 4
- action dim = 1
- hidden dim = small fixed number, such as 8

## Data Layout

Assume row-major flattened weights.

### Layer 1

- `inputWeights[hiddenDim * inputDim]`
- `inputBias[hiddenDim]`

Interpretation:

- neuron `h` reads weights from
  `inputWeights[h * inputDim ..< (h + 1) * inputDim]`

### Policy Head

- `outputWeights[actionDim * hiddenDim]`
- `outputBias[actionDim]`

Interpretation:

- action output `a` reads weights from
  `outputWeights[a * hiddenDim ..< (a + 1) * hiddenDim]`

### Value Head

- `valueWeights[hiddenDim]`
- `valueBias[1]`

Interpretation:

- the scalar state value reads one weight per hidden unit

## CPU Reference Pseudocode

```text
for each env e:
    obs = observations[e]

    for each hidden neuron h:
        hidden[h] = inputBias[h]
        for each input i:
            hidden[h] += inputWeights[h, i] * obs[i]
        hidden[h] = relu(hidden[h])

    for each action output a:
        action[a] = outputBias[a]
        for each hidden neuron h:
            action[a] += outputWeights[a, h] * hidden[h]
        action[a] = clamp(action[a], minAction, maxAction)

    value = valueBias
    for each hidden neuron h:
        value += valueWeights[h] * hidden[h]
```

Equivalent scalar formulas:

```text
hidden_h = relu(inputBias_h + sum_i inputWeights[h, i] * obs_i)
action_a = clamp(outputBias_a + sum_h outputWeights[a, h] * hidden_h)
value = valueBias + sum_h valueWeights[h] * hidden_h
```

## GPU Kernel Pseudocode

Use one thread per environment.

That means each thread:

- reads one env observation slice
- computes the shared hidden layer once
- writes all action outputs for that env
- writes one scalar value output for that env

Inside the thread:

```text
if gid >= envCount:
    return

envIndex = gid

load observation slice for envIndex

for each hidden neuron h:
    hidden[h] = inputBias[h]
    for each input i:
        hidden[h] += inputWeights[h, i] * obs[i]
    hidden[h] = relu(hidden[h])

for each action output a:
    action[a] = outputBias[a]
    for each hidden neuron h:
        action[a] += outputWeights[a, h] * hidden[h]
    actions[envIndex, a] = clamp(action[a], minAction, maxAction)

value = valueBias
for each hidden neuron h:
    value += valueWeights[h] * hidden[h]

values[envIndex] = value
```

## Why This Kernel Shape Is Fine For Now

This is not the final optimized network implementation.

It is acceptable for the first step because:

- cartpole is tiny
- action dimension is tiny
- hidden dimension is tiny
- correctness matters more than optimal kernel fusion

The main thing we are validating is:

- buffer binding
- weight layout
- per-env indexing
- activation math
- CPU/GPU parity

## What We Will Test

Before trusting the MLP, we should verify all of these:

1. Direct CPU vs GPU action and value parity on the same observation batch.
2. Same-seed GPU rollout replay matches exactly.
3. CPU MLP rollout and GPU MLP rollout match within tolerance.
4. Alternate MLP weights actually change actions.
5. Output actions always stay inside the action bounds.
6. Value-bearing rollout storage replays exactly with the same seed.

## Failure Modes To Expect First

The most likely early failures are:

- wrong weight flattening order
- output/action index mapping errors
- host/MSL parameter mismatch
- hidden-layer temporary storage bugs
- clamping differences between CPU and GPU
- reading the wrong observation slice for each env

If the first parity test fails, reproduce it with the exact same observation batch before changing code. Do not guess. The direct CPU-vs-GPU action/value probe is the smallest failing case and should remain the first debugging target.

## What This Does Not Do Yet

This step does not include:

- log-probability computation
- sampling from a policy distribution
- gradients
- optimizer state
- PPO losses

Those should come only after this action-and-value forward path is stable.
