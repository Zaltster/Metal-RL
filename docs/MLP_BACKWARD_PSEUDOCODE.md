# MLP Backward Pseudocode

This document describes the first training step for the repository:

- CPU only
- one-hidden-layer actor-critic MLP
- hand-derived gradients
- simple SGD
- no custom autodiff engine yet

The goal is to prove that the current PPO math can actually drive parameter updates before moving gradients or optimizer state to Metal.

## Model

For one sample:

```text
z = W1 * obs + b1
h = relu(z)

mean = Wpi * h + bpi
value = Wv * h + bv
```

The policy uses a fixed diagonal Gaussian log-std, so only the mean is trainable in this step.

## PPO Terms Used

For one sample:

```text
ratio = exp(newLogProb - oldLogProb)
clippedRatio = clamp(ratio, 1 - eps, 1 + eps)

surrogate1 = ratio * advantage
surrogate2 = clippedRatio * advantage
policyTerm = min(surrogate1, surrogate2)

policyLoss = -policyTerm
valueLoss = 0.5 * valueCoef * (value - return)^2
totalLoss = policyLoss + valueLoss
```

Entropy has no gradient here because log-std is fixed.

## Gradient With Respect To New Log-Prob

Only the unclipped branch contributes gradient.

```text
if surrogate1 < surrogate2:
    dLoss_dLogProb = -(ratio * advantage) / batchSize
else:
    dLoss_dLogProb = 0
```

## Gradient With Respect To Policy Mean

For a diagonal Gaussian with fixed variance:

```text
logProb = sum_i -0.5 * ((action_i - mean_i)^2 / var_i + const)
```

So:

```text
dLogProb_dMean_i = (action_i - mean_i) / var_i
dLoss_dMean_i = dLoss_dLogProb * dLogProb_dMean_i
```

## Gradient With Respect To Value

```text
dLoss_dValue = valueCoef * (value - return) / batchSize
```

## Head Gradients

Policy head:

```text
dWpi += outer(dLoss_dMean, h)
dbpi += dLoss_dMean
```

Value head:

```text
dWv += dLoss_dValue * h
dbv += dLoss_dValue
```

## Hidden Gradient

```text
dLoss_dh = Wpi^T * dLoss_dMean + Wv * dLoss_dValue
```

Then apply the ReLU mask:

```text
dLoss_dz_j = dLoss_dh_j if z_j > 0 else 0
```

## Trunk Gradients

```text
dW1 += outer(dLoss_dz, obs)
db1 += dLoss_dz
```

## SGD Update

After summing gradients over the batch:

```text
param = param - learningRate * grad
```

## Validation Plan

1. Build a synthetic PPO batch with actions not equal to the current means.
2. Compute loss before the update.
3. Apply one or a few SGD steps.
4. Verify:
   - parameters changed
   - forward outputs changed
   - loss decreased on the synthetic batch
5. Repeat on the real stored cartpole rollout batch.

## What This Does Not Do Yet

This step does not include:

- Adam or momentum
- minibatch shuffling
- gradient clipping
- value clipping
- Metal-side backprop
- a general autodiff engine
