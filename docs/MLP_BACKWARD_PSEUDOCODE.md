# MLP Backward Pseudocode

This document describes the first training-gradient path for the repository:

- CPU reference path plus a first Metal parity kernel
- one-hidden-layer actor-critic MLP
- hand-derived gradients
- simple SGD and Adam
- host-side checkpoint round-trip for trained weights and persistent GPU Adam state
- explicit deterministic-mean and seeded stochastic-Gaussian rollout sampling modes
- no custom autodiff engine yet

The goal is to prove that the current PPO math can drive parameter updates, then reproduce the same gradient and optimizer math in Metal before expanding the GPU-native training path.

## Model

For one sample:

```text
z = W1 * obs + b1
h = relu(z)

mean = Wpi * h + bpi
value = Wv * h + bv
```

The policy uses a fixed diagonal Gaussian log-std, so only the mean is trainable in this step.
Actor-critic rollouts can either use deterministic mean actions or seeded stochastic Gaussian samples. Stochastic rollout actions are clamped to the environment action bounds before stepping, and the PPO batch stores the Gaussian log-probability of the executed action.

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

## First Metal Gradient Kernel

The current Metal milestone computes per-sample gradients:

```text
sampleGrad[i] = gradient contribution for sample i
```

The current reduction kernel sums those per-sample gradient buffers on GPU:

```text
grad = sum_i sampleGrad[i]
```

The current Metal SGD update applies the same rule to GPU parameter buffers:

```text
param_gpu = param_gpu - learningRate * grad_gpu
```

The persistent Metal Adam update keeps first- and second-moment buffers resident beside the parameter buffers:

```text
m_gpu = beta1 * m_gpu + (1 - beta1) * grad_gpu
v_gpu = beta2 * v_gpu + (1 - beta2) * grad_gpu^2
mHat = m_gpu / (1 - beta1^t)
vHat = v_gpu / (1 - beta2^t)
param_gpu = param_gpu - learningRate * mHat / (sqrt(vHat) + epsilon)
```

The current single-step helper computes pre/post PPO loss around the Metal SGD update and compares the result against the CPU SGD step on both synthetic and real rollout PPO batches. A persistent Metal trainable model can also keep parameter and Adam state buffers resident across repeated optimizer updates and match repeated CPU SGD or Adam. A tiny persistent-GPU optimizer training loop is validated against the CPU SGD and Adam loops, rollout policy sync copies directly between Metal buffers, and a training-state checkpoint can restore GPU parameters plus Adam state before the next update.

## Validation Plan

1. Build a synthetic PPO batch with actions not equal to the current means.
2. Choose old log-probabilities that exercise nonzero unclipped policy-gradient branches.
3. Compare CPU gradients against Metal per-sample gradients reduced by the Metal reduction kernel.
4. Compare CPU-updated weights against Metal-updated weights after one SGD step.
5. Compare CPU and Metal single-step pre/post losses on synthetic and real rollout PPO batches.
6. Compare repeated CPU SGD against repeated persistent-buffer Metal SGD.
7. Compare repeated CPU Adam against repeated persistent-buffer Metal Adam.
8. Compare tiny CPU SGD and Adam training loops against persistent-GPU optimizer training loops.
9. Save and restore persistent Metal parameters plus Adam state, then compare the next Adam step against the uninterrupted path.
10. Require rollout code to declare `deterministic-mean` or `stochastic-gaussian` sampling, and validate seeded stochastic replay plus CPU/GPU rollout-storage parity.
11. Copy persistent trainable GPU buffers directly into rollout policy GPU buffers before rollout.
12. Compute loss before the update.
13. Apply one or a few optimizer steps.
14. Verify:
   - parameters changed
   - forward outputs changed
   - loss decreased on the synthetic batch
15. Repeat on the real stored cartpole rollout batch.

## What This Does Not Do Yet

This step does not include:

- gradient clipping
- value clipping
- learned log-std
- production resume flow around persistent GPU optimizer-state checkpoints
- a general autodiff engine
