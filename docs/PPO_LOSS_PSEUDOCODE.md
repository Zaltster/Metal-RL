# PPO Loss Pseudocode

This step adds host-side PPO loss computation on top of:

- actor-critic rollout storage
- GAE advantages
- returns
- stored old log-probabilities

It is still:

- no gradients
- no optimizer
- no minibatch updates

The goal is to verify that the PPO objective math is correct before we introduce learning.

## Inputs

Per stored transition `(t, e)`:

- `oldLogProb[t, e]`
- `advantage[t, e]`
- `return[t, e]`
- `storedAction[t, e]`
- `storedObservation[t, e]`

From the current policy evaluated on the stored observation/action pair:

- `newLogProb[t, e]`
- `newValue[t, e]`
- `entropy[t, e]`

## Ratio

```text
ratio[t, e] = exp(newLogProb[t, e] - oldLogProb[t, e])
```

## Clipped Policy Objective

```text
unclipped = ratio * advantage
clippedRatio = clamp(ratio, 1 - clipEpsilon, 1 + clipEpsilon)
clipped = clippedRatio * advantage

policyLossTerm = min(unclipped, clipped)
policyLoss = -mean(policyLossTerm)
```

The minus sign is there because PPO usually maximizes the surrogate objective, while loss minimization frameworks expect the opposite sign.

## Value Loss

Use the simple unclipped value loss first:

```text
valueLoss = 0.5 * mean((newValue - return)^2)
```

## Entropy Bonus

```text
entropyBonus = mean(entropy)
```

## Total Loss

```text
totalLoss = policyLoss
          + valueCoef * valueLoss
          - entropyCoef * entropyBonus
```

## Array-Level Pseudocode

```text
for each sample i:
    ratio = exp(newLogProb[i] - oldLogProb[i])
    clippedRatio = clamp(ratio, 1 - clipEpsilon, 1 + clipEpsilon)

    surrogate1 = ratio * advantage[i]
    surrogate2 = clippedRatio * advantage[i]

    policyTerms[i] = min(surrogate1, surrogate2)
    valueError = newValue[i] - return[i]
    valueTerms[i] = valueError * valueError

policyLoss = -mean(policyTerms)
valueLoss = 0.5 * mean(valueTerms)
entropyBonus = mean(entropy)

totalLoss = policyLoss + valueCoef * valueLoss - entropyCoef * entropyBonus
```

## Validation Targets

1. Synthetic hand-computed example matches expected loss components.
2. Evaluating the same fixed policy used to produce the rollout gives:
   - ratio = 1 everywhere
   - deterministic replay of the same PPO losses
3. CPU and GPU fixed MLP policies produce matching PPO losses on matching stored rollouts.
4. Alternate fixed policies change the PPO loss.

## Current Policy Distribution Scope

The fixed MLP actor-critic policy uses:

- deterministic action means or seeded stochastic Gaussian samples as stored rollout actions
- a fixed diagonal Gaussian log-std for log-probability and entropy math

The log-std is not learned yet. Stochastic samples are clamped to environment action bounds before stepping, and PPO stores the Gaussian log-probability of the executed action.
