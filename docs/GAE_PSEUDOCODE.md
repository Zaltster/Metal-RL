# GAE Pseudocode

This document describes the first trajectory postprocessing step after fixed actor-critic rollout storage:

- host-side only
- no gradients
- no PPO loss yet
- deterministic validation against hand-computed examples

The goal is to turn stored rollout data into:

- advantages
- returns

using Generalized Advantage Estimation (GAE).

## Inputs

For each time step `t` and environment `e`, we need:

- `reward[t, e]`
- `done[t, e]`
- `value[t, e]`

And for the bootstrap at the end of the rollout:

- `finalValue[e]`

The current rollout storage already contains exactly that information.

## Core Formula

For each `(t, e)`:

```text
nextNonterminal = 1 - done[t, e]
nextValue =
    finalValue[e]                  if t is the last step
    value[t + 1, e]               otherwise

delta[t, e] = reward[t, e]
            + gamma * nextValue * nextNonterminal
            - value[t, e]
```

Then compute advantages backward in time:

```text
advantage[t, e] = delta[t, e]
                + gamma * lambda * nextNonterminal * advantage[t + 1, e]
```

with the recursion starting from the last step and moving to the first.

Finally:

```text
return[t, e] = advantage[t, e] + value[t, e]
```

## Backward Pass Pseudocode

```text
for each env e:
    lastAdvantage = 0

    for t from horizon - 1 down to 0:
        if t == horizon - 1:
            nextValue = finalValue[e]
        else:
            nextValue = value[t + 1, e]

        nextNonterminal = 1 - done[t, e]

        delta = reward[t, e]
              + gamma * nextValue * nextNonterminal
              - value[t, e]

        lastAdvantage = delta
                      + gamma * lambda * nextNonterminal * lastAdvantage

        advantage[t, e] = lastAdvantage
        return[t, e] = advantage[t, e] + value[t, e]
```

## Why Done Masking Matters

If `done[t, e] = 1`, then:

- the bootstrap term must be cut
- the recursive carry from future steps must be cut

So for terminal transitions:

```text
nextNonterminal = 0
delta = reward[t, e] - value[t, e]
advantage[t, e] = delta
```

That prevents information from the next episode from leaking backward into the previous one.

## Hand-Checkable Example

Use:

- `gamma = 0.9`
- `lambda = 0.8`

For one environment:

```text
rewards = [1.0, 2.0, 3.0]
dones   = [0,   0,   1]
values  = [0.5, 0.6, 0.7]
```

Then:

```text
adv[2] = 3.0 - 0.7 = 2.3
adv[1] = (2.0 + 0.9 * 0.7 - 0.6) + 0.9 * 0.8 * 2.3
       = 2.03 + 1.656
       = 3.686
adv[0] = (1.0 + 0.9 * 0.6 - 0.5) + 0.9 * 0.8 * 3.686
       = 1.04 + 2.65392
       = 3.69392
```

Returns:

```text
ret[0] = 3.69392 + 0.5 = 4.19392
ret[1] = 3.686   + 0.6 = 4.286
ret[2] = 2.3     + 0.7 = 3.0
```

That is the kind of example we should keep in the validation harness.

## What We Will Test

1. Hand-computed synthetic example matches exactly within tolerance.
2. Done masking cuts recursion at episode boundaries.
3. Final bootstrap value is used only when the last step is nonterminal.
4. `returns = advantages + values` for every stored transition.
5. Real actor-critic storage from the cartpole harness produces finite outputs with the correct shapes.

## What This Does Not Do Yet

This step does not include:

- policy log-probabilities
- PPO clipped objective
- value loss
- entropy bonus
- minibatch shuffling
- gradient updates

Those come after GAE is verified.
