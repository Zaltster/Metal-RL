import Foundation

struct GAEConfig {
    let gamma: Float
    let lambda: Float
}

struct AdvantageEstimates {
    let horizon: Int
    let envCount: Int
    let advantages: [Float]
    let returns: [Float]

    init(horizon: Int, envCount: Int, advantages: [Float], returns: [Float]) throws {
        let expectedCount = horizon * envCount
        if advantages.count != expectedCount {
            throw EnvProjectError.validationFailed(
                message: "AdvantageEstimates advantage size mismatch: expected \(expectedCount), got \(advantages.count)."
            )
        }
        if returns.count != expectedCount {
            throw EnvProjectError.validationFailed(
                message: "AdvantageEstimates return size mismatch: expected \(expectedCount), got \(returns.count)."
            )
        }

        self.horizon = horizon
        self.envCount = envCount
        self.advantages = advantages
        self.returns = returns
    }

    func advantage(step: Int, env: Int) -> Float {
        advantages[step * envCount + env]
    }

    func `return`(step: Int, env: Int) -> Float {
        returns[step * envCount + env]
    }
}

func computeGAE(
    storage: VectorRolloutStorage,
    config: GAEConfig
) throws -> AdvantageEstimates {
    guard let values = storage.values else {
        throw EnvProjectError.validationFailed(message: "GAE requires rollout storage with per-step values.")
    }
    guard let finalValues = storage.finalValues else {
        throw EnvProjectError.validationFailed(message: "GAE requires rollout storage with final bootstrap values.")
    }

    let horizon = storage.horizon
    let envCount = storage.envCount
    let envStride = envCount
    var advantages = Array(repeating: Float.zero, count: horizon * envCount)
    var returns = Array(repeating: Float.zero, count: horizon * envCount)

    for env in 0..<envCount {
        var lastAdvantage: Float = 0.0

        for step in Swift.stride(from: horizon - 1, through: 0, by: -1) {
            let index = step * envStride + env
            let done = storage.dones[index]
            let nextNonterminal: Float = done == 0 ? 1.0 : 0.0

            let nextValue: Float
            if step == horizon - 1 {
                nextValue = finalValues[env]
            } else {
                nextValue = values[(step + 1) * envStride + env]
            }

            let delta = storage.rewards[index] + config.gamma * nextValue * nextNonterminal - values[index]
            lastAdvantage = delta + config.gamma * config.lambda * nextNonterminal * lastAdvantage
            advantages[index] = lastAdvantage
            returns[index] = lastAdvantage + values[index]
        }
    }

    return try AdvantageEstimates(
        horizon: horizon,
        envCount: envCount,
        advantages: advantages,
        returns: returns
    )
}
