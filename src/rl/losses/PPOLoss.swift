import Foundation

struct PPOConfig {
    let clipEpsilon: Float
    let valueCoefficient: Float
    let entropyCoefficient: Float
}

struct PPOLossBreakdown {
    let sampleCount: Int
    let policyLoss: Float
    let valueLoss: Float
    let entropyBonus: Float
    let totalLoss: Float
    let meanRatio: Float
}

func computePPOLoss(
    oldLogProbs: [Float],
    newLogProbs: [Float],
    advantages: [Float],
    returns: [Float],
    newValues: [Float],
    entropies: [Float],
    config: PPOConfig
) throws -> PPOLossBreakdown {
    let sampleCount = oldLogProbs.count

    guard newLogProbs.count == sampleCount,
          advantages.count == sampleCount,
          returns.count == sampleCount,
          newValues.count == sampleCount,
          entropies.count == sampleCount else {
        throw EnvProjectError.validationFailed(message: "PPO loss input sizes do not match.")
    }
    if sampleCount == 0 {
        throw EnvProjectError.validationFailed(message: "PPO loss requires at least one sample.")
    }

    var policySum: Float = 0.0
    var valueErrorSum: Float = 0.0
    var entropySum: Float = 0.0
    var ratioSum: Float = 0.0

    for index in 0..<sampleCount {
        let ratio = exp(newLogProbs[index] - oldLogProbs[index])
        let clippedRatio = min(max(ratio, 1.0 - config.clipEpsilon), 1.0 + config.clipEpsilon)
        let surrogate1 = ratio * advantages[index]
        let surrogate2 = clippedRatio * advantages[index]
        let selectedPolicyTerm = min(surrogate1, surrogate2)
        let valueError = newValues[index] - returns[index]

        if !ratio.isFinite || !selectedPolicyTerm.isFinite || !valueError.isFinite || !entropies[index].isFinite {
            throw EnvProjectError.validationFailed(message: "PPO loss encountered non-finite intermediate values.")
        }

        policySum += selectedPolicyTerm
        valueErrorSum += valueError * valueError
        entropySum += entropies[index]
        ratioSum += ratio
    }

    let invCount = 1.0 / Float(sampleCount)
    let policyLoss = -(policySum * invCount)
    let valueLoss = 0.5 * valueErrorSum * invCount
    let entropyBonus = entropySum * invCount
    let meanRatio = ratioSum * invCount
    let totalLoss = policyLoss + config.valueCoefficient * valueLoss - config.entropyCoefficient * entropyBonus

    return PPOLossBreakdown(
        sampleCount: sampleCount,
        policyLoss: policyLoss,
        valueLoss: valueLoss,
        entropyBonus: entropyBonus,
        totalLoss: totalLoss,
        meanRatio: meanRatio
    )
}

func computePPOLoss(
    storage: VectorRolloutStorage,
    estimates: AdvantageEstimates,
    policy: VectorGaussianActorCriticPolicy,
    observationSpec: VectorObservationSpec,
    actionSpec: VectorActionSpec,
    config: PPOConfig
) throws -> PPOLossBreakdown {
    guard let oldLogProbs = storage.logProbs else {
        throw EnvProjectError.validationFailed(message: "PPO loss requires rollout storage with old log-probabilities.")
    }

    if estimates.horizon != storage.horizon || estimates.envCount != storage.envCount {
        throw EnvProjectError.validationFailed(message: "PPO loss requires matching GAE and storage shapes.")
    }

    let envCount = storage.envCount
    let actionDim = storage.actionDim
    let observationDim = storage.observationDim
    let sampleCount = storage.horizon * envCount
    var newLogProbs: [Float] = []
    var newValues: [Float] = []
    var entropies: [Float] = []

    newLogProbs.reserveCapacity(sampleCount)
    newValues.reserveCapacity(sampleCount)
    entropies.reserveCapacity(sampleCount)

    for step in 0..<storage.horizon {
        let observationBase = step * envCount * observationDim
        let actionBase = step * envCount * actionDim
        let observationSlice = Array(storage.observations[observationBase..<(observationBase + envCount * observationDim)])
        let actionSlice = Array(storage.actions[actionBase..<(actionBase + envCount * actionDim)])

        let evaluation = try policy.evaluateGaussian(
            for: observationSlice,
            taking: actionSlice,
            envCount: envCount,
            observationSpec: observationSpec,
            actionSpec: actionSpec
        )

        newLogProbs.append(contentsOf: evaluation.logProbs)
        newValues.append(contentsOf: evaluation.values)
        entropies.append(contentsOf: evaluation.entropies)
    }

    return try computePPOLoss(
        oldLogProbs: oldLogProbs,
        newLogProbs: newLogProbs,
        advantages: estimates.advantages,
        returns: estimates.returns,
        newValues: newValues,
        entropies: entropies,
        config: config
    )
}
