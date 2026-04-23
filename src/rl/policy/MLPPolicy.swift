import Foundation

struct MLPPolicy: VectorActorCriticPolicy {
    let inputWeights: [Float]
    let inputBias: [Float]
    let outputWeights: [Float]
    let outputBias: [Float]
    let valueWeights: [Float]
    let valueBias: Float

    var inputDim: Int {
        inputWeights.isEmpty ? 0 : inputWeights.count / hiddenDim
    }

    var hiddenDim: Int {
        inputBias.count
    }

    var actionDim: Int {
        outputBias.count
    }

    func evaluate(
        for observations: [Float],
        envCount: Int,
        observationSpec: VectorObservationSpec,
        actionSpec: VectorActionSpec
    ) throws -> PolicyValueOutputs {
        let obsDim = observationSpec.elementsPerEnv
        let actionDim = actionSpec.dimensionsPerEnv

        if observations.count != envCount * obsDim {
            throw EnvProjectError.validationFailed(
                message: "MLPPolicy observation size mismatch: expected \(envCount * obsDim), got \(observations.count)."
            )
        }
        if inputWeights.count != hiddenDim * obsDim {
            throw EnvProjectError.validationFailed(
                message: "MLPPolicy input weight size mismatch: expected \(hiddenDim * obsDim), got \(inputWeights.count)."
            )
        }
        if outputWeights.count != actionDim * hiddenDim {
            throw EnvProjectError.validationFailed(
                message: "MLPPolicy output weight size mismatch: expected \(actionDim * hiddenDim), got \(outputWeights.count)."
            )
        }
        if valueWeights.count != hiddenDim {
            throw EnvProjectError.validationFailed(
                message: "MLPPolicy value weight size mismatch: expected \(hiddenDim), got \(valueWeights.count)."
            )
        }
        if outputBias.count != actionDim {
            throw EnvProjectError.validationFailed(
                message: "MLPPolicy output bias size mismatch: expected \(actionDim), got \(outputBias.count)."
            )
        }

        var hidden = Array(repeating: Float.zero, count: hiddenDim)
        var actions = Array(repeating: Float.zero, count: envCount * actionDim)
        var values = Array(repeating: Float.zero, count: envCount)

        for envIndex in 0..<envCount {
            let obsBase = envIndex * obsDim

            for hiddenIndex in 0..<hiddenDim {
                let weightBase = hiddenIndex * obsDim
                var value = inputBias[hiddenIndex]
                for obsIndex in 0..<obsDim {
                    value += inputWeights[weightBase + obsIndex] * observations[obsBase + obsIndex]
                }
                hidden[hiddenIndex] = max(0.0, value)
            }

            let actionBase = envIndex * actionDim
            for actionIndex in 0..<actionDim {
                let weightBase = actionIndex * hiddenDim
                var value = outputBias[actionIndex]
                for hiddenIndex in 0..<hiddenDim {
                    value += outputWeights[weightBase + hiddenIndex] * hidden[hiddenIndex]
                }
                actions[actionBase + actionIndex] = min(max(value, actionSpec.minValue), actionSpec.maxValue)
            }

            var stateValue = valueBias
            for hiddenIndex in 0..<hiddenDim {
                stateValue += valueWeights[hiddenIndex] * hidden[hiddenIndex]
            }
            values[envIndex] = stateValue
        }

        return PolicyValueOutputs(actions: actions, values: values)
    }
}

extension MLPPolicy: VectorGaussianActorCriticPolicy {
    func evaluateGaussian(
        for observations: [Float],
        taking actions: [Float]?,
        envCount: Int,
        observationSpec: VectorObservationSpec,
        actionSpec: VectorActionSpec
    ) throws -> GaussianPolicyOutputs {
        let base = try evaluate(
            for: observations,
            envCount: envCount,
            observationSpec: observationSpec,
            actionSpec: actionSpec
        )
        let actionDim = actionSpec.dimensionsPerEnv
        let chosenActions = actions ?? base.actions

        if chosenActions.count != envCount * actionDim {
            throw EnvProjectError.validationFailed(
                message: "MLPPolicy chosen-action size mismatch: expected \(envCount * actionDim), got \(chosenActions.count)."
            )
        }

        let logStd = Array(repeating: Float(-0.35), count: actionDim)
        let entropies = Array(repeating: gaussianEntropy(logStd: logStd), count: envCount)
        var logProbs = Array(repeating: Float.zero, count: envCount)

        for envIndex in 0..<envCount {
            let actionBase = envIndex * actionDim
            let actionSlice = Array(chosenActions[actionBase..<(actionBase + actionDim)])
            let meanSlice = Array(base.actions[actionBase..<(actionBase + actionDim)])
            logProbs[envIndex] = gaussianLogProb(action: actionSlice, mean: meanSlice, logStd: logStd)
        }

        return GaussianPolicyOutputs(
            actions: chosenActions,
            actionMeans: base.actions,
            values: base.values,
            logProbs: logProbs,
            entropies: entropies,
            logStd: logStd
        )
    }
}

func gaussianLogProb(action: [Float], mean: [Float], logStd: [Float]) -> Float {
    let logTwoPi = Float(log(2.0 * Double.pi))
    var total: Float = 0.0

    for index in action.indices {
        let diff = action[index] - mean[index]
        let currentLogStd = logStd[index]
        let variance = exp(2.0 * currentLogStd)
        total += -0.5 * ((diff * diff) / variance + 2.0 * currentLogStd + logTwoPi)
    }

    return total
}

func gaussianEntropy(logStd: [Float]) -> Float {
    let constant = Float(0.5 * (1.0 + log(2.0 * Double.pi)))
    return logStd.reduce(0.0) { partial, value in
        partial + value + constant
    }
}

func makeReferenceMLPPolicy(
    observationSpec: VectorObservationSpec,
    actionSpec: VectorActionSpec
) throws -> MLPPolicy {
    if observationSpec.elementsPerEnv != 4 || actionSpec.dimensionsPerEnv != 1 {
        throw EnvProjectError.validationFailed(
            message: "Reference MLP policy currently expects cartpole dims obs=4 and act=1."
        )
    }

    return MLPPolicy(
        inputWeights: [
            0.60, -0.20, -1.10, 0.15,
            -0.35, 0.40, 0.95, -0.25,
            0.20, 0.55, -0.60, 0.70,
            -0.75, -0.10, 1.30, 0.35,
            0.90, 0.25, -0.45, -0.80,
            -0.15, 0.70, 0.40, 0.50,
            0.30, -0.65, 0.20, -0.55,
            -0.50, 0.85, -0.30, 0.45,
        ],
        inputBias: [0.10, -0.05, 0.08, -0.02, 0.04, 0.12, -0.09, 0.03],
        outputWeights: [0.70, -0.45, 0.30, -0.85, 0.55, 0.25, -0.60, 0.40],
        outputBias: [0.02],
        valueWeights: [0.40, -0.15, 0.35, -0.55, 0.20, 0.10, -0.30, 0.25],
        valueBias: 0.12
    )
}

func makeAlternateMLPPolicy(
    observationSpec: VectorObservationSpec,
    actionSpec: VectorActionSpec
) throws -> MLPPolicy {
    if observationSpec.elementsPerEnv != 4 || actionSpec.dimensionsPerEnv != 1 {
        throw EnvProjectError.validationFailed(
            message: "Alternate MLP policy currently expects cartpole dims obs=4 and act=1."
        )
    }

    return MLPPolicy(
        inputWeights: [
            0.35, 0.15, -0.85, -0.20,
            -0.10, 0.55, 0.70, -0.45,
            0.50, -0.25, -0.35, 0.60,
            -0.55, 0.05, 1.05, 0.20,
            0.65, -0.30, -0.15, -0.60,
            0.05, 0.45, 0.25, 0.35,
            0.15, -0.40, 0.55, -0.25,
            -0.30, 0.60, -0.10, 0.20,
        ],
        inputBias: [-0.03, 0.07, 0.02, 0.05, -0.01, 0.09, -0.04, 0.11],
        outputWeights: [0.45, -0.20, 0.55, -0.65, 0.35, 0.10, -0.30, 0.50],
        outputBias: [-0.06],
        valueWeights: [0.25, -0.05, 0.15, -0.35, 0.45, 0.20, -0.10, 0.30],
        valueBias: -0.08
    )
}
