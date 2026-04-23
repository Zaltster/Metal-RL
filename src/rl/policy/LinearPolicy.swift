import Foundation

protocol VectorPolicy {
    func actions(
        for observations: [Float],
        envCount: Int,
        observationSpec: VectorObservationSpec,
        actionSpec: VectorActionSpec
    ) throws -> [Float]
}

struct LinearPolicy: VectorPolicy {
    let weights: [Float]
    let bias: [Float]

    func actions(
        for observations: [Float],
        envCount: Int,
        observationSpec: VectorObservationSpec,
        actionSpec: VectorActionSpec
    ) throws -> [Float] {
        let obsDim = observationSpec.elementsPerEnv
        let actionDim = actionSpec.dimensionsPerEnv

        if observations.count != envCount * obsDim {
            throw EnvProjectError.validationFailed(
                message: "LinearPolicy observation size mismatch: expected \(envCount * obsDim), got \(observations.count)."
            )
        }
        if weights.count != actionDim * obsDim {
            throw EnvProjectError.validationFailed(
                message: "LinearPolicy weight size mismatch: expected \(actionDim * obsDim), got \(weights.count)."
            )
        }
        if bias.count != actionDim {
            throw EnvProjectError.validationFailed(
                message: "LinearPolicy bias size mismatch: expected \(actionDim), got \(bias.count)."
            )
        }

        var actions = Array(repeating: Float.zero, count: envCount * actionDim)
        for envIndex in 0..<envCount {
            let obsBase = envIndex * obsDim
            let actionBase = envIndex * actionDim
            for actionIndex in 0..<actionDim {
                var value = bias[actionIndex]
                let weightBase = actionIndex * obsDim
                for obsIndex in 0..<obsDim {
                    value += weights[weightBase + obsIndex] * observations[obsBase + obsIndex]
                }
                actions[actionBase + actionIndex] = min(max(value, actionSpec.minValue), actionSpec.maxValue)
            }
        }

        return actions
    }
}

func makeReferenceLinearPolicy(
    observationSpec: VectorObservationSpec,
    actionSpec: VectorActionSpec
) throws -> LinearPolicy {
    if observationSpec.elementsPerEnv != 4 || actionSpec.dimensionsPerEnv != 1 {
        throw EnvProjectError.validationFailed(
            message: "Reference linear policy currently expects cartpole dims obs=4 and act=1."
        )
    }

    return LinearPolicy(
        weights: [0.85, 0.15, -2.75, -0.45],
        bias: [0.05]
    )
}

func makeAlternateLinearPolicy(
    observationSpec: VectorObservationSpec,
    actionSpec: VectorActionSpec
) throws -> LinearPolicy {
    if observationSpec.elementsPerEnv != 4 || actionSpec.dimensionsPerEnv != 1 {
        throw EnvProjectError.validationFailed(
            message: "Alternate linear policy currently expects cartpole dims obs=4 and act=1."
        )
    }

    return LinearPolicy(
        weights: [0.45, -0.35, -1.95, 0.25],
        bias: [-0.08]
    )
}
