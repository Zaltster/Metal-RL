import Foundation

struct PPOBatch {
    let sampleCount: Int
    let observationDim: Int
    let actionDim: Int
    let observations: [Float]
    let actions: [Float]
    let oldLogProbs: [Float]
    let advantages: [Float]
    let returns: [Float]

    init(
        sampleCount: Int,
        observationDim: Int,
        actionDim: Int,
        observations: [Float],
        actions: [Float],
        oldLogProbs: [Float],
        advantages: [Float],
        returns: [Float]
    ) throws {
        if sampleCount <= 0 {
            throw EnvProjectError.validationFailed(message: "PPOBatch requires at least one sample.")
        }
        if observationDim <= 0 || actionDim <= 0 {
            throw EnvProjectError.validationFailed(message: "PPOBatch dimensions must be positive.")
        }
        if observations.count != sampleCount * observationDim {
            throw EnvProjectError.validationFailed(
                message: "PPOBatch observation size mismatch: expected \(sampleCount * observationDim), got \(observations.count)."
            )
        }
        if actions.count != sampleCount * actionDim {
            throw EnvProjectError.validationFailed(
                message: "PPOBatch action size mismatch: expected \(sampleCount * actionDim), got \(actions.count)."
            )
        }
        if oldLogProbs.count != sampleCount || advantages.count != sampleCount || returns.count != sampleCount {
            throw EnvProjectError.validationFailed(message: "PPOBatch scalar size mismatch.")
        }

        self.sampleCount = sampleCount
        self.observationDim = observationDim
        self.actionDim = actionDim
        self.observations = observations
        self.actions = actions
        self.oldLogProbs = oldLogProbs
        self.advantages = advantages
        self.returns = returns
    }
}

struct SGDConfig {
    let learningRate: Float
}

struct AdamConfig {
    let learningRate: Float
    let beta1: Float
    let beta2: Float
    let epsilon: Float
}

enum CPUOptimizerConfig {
    case sgd(SGDConfig)
    case adam(AdamConfig)
}

struct TrainStepSummary {
    let preLoss: PPOLossBreakdown
    let postLoss: PPOLossBreakdown
    let parameterDeltaL1: Float
}

struct MLPGradients {
    var inputWeights: [Float]
    var inputBias: [Float]
    var outputWeights: [Float]
    var outputBias: [Float]
    var valueWeights: [Float]
    var valueBias: Float

    init(model: TrainableMLPActorCritic) {
        inputWeights = Array(repeating: 0.0, count: model.inputWeights.count)
        inputBias = Array(repeating: 0.0, count: model.inputBias.count)
        outputWeights = Array(repeating: 0.0, count: model.outputWeights.count)
        outputBias = Array(repeating: 0.0, count: model.outputBias.count)
        valueWeights = Array(repeating: 0.0, count: model.valueWeights.count)
        valueBias = 0.0
    }
}

struct AdamState {
    var timestep: Int
    var inputWeightsM: [Float]
    var inputWeightsV: [Float]
    var inputBiasM: [Float]
    var inputBiasV: [Float]
    var outputWeightsM: [Float]
    var outputWeightsV: [Float]
    var outputBiasM: [Float]
    var outputBiasV: [Float]
    var valueWeightsM: [Float]
    var valueWeightsV: [Float]
    var valueBiasM: Float
    var valueBiasV: Float

    init(model: TrainableMLPActorCritic) {
        timestep = 0
        inputWeightsM = Array(repeating: 0.0, count: model.inputWeights.count)
        inputWeightsV = Array(repeating: 0.0, count: model.inputWeights.count)
        inputBiasM = Array(repeating: 0.0, count: model.inputBias.count)
        inputBiasV = Array(repeating: 0.0, count: model.inputBias.count)
        outputWeightsM = Array(repeating: 0.0, count: model.outputWeights.count)
        outputWeightsV = Array(repeating: 0.0, count: model.outputWeights.count)
        outputBiasM = Array(repeating: 0.0, count: model.outputBias.count)
        outputBiasV = Array(repeating: 0.0, count: model.outputBias.count)
        valueWeightsM = Array(repeating: 0.0, count: model.valueWeights.count)
        valueWeightsV = Array(repeating: 0.0, count: model.valueWeights.count)
        valueBiasM = 0.0
        valueBiasV = 0.0
    }
}

struct TrainableMLPActorCritic: VectorGaussianActorCriticPolicy {
    var inputWeights: [Float]
    var inputBias: [Float]
    var outputWeights: [Float]
    var outputBias: [Float]
    var valueWeights: [Float]
    var valueBias: Float
    var logStd: [Float]

    var hiddenDim: Int { inputBias.count }

    init(policy: MLPPolicy, logStd: [Float]? = nil) {
        self.inputWeights = policy.inputWeights
        self.inputBias = policy.inputBias
        self.outputWeights = policy.outputWeights
        self.outputBias = policy.outputBias
        self.valueWeights = policy.valueWeights
        self.valueBias = policy.valueBias
        self.logStd = logStd ?? Array(repeating: Float(-0.35), count: policy.outputBias.count)
    }

    func asPolicy() -> MLPPolicy {
        MLPPolicy(
            inputWeights: inputWeights,
            inputBias: inputBias,
            outputWeights: outputWeights,
            outputBias: outputBias,
            valueWeights: valueWeights,
            valueBias: valueBias
        )
    }

    func evaluateGaussian(
        for observations: [Float],
        taking actions: [Float]?,
        envCount: Int,
        observationSpec: VectorObservationSpec,
        actionSpec: VectorActionSpec
    ) throws -> GaussianPolicyOutputs {
        let policy = asPolicy()
        let base = try policy.evaluate(
            for: observations,
            envCount: envCount,
            observationSpec: observationSpec,
            actionSpec: actionSpec
        )
        let actionDim = actionSpec.dimensionsPerEnv
        let chosenActions = actions ?? base.actions

        if chosenActions.count != envCount * actionDim {
            throw EnvProjectError.validationFailed(
                message: "TrainableMLPActorCritic chosen-action size mismatch: expected \(envCount * actionDim), got \(chosenActions.count)."
            )
        }

        let entropyValue = gaussianEntropy(logStd: logStd)
        let entropies = Array(repeating: entropyValue, count: envCount)
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

    mutating func applySGDStep(
        batch: PPOBatch,
        observationSpec: VectorObservationSpec,
        actionSpec: VectorActionSpec,
        ppoConfig: PPOConfig,
        sgdConfig: SGDConfig
    ) throws -> TrainStepSummary {
        let preLoss = try computePPOLoss(
            oldLogProbs: batch.oldLogProbs,
            newLogProbs: evaluateGaussian(
                for: batch.observations,
                taking: batch.actions,
                envCount: batch.sampleCount,
                observationSpec: observationSpec,
                actionSpec: actionSpec
            ).logProbs,
            advantages: batch.advantages,
            returns: batch.returns,
            newValues: evaluateGaussian(
                for: batch.observations,
                taking: batch.actions,
                envCount: batch.sampleCount,
                observationSpec: observationSpec,
                actionSpec: actionSpec
            ).values,
            entropies: evaluateGaussian(
                for: batch.observations,
                taking: batch.actions,
                envCount: batch.sampleCount,
                observationSpec: observationSpec,
                actionSpec: actionSpec
            ).entropies,
            config: ppoConfig
        )

        let gradients = try computeGradients(
            batch: batch,
            observationSpec: observationSpec,
            actionSpec: actionSpec,
            ppoConfig: ppoConfig
        )
        let delta = apply(grads: gradients, learningRate: sgdConfig.learningRate)

        let postEvaluation = try evaluateGaussian(
            for: batch.observations,
            taking: batch.actions,
            envCount: batch.sampleCount,
            observationSpec: observationSpec,
            actionSpec: actionSpec
        )
        let postLoss = try computePPOLoss(
            oldLogProbs: batch.oldLogProbs,
            newLogProbs: postEvaluation.logProbs,
            advantages: batch.advantages,
            returns: batch.returns,
            newValues: postEvaluation.values,
            entropies: postEvaluation.entropies,
            config: ppoConfig
        )

        return TrainStepSummary(preLoss: preLoss, postLoss: postLoss, parameterDeltaL1: delta)
    }

    mutating func applyAdamStep(
        batch: PPOBatch,
        observationSpec: VectorObservationSpec,
        actionSpec: VectorActionSpec,
        ppoConfig: PPOConfig,
        adamState: inout AdamState,
        adamConfig: AdamConfig
    ) throws -> TrainStepSummary {
        let preLoss = try evaluateBatchLoss(
            batch: batch,
            observationSpec: observationSpec,
            actionSpec: actionSpec,
            ppoConfig: ppoConfig
        )

        let gradients = try computeGradients(
            batch: batch,
            observationSpec: observationSpec,
            actionSpec: actionSpec,
            ppoConfig: ppoConfig
        )
        let delta = applyAdam(grads: gradients, state: &adamState, config: adamConfig)

        let postLoss = try evaluateBatchLoss(
            batch: batch,
            observationSpec: observationSpec,
            actionSpec: actionSpec,
            ppoConfig: ppoConfig
        )

        return TrainStepSummary(preLoss: preLoss, postLoss: postLoss, parameterDeltaL1: delta)
    }

    mutating func applyOptimizerStep(
        batch: PPOBatch,
        observationSpec: VectorObservationSpec,
        actionSpec: VectorActionSpec,
        ppoConfig: PPOConfig,
        optimizer: CPUOptimizerConfig,
        adamState: inout AdamState?
    ) throws -> TrainStepSummary {
        switch optimizer {
        case let .sgd(config):
            return try applySGDStep(
                batch: batch,
                observationSpec: observationSpec,
                actionSpec: actionSpec,
                ppoConfig: ppoConfig,
                sgdConfig: config
            )
        case let .adam(config):
            if adamState == nil {
                adamState = AdamState(model: self)
            }
            return try applyAdamStep(
                batch: batch,
                observationSpec: observationSpec,
                actionSpec: actionSpec,
                ppoConfig: ppoConfig,
                adamState: &adamState!,
                adamConfig: config
            )
        }
    }

    func computeGradients(
        batch: PPOBatch,
        observationSpec: VectorObservationSpec,
        actionSpec: VectorActionSpec,
        ppoConfig: PPOConfig
    ) throws -> MLPGradients {
        let obsDim = observationSpec.elementsPerEnv
        let actionDim = actionSpec.dimensionsPerEnv
        if obsDim != batch.observationDim || actionDim != batch.actionDim {
            throw EnvProjectError.validationFailed(message: "TrainableMLPActorCritic batch spec mismatch.")
        }

        var grads = MLPGradients(model: self)

        for sampleIndex in 0..<batch.sampleCount {
            let obsBase = sampleIndex * obsDim
            let actionBase = sampleIndex * actionDim
            let obs = Array(batch.observations[obsBase..<(obsBase + obsDim)])
            let action = Array(batch.actions[actionBase..<(actionBase + actionDim)])
            let forward = forwardSingle(observation: obs)

            let newLogProb = gaussianLogProb(action: action, mean: forward.mean, logStd: logStd)
            let ratio = exp(newLogProb - batch.oldLogProbs[sampleIndex])
            let clippedRatio = min(max(ratio, 1.0 - ppoConfig.clipEpsilon), 1.0 + ppoConfig.clipEpsilon)
            let surrogate1 = ratio * batch.advantages[sampleIndex]
            let surrogate2 = clippedRatio * batch.advantages[sampleIndex]

            let dLoss_dLogProb: Float
            if surrogate1 < surrogate2 {
                dLoss_dLogProb = -(ratio * batch.advantages[sampleIndex]) / Float(batch.sampleCount)
            } else {
                dLoss_dLogProb = 0.0
            }

            var dLoss_dMean = Array(repeating: Float.zero, count: actionDim)
            for actionIndex in 0..<actionDim {
                let variance = exp(2.0 * logStd[actionIndex])
                dLoss_dMean[actionIndex] = dLoss_dLogProb * ((action[actionIndex] - forward.mean[actionIndex]) / variance)
            }

            let dLoss_dValue = ppoConfig.valueCoefficient * (forward.value - batch.returns[sampleIndex]) / Float(batch.sampleCount)

            for actionIndex in 0..<actionDim {
                let outputWeightBase = actionIndex * hiddenDim
                for hiddenIndex in 0..<hiddenDim {
                    grads.outputWeights[outputWeightBase + hiddenIndex] += dLoss_dMean[actionIndex] * forward.hidden[hiddenIndex]
                }
                grads.outputBias[actionIndex] += dLoss_dMean[actionIndex]
            }

            for hiddenIndex in 0..<hiddenDim {
                grads.valueWeights[hiddenIndex] += dLoss_dValue * forward.hidden[hiddenIndex]
            }
            grads.valueBias += dLoss_dValue

            var dLoss_dHidden = Array(repeating: Float.zero, count: hiddenDim)
            for hiddenIndex in 0..<hiddenDim {
                var total: Float = valueWeights[hiddenIndex] * dLoss_dValue
                for actionIndex in 0..<actionDim {
                    total += outputWeights[actionIndex * hiddenDim + hiddenIndex] * dLoss_dMean[actionIndex]
                }
                dLoss_dHidden[hiddenIndex] = forward.preActivation[hiddenIndex] > 0.0 ? total : 0.0
            }

            for hiddenIndex in 0..<hiddenDim {
                let inputWeightBase = hiddenIndex * obsDim
                for obsIndex in 0..<obsDim {
                    grads.inputWeights[inputWeightBase + obsIndex] += dLoss_dHidden[hiddenIndex] * obs[obsIndex]
                }
                grads.inputBias[hiddenIndex] += dLoss_dHidden[hiddenIndex]
            }
        }

        return grads
    }

    private func evaluateBatchLoss(
        batch: PPOBatch,
        observationSpec: VectorObservationSpec,
        actionSpec: VectorActionSpec,
        ppoConfig: PPOConfig
    ) throws -> PPOLossBreakdown {
        let evaluation = try evaluateGaussian(
            for: batch.observations,
            taking: batch.actions,
            envCount: batch.sampleCount,
            observationSpec: observationSpec,
            actionSpec: actionSpec
        )

        return try computePPOLoss(
            oldLogProbs: batch.oldLogProbs,
            newLogProbs: evaluation.logProbs,
            advantages: batch.advantages,
            returns: batch.returns,
            newValues: evaluation.values,
            entropies: evaluation.entropies,
            config: ppoConfig
        )
    }

    private mutating func apply(grads: MLPGradients, learningRate: Float) -> Float {
        var totalDelta: Float = 0.0

        for index in inputWeights.indices {
            let delta = learningRate * grads.inputWeights[index]
            inputWeights[index] -= delta
            totalDelta += abs(delta)
        }
        for index in inputBias.indices {
            let delta = learningRate * grads.inputBias[index]
            inputBias[index] -= delta
            totalDelta += abs(delta)
        }
        for index in outputWeights.indices {
            let delta = learningRate * grads.outputWeights[index]
            outputWeights[index] -= delta
            totalDelta += abs(delta)
        }
        for index in outputBias.indices {
            let delta = learningRate * grads.outputBias[index]
            outputBias[index] -= delta
            totalDelta += abs(delta)
        }
        for index in valueWeights.indices {
            let delta = learningRate * grads.valueWeights[index]
            valueWeights[index] -= delta
            totalDelta += abs(delta)
        }
        let valueBiasDelta = learningRate * grads.valueBias
        valueBias -= valueBiasDelta
        totalDelta += abs(valueBiasDelta)

        return totalDelta
    }

    private mutating func applyAdam(
        grads: MLPGradients,
        state: inout AdamState,
        config: AdamConfig
    ) -> Float {
        state.timestep += 1
        let t = Float(state.timestep)
        let biasCorrection1 = 1.0 - pow(config.beta1, t)
        let biasCorrection2 = 1.0 - pow(config.beta2, t)
        var totalDelta: Float = 0.0

        for index in inputWeights.indices {
            let delta = adamUpdate(
                param: &inputWeights[index],
                grad: grads.inputWeights[index],
                m: &state.inputWeightsM[index],
                v: &state.inputWeightsV[index],
                config: config,
                biasCorrection1: biasCorrection1,
                biasCorrection2: biasCorrection2
            )
            totalDelta += abs(delta)
        }
        for index in inputBias.indices {
            let delta = adamUpdate(
                param: &inputBias[index],
                grad: grads.inputBias[index],
                m: &state.inputBiasM[index],
                v: &state.inputBiasV[index],
                config: config,
                biasCorrection1: biasCorrection1,
                biasCorrection2: biasCorrection2
            )
            totalDelta += abs(delta)
        }
        for index in outputWeights.indices {
            let delta = adamUpdate(
                param: &outputWeights[index],
                grad: grads.outputWeights[index],
                m: &state.outputWeightsM[index],
                v: &state.outputWeightsV[index],
                config: config,
                biasCorrection1: biasCorrection1,
                biasCorrection2: biasCorrection2
            )
            totalDelta += abs(delta)
        }
        for index in outputBias.indices {
            let delta = adamUpdate(
                param: &outputBias[index],
                grad: grads.outputBias[index],
                m: &state.outputBiasM[index],
                v: &state.outputBiasV[index],
                config: config,
                biasCorrection1: biasCorrection1,
                biasCorrection2: biasCorrection2
            )
            totalDelta += abs(delta)
        }
        for index in valueWeights.indices {
            let delta = adamUpdate(
                param: &valueWeights[index],
                grad: grads.valueWeights[index],
                m: &state.valueWeightsM[index],
                v: &state.valueWeightsV[index],
                config: config,
                biasCorrection1: biasCorrection1,
                biasCorrection2: biasCorrection2
            )
            totalDelta += abs(delta)
        }
        totalDelta += abs(adamUpdate(
            param: &valueBias,
            grad: grads.valueBias,
            m: &state.valueBiasM,
            v: &state.valueBiasV,
            config: config,
            biasCorrection1: biasCorrection1,
            biasCorrection2: biasCorrection2
        ))

        return totalDelta
    }

    private func forwardSingle(observation: [Float]) -> ForwardCache {
        let obsDim = observation.count
        let actionDim = outputBias.count
        var preActivation = Array(repeating: Float.zero, count: hiddenDim)
        var hidden = Array(repeating: Float.zero, count: hiddenDim)

        for hiddenIndex in 0..<hiddenDim {
            let weightBase = hiddenIndex * obsDim
            var value = inputBias[hiddenIndex]
            for obsIndex in 0..<obsDim {
                value += inputWeights[weightBase + obsIndex] * observation[obsIndex]
            }
            preActivation[hiddenIndex] = value
            hidden[hiddenIndex] = max(0.0, value)
        }

        var mean = Array(repeating: Float.zero, count: actionDim)
        for actionIndex in 0..<actionDim {
            let weightBase = actionIndex * hiddenDim
            var value = outputBias[actionIndex]
            for hiddenIndex in 0..<hiddenDim {
                value += outputWeights[weightBase + hiddenIndex] * hidden[hiddenIndex]
            }
            mean[actionIndex] = value
        }

        var value = valueBias
        for hiddenIndex in 0..<hiddenDim {
            value += valueWeights[hiddenIndex] * hidden[hiddenIndex]
        }

        return ForwardCache(preActivation: preActivation, hidden: hidden, mean: mean, value: value)
    }
}

private func adamUpdate(
    param: inout Float,
    grad: Float,
    m: inout Float,
    v: inout Float,
    config: AdamConfig,
    biasCorrection1: Float,
    biasCorrection2: Float
) -> Float {
    m = config.beta1 * m + (1.0 - config.beta1) * grad
    v = config.beta2 * v + (1.0 - config.beta2) * grad * grad
    let mHat = m / biasCorrection1
    let vHat = v / biasCorrection2
    let delta = config.learningRate * mHat / (sqrt(vHat) + config.epsilon)
    param -= delta
    return delta
}

private struct ForwardCache {
    let preActivation: [Float]
    let hidden: [Float]
    let mean: [Float]
    let value: Float
}

func makePPOBatch(storage: VectorRolloutStorage, estimates: AdvantageEstimates) throws -> PPOBatch {
    guard let oldLogProbs = storage.logProbs else {
        throw EnvProjectError.validationFailed(message: "PPOBatch requires rollout storage with old log-probabilities.")
    }

    return try PPOBatch(
        sampleCount: storage.horizon * storage.envCount,
        observationDim: storage.observationDim,
        actionDim: storage.actionDim,
        observations: storage.observations,
        actions: storage.actions,
        oldLogProbs: oldLogProbs,
        advantages: estimates.advantages,
        returns: estimates.returns
    )
}
