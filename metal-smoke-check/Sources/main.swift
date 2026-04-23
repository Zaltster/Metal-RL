import Foundation
import Metal

func compareRolloutActions(lhs: VectorRollout, rhs: VectorRollout, tolerance: Float) throws {
    if lhs.steps.count != rhs.steps.count {
        throw EnvProjectError.validationFailed(
            message: "Random-policy rollout length mismatch: expected \(lhs.steps.count), got \(rhs.steps.count)."
        )
    }

    for stepIndex in lhs.steps.indices {
        let left = lhs.steps[stepIndex]
        let right = rhs.steps[stepIndex]

        if left.actions.count != right.actions.count {
            throw EnvProjectError.validationFailed(
                message: "Random-policy action width mismatch at step \(stepIndex): expected \(left.actions.count), got \(right.actions.count)."
            )
        }

        for actionIndex in left.actions.indices {
            let expected = left.actions[actionIndex]
            let actual = right.actions[actionIndex]
            if abs(expected - actual) > tolerance {
                throw EnvProjectError.validationFailed(
                    message: "Random-policy action mismatch at step \(stepIndex), index \(actionIndex): expected \(expected), got \(actual)."
                )
            }
        }
    }
}

func compareRolloutBatches(lhs: VectorRollout, rhs: VectorRollout, tolerance: Float) throws {
    func compareFloatArrays(_ name: String, _ lhs: [Float], _ rhs: [Float], stepIndex: Int?) throws {
        if lhs.count != rhs.count {
            throw EnvProjectError.validationFailed(
                message: "\(name) length mismatch\(stepIndex.map { " at step \($0)" } ?? ""): expected \(lhs.count), got \(rhs.count)."
            )
        }
        for index in lhs.indices {
            if abs(lhs[index] - rhs[index]) > tolerance {
                throw EnvProjectError.validationFailed(
                    message: "\(name) mismatch\(stepIndex.map { " at step \($0)" } ?? "") index \(index): expected \(lhs[index]), got \(rhs[index])."
                )
            }
        }
    }

    func compareUIntArrays(_ name: String, _ lhs: [UInt32], _ rhs: [UInt32], stepIndex: Int?) throws {
        if lhs.count != rhs.count {
            throw EnvProjectError.validationFailed(
                message: "\(name) length mismatch\(stepIndex.map { " at step \($0)" } ?? ""): expected \(lhs.count), got \(rhs.count)."
            )
        }
        for index in lhs.indices {
            if lhs[index] != rhs[index] {
                throw EnvProjectError.validationFailed(
                    message: "\(name) mismatch\(stepIndex.map { " at step \($0)" } ?? "") index \(index): expected \(lhs[index]), got \(rhs[index])."
                )
            }
        }
    }

    try compareFloatArrays("initial observations", lhs.initialBatch.observations, rhs.initialBatch.observations, stepIndex: nil)
    try compareFloatArrays("initial rewards", lhs.initialBatch.rewards, rhs.initialBatch.rewards, stepIndex: nil)
    try compareUIntArrays("initial dones", lhs.initialBatch.dones, rhs.initialBatch.dones, stepIndex: nil)
    try compareUIntArrays("initial resetCounts", lhs.initialBatch.resetCounts, rhs.initialBatch.resetCounts, stepIndex: nil)

    if lhs.steps.count != rhs.steps.count {
        throw EnvProjectError.validationFailed(
            message: "Random-policy rollout step-count mismatch: expected \(lhs.steps.count), got \(rhs.steps.count)."
        )
    }

    for stepIndex in lhs.steps.indices {
        let left = lhs.steps[stepIndex]
        let right = rhs.steps[stepIndex]
        try compareFloatArrays("observationsBefore", left.observationsBefore, right.observationsBefore, stepIndex: stepIndex)
        try compareFloatArrays("actions", left.actions, right.actions, stepIndex: stepIndex)
        try compareFloatArrays("rewards", left.rewards, right.rewards, stepIndex: stepIndex)
        try compareUIntArrays("dones", left.dones, right.dones, stepIndex: stepIndex)
        try compareFloatArrays("observationsAfterReset", left.observationsAfterReset, right.observationsAfterReset, stepIndex: stepIndex)
        try compareUIntArrays("resetCounts", left.resetCounts, right.resetCounts, stepIndex: stepIndex)
    }
}

func ensureDifferentActionSeedChangesRollout(reference: VectorRollout, alternate: VectorRollout, tolerance: Float) throws {
    for stepIndex in reference.steps.indices {
        let left = reference.steps[stepIndex]
        let right = alternate.steps[stepIndex]
        for actionIndex in left.actions.indices {
            if abs(left.actions[actionIndex] - right.actions[actionIndex]) > tolerance {
                return
            }
        }
    }

    throw EnvProjectError.validationFailed(message: "Different random-policy action seed produced no observable action change.")
}

func validateActionBounds(rollout: VectorRollout, actionSpec: VectorActionSpec, tolerance: Float) throws {
    for stepIndex in rollout.steps.indices {
        for actionIndex in rollout.steps[stepIndex].actions.indices {
            let action = rollout.steps[stepIndex].actions[actionIndex]
            if action < actionSpec.minValue - tolerance || action > actionSpec.maxValue + tolerance {
                throw EnvProjectError.validationFailed(
                    message: "Action bound violation at step \(stepIndex), index \(actionIndex): value \(action) outside [\(actionSpec.minValue), \(actionSpec.maxValue)]."
                )
            }
        }
    }
}

func ensureDifferentPoliciesChangeRollout(reference: VectorRollout, alternate: VectorRollout, tolerance: Float) throws {
    for stepIndex in reference.steps.indices {
        let left = reference.steps[stepIndex]
        let right = alternate.steps[stepIndex]
        for actionIndex in left.actions.indices {
            if abs(left.actions[actionIndex] - right.actions[actionIndex]) > tolerance {
                return
            }
        }
    }

    throw EnvProjectError.validationFailed(message: "Alternate linear policy produced no observable action change.")
}

func compareActionArrays(lhs: [Float], rhs: [Float], tolerance: Float, context: String) throws {
    if lhs.count != rhs.count {
        throw EnvProjectError.validationFailed(
            message: "\(context) action length mismatch: expected \(lhs.count), got \(rhs.count)."
        )
    }

    for index in lhs.indices {
        if abs(lhs[index] - rhs[index]) > tolerance {
            throw EnvProjectError.validationFailed(
                message: "\(context) action mismatch at index \(index): expected \(lhs[index]), got \(rhs[index])."
            )
        }
    }
}

func compareStorage(lhs: VectorRolloutStorage, rhs: VectorRolloutStorage, tolerance: Float, context: String) throws {
    if lhs.horizon != rhs.horizon || lhs.envCount != rhs.envCount || lhs.observationDim != rhs.observationDim || lhs.actionDim != rhs.actionDim {
        throw EnvProjectError.validationFailed(
            message: "\(context) storage shape mismatch."
        )
    }

    try compareActionArrays(lhs: lhs.observations, rhs: rhs.observations, tolerance: tolerance, context: "\(context) observations")
    try compareActionArrays(lhs: lhs.actions, rhs: rhs.actions, tolerance: tolerance, context: "\(context) actions")
    try compareActionArrays(lhs: lhs.rewards, rhs: rhs.rewards, tolerance: tolerance, context: "\(context) rewards")
    try compareActionArrays(lhs: lhs.nextObservations, rhs: rhs.nextObservations, tolerance: tolerance, context: "\(context) nextObservations")

    if lhs.dones != rhs.dones {
        throw EnvProjectError.validationFailed(
            message: "\(context) dones mismatch."
        )
    }
    if lhs.resetCounts != rhs.resetCounts {
        throw EnvProjectError.validationFailed(
            message: "\(context) resetCounts mismatch."
        )
    }
    switch (lhs.values, rhs.values) {
    case let (.some(left), .some(right)):
        try compareActionArrays(lhs: left, rhs: right, tolerance: tolerance, context: "\(context) values")
    case (nil, nil):
        break
    default:
        throw EnvProjectError.validationFailed(message: "\(context) values presence mismatch.")
    }
    switch (lhs.finalValues, rhs.finalValues) {
    case let (.some(left), .some(right)):
        try compareActionArrays(lhs: left, rhs: right, tolerance: tolerance, context: "\(context) finalValues")
    case (nil, nil):
        break
    default:
        throw EnvProjectError.validationFailed(message: "\(context) finalValues presence mismatch.")
    }
    switch (lhs.logProbs, rhs.logProbs) {
    case let (.some(left), .some(right)):
        try compareActionArrays(lhs: left, rhs: right, tolerance: tolerance, context: "\(context) logProbs")
    case (nil, nil):
        break
    default:
        throw EnvProjectError.validationFailed(message: "\(context) logProbs presence mismatch.")
    }
}

func ensureDifferentStorageValuesOrActions(reference: VectorRolloutStorage, alternate: VectorRolloutStorage, tolerance: Float, context: String) throws {
    for index in reference.actions.indices {
        if abs(reference.actions[index] - alternate.actions[index]) > tolerance {
            return
        }
    }

    if let referenceValues = reference.values, let alternateValues = alternate.values {
        for index in referenceValues.indices {
            if abs(referenceValues[index] - alternateValues[index]) > tolerance {
                return
            }
        }
    }

    throw EnvProjectError.validationFailed(message: "\(context) produced no observable action or value change.")
}

func validateGAESyntheticCase(tolerance: Float) throws {
    let storage = try VectorRolloutStorage(
        horizon: 3,
        envCount: 2,
        observationDim: 1,
        actionDim: 1,
        observations: [0, 0, 0, 0, 0, 0],
        actions: [0, 0, 0, 0, 0, 0],
        rewards: [
            1.0, 0.5,
            2.0, 1.0,
            3.0, 1.5,
        ],
        dones: [
            0, 1,
            0, 0,
            1, 0,
        ],
        resetCounts: [0, 0, 0, 0, 0, 0],
        nextObservations: [0, 0, 0, 0, 0, 0],
        values: [
            0.5, 0.2,
            0.6, 0.3,
            0.7, 0.4,
        ],
        finalValues: [0.0, 0.9]
    )
    let estimates = try computeGAE(storage: storage, config: GAEConfig(gamma: 0.9, lambda: 0.8))

    let expectedAdvantages: [Float] = [
        3.69392, 0.3,
        3.68600, 2.4352,
        2.3,     1.91,
    ]
    let expectedReturns: [Float] = [
        4.19392, 0.5,
        4.28600, 2.7352,
        3.0,     2.31,
    ]

    try compareActionArrays(
        lhs: estimates.advantages,
        rhs: expectedAdvantages,
        tolerance: tolerance,
        context: "GAE synthetic advantages"
    )
    try compareActionArrays(
        lhs: estimates.returns,
        rhs: expectedReturns,
        tolerance: tolerance,
        context: "GAE synthetic returns"
    )
}

func validateGAEConsistency(
    estimates: AdvantageEstimates,
    storage: VectorRolloutStorage,
    tolerance: Float,
    context: String
) throws {
    guard let values = storage.values else {
        throw EnvProjectError.validationFailed(message: "\(context) storage is missing values.")
    }

    if estimates.horizon != storage.horizon || estimates.envCount != storage.envCount {
        throw EnvProjectError.validationFailed(message: "\(context) shape mismatch between GAE output and storage.")
    }

    for step in 0..<storage.horizon {
        for env in 0..<storage.envCount {
            let index = step * storage.envCount + env
            let advantage = estimates.advantages[index]
            let returnValue = estimates.returns[index]
            let value = values[index]

            if !advantage.isFinite || !returnValue.isFinite {
                throw EnvProjectError.validationFailed(
                    message: "\(context) produced non-finite GAE outputs at step \(step), env \(env)."
                )
            }

            if abs((advantage + value) - returnValue) > tolerance {
                throw EnvProjectError.validationFailed(
                    message: "\(context) return mismatch at step \(step), env \(env): expected \(advantage + value), got \(returnValue)."
                )
            }
        }
    }
}

func comparePPOLosses(lhs: PPOLossBreakdown, rhs: PPOLossBreakdown, tolerance: Float, context: String) throws {
    if lhs.sampleCount != rhs.sampleCount {
        throw EnvProjectError.validationFailed(message: "\(context) sample-count mismatch: expected \(lhs.sampleCount), got \(rhs.sampleCount).")
    }

    func check(_ name: String, _ expected: Float, _ actual: Float) throws {
        if abs(expected - actual) > tolerance {
            throw EnvProjectError.validationFailed(
                message: "\(context) \(name) mismatch: expected \(expected), got \(actual)."
            )
        }
    }

    try check("policyLoss", lhs.policyLoss, rhs.policyLoss)
    try check("valueLoss", lhs.valueLoss, rhs.valueLoss)
    try check("entropyBonus", lhs.entropyBonus, rhs.entropyBonus)
    try check("totalLoss", lhs.totalLoss, rhs.totalLoss)
    try check("meanRatio", lhs.meanRatio, rhs.meanRatio)
}

func compareTrainingRuns(lhs: CPUTrainingRunSummary, rhs: CPUTrainingRunSummary, tolerance: Float, context: String) throws {
    if lhs.iterations.count != rhs.iterations.count {
        throw EnvProjectError.validationFailed(
            message: "\(context) training iteration-count mismatch: expected \(lhs.iterations.count), got \(rhs.iterations.count)."
        )
    }

    func check(_ name: String, _ expected: Float, _ actual: Float, iteration: Int?) throws {
        if abs(expected - actual) > tolerance {
            throw EnvProjectError.validationFailed(
                message: "\(context) \(name) mismatch\(iteration.map { " at iteration \($0)" } ?? ""): expected \(expected), got \(actual)."
            )
        }
    }

    try check("totalParameterDeltaL1", lhs.totalParameterDeltaL1, rhs.totalParameterDeltaL1, iteration: nil)

    for index in lhs.iterations.indices {
        let left = lhs.iterations[index]
        let right = rhs.iterations[index]

        if left.iteration != right.iteration || left.doneCount != right.doneCount {
            throw EnvProjectError.validationFailed(message: "\(context) discrete training summary mismatch at iteration \(index).")
        }

        try check("meanReward", left.meanReward, right.meanReward, iteration: index)
        try check("parameterDeltaL1", left.parameterDeltaL1, right.parameterDeltaL1, iteration: index)
        try comparePPOLosses(lhs: left.preLoss, rhs: right.preLoss, tolerance: tolerance, context: "\(context) preLoss iteration \(index)")
        try comparePPOLosses(lhs: left.postLoss, rhs: right.postLoss, tolerance: tolerance, context: "\(context) postLoss iteration \(index)")
    }
}

func ensureDifferentTrainingRuns(lhs: CPUTrainingRunSummary, rhs: CPUTrainingRunSummary, tolerance: Float, context: String) throws {
    if abs(lhs.totalParameterDeltaL1 - rhs.totalParameterDeltaL1) > tolerance {
        return
    }

    for index in lhs.iterations.indices {
        let left = lhs.iterations[index]
        let right = rhs.iterations[index]

        if abs(left.meanReward - right.meanReward) > tolerance ||
            abs(left.postLoss.totalLoss - right.postLoss.totalLoss) > tolerance ||
            left.doneCount != right.doneCount {
            return
        }
    }

    throw EnvProjectError.validationFailed(message: "\(context) produced no observable change.")
}

func validateTrainingRun(summary: CPUTrainingRunSummary, tolerance: Float, context: String) throws {
    if summary.iterations.isEmpty {
        throw EnvProjectError.validationFailed(message: "\(context) produced no iteration summaries.")
    }
    if summary.totalParameterDeltaL1 <= tolerance {
        throw EnvProjectError.validationFailed(message: "\(context) produced no parameter movement.")
    }

    for iteration in summary.iterations {
        if !iteration.meanReward.isFinite ||
            !iteration.preLoss.totalLoss.isFinite ||
            !iteration.postLoss.totalLoss.isFinite {
            throw EnvProjectError.validationFailed(message: "\(context) produced non-finite training metrics at iteration \(iteration.iteration).")
        }
        if iteration.parameterDeltaL1 <= tolerance {
            throw EnvProjectError.validationFailed(message: "\(context) produced no update at iteration \(iteration.iteration).")
        }
    }
}

func validatePPOSyntheticCase(tolerance: Float) throws {
    let config = PPOConfig(clipEpsilon: 0.2, valueCoefficient: 0.5, entropyCoefficient: 0.01)
    let result = try computePPOLoss(
        oldLogProbs: [0.0, 0.0],
        newLogProbs: [Float(log(1.1)), Float(log(0.6))],
        advantages: [1.0, -1.0],
        returns: [1.5, -0.2],
        newValues: [1.4, -0.1],
        entropies: [0.3, 0.3],
        config: config
    )

    let expected = PPOLossBreakdown(
        sampleCount: 2,
        policyLoss: -0.15,
        valueLoss: 0.005,
        entropyBonus: 0.3,
        totalLoss: -0.1505,
        meanRatio: 0.85
    )
    try comparePPOLosses(lhs: expected, rhs: result, tolerance: tolerance, context: "PPO synthetic case")
}

func validateSyntheticBackwardStep(
    basePolicy: MLPPolicy,
    observationSpec: VectorObservationSpec,
    actionSpec: VectorActionSpec,
    ppoConfig: PPOConfig,
    tolerance: Float
) throws {
    var model = TrainableMLPActorCritic(policy: basePolicy)
    let observations: [Float] = [
        0.20, -0.10, 0.05, 0.30,
        -0.15, 0.25, -0.20, 0.10,
    ]
    let initialEval = try model.evaluateGaussian(
        for: observations,
        taking: nil,
        envCount: 2,
        observationSpec: observationSpec,
        actionSpec: actionSpec
    )
    let actions: [Float] = [
        initialEval.actionMeans[0] + 0.18,
        initialEval.actionMeans[1] - 0.16,
    ]
    let oldEval = try model.evaluateGaussian(
        for: observations,
        taking: actions,
        envCount: 2,
        observationSpec: observationSpec,
        actionSpec: actionSpec
    )
    let batch = try PPOBatch(
        sampleCount: 2,
        observationDim: observationSpec.elementsPerEnv,
        actionDim: actionSpec.dimensionsPerEnv,
        observations: observations,
        actions: actions,
        oldLogProbs: oldEval.logProbs,
        advantages: [1.0, -0.8],
        returns: [oldEval.values[0] + 0.6, oldEval.values[1] - 0.4]
    )

    let summary = try model.applySGDStep(
        batch: batch,
        observationSpec: observationSpec,
        actionSpec: actionSpec,
        ppoConfig: ppoConfig,
        sgdConfig: SGDConfig(learningRate: 0.02)
    )

    if summary.parameterDeltaL1 <= tolerance {
        throw EnvProjectError.validationFailed(message: "Synthetic backward step produced no parameter update.")
    }
    if summary.postLoss.totalLoss >= summary.preLoss.totalLoss {
        throw EnvProjectError.validationFailed(
            message: "Synthetic backward step did not reduce total loss: before \(summary.preLoss.totalLoss), after \(summary.postLoss.totalLoss)."
        )
    }
}

func validateSyntheticAdamStep(
    basePolicy: MLPPolicy,
    observationSpec: VectorObservationSpec,
    actionSpec: VectorActionSpec,
    ppoConfig: PPOConfig,
    tolerance: Float
) throws {
    var model = TrainableMLPActorCritic(policy: basePolicy)
    let observations: [Float] = [
        -0.22, 0.14, -0.08, 0.19,
        0.11, -0.18, 0.26, -0.07,
    ]
    let initialEval = try model.evaluateGaussian(
        for: observations,
        taking: nil,
        envCount: 2,
        observationSpec: observationSpec,
        actionSpec: actionSpec
    )
    let actions: [Float] = [
        initialEval.actionMeans[0] - 0.12,
        initialEval.actionMeans[1] + 0.15,
    ]
    let oldEval = try model.evaluateGaussian(
        for: observations,
        taking: actions,
        envCount: 2,
        observationSpec: observationSpec,
        actionSpec: actionSpec
    )
    let batch = try PPOBatch(
        sampleCount: 2,
        observationDim: observationSpec.elementsPerEnv,
        actionDim: actionSpec.dimensionsPerEnv,
        observations: observations,
        actions: actions,
        oldLogProbs: oldEval.logProbs,
        advantages: [0.9, -0.7],
        returns: [oldEval.values[0] + 0.5, oldEval.values[1] - 0.3]
    )
    var adamState = AdamState(model: model)
    let summary = try model.applyAdamStep(
        batch: batch,
        observationSpec: observationSpec,
        actionSpec: actionSpec,
        ppoConfig: ppoConfig,
        adamState: &adamState,
        adamConfig: AdamConfig(learningRate: 0.01, beta1: 0.9, beta2: 0.999, epsilon: 1e-8)
    )

    if summary.parameterDeltaL1 <= tolerance {
        throw EnvProjectError.validationFailed(message: "Synthetic Adam step produced no parameter update.")
    }
    if summary.postLoss.totalLoss >= summary.preLoss.totalLoss {
        throw EnvProjectError.validationFailed(
            message: "Synthetic Adam step did not reduce total loss: before \(summary.preLoss.totalLoss), after \(summary.postLoss.totalLoss)."
        )
    }
}

func validateRealBackwardStep(
    basePolicy: MLPPolicy,
    storage: VectorRolloutStorage,
    estimates: AdvantageEstimates,
    observationSpec: VectorObservationSpec,
    actionSpec: VectorActionSpec,
    ppoConfig: PPOConfig,
    tolerance: Float
) throws {
    var model = TrainableMLPActorCritic(policy: basePolicy)
    let batch = try makePPOBatch(storage: storage, estimates: estimates)
    var firstSummary: TrainStepSummary?
    var finalSummary: TrainStepSummary?

    for _ in 0..<5 {
        let summary = try model.applySGDStep(
            batch: batch,
            observationSpec: observationSpec,
            actionSpec: actionSpec,
            ppoConfig: ppoConfig,
            sgdConfig: SGDConfig(learningRate: 0.001)
        )
        if firstSummary == nil {
            firstSummary = summary
        }
        finalSummary = summary
    }

    guard let firstSummary, let finalSummary else {
        throw EnvProjectError.validationFailed(message: "Real backward validation did not execute any SGD steps.")
    }

    if firstSummary.parameterDeltaL1 <= tolerance {
        throw EnvProjectError.validationFailed(message: "Real backward step produced no parameter update.")
    }
    if finalSummary.postLoss.totalLoss >= firstSummary.preLoss.totalLoss {
        throw EnvProjectError.validationFailed(
            message: "Real backward steps did not reduce total loss: before \(firstSummary.preLoss.totalLoss), after \(finalSummary.postLoss.totalLoss)."
        )
    }
}

func validateStorageAgainstRollout(storage: VectorRolloutStorage, rollout: VectorRollout, tolerance: Float, context: String) throws {
    if storage.horizon != rollout.steps.count {
        throw EnvProjectError.validationFailed(
            message: "\(context) storage horizon mismatch: expected \(rollout.steps.count), got \(storage.horizon)."
        )
    }

    for step in 0..<storage.horizon {
        let rolloutStep = rollout.steps[step]
        for env in 0..<storage.envCount {
            let obsBase = env * storage.observationDim
            let actionBase = env * storage.actionDim

            let expectedObservation = Array(rolloutStep.observationsBefore[obsBase..<(obsBase + storage.observationDim)])
            try compareActionArrays(
                lhs: storage.observation(step: step, env: env),
                rhs: expectedObservation,
                tolerance: tolerance,
                context: "\(context) observation step \(step) env \(env)"
            )

            let expectedAction = Array(rolloutStep.actions[actionBase..<(actionBase + storage.actionDim)])
            try compareActionArrays(
                lhs: storage.action(step: step, env: env),
                rhs: expectedAction,
                tolerance: tolerance,
                context: "\(context) action step \(step) env \(env)"
            )

            let expectedNextObservation = Array(rolloutStep.observationsAfterReset[obsBase..<(obsBase + storage.observationDim)])
            try compareActionArrays(
                lhs: storage.nextObservation(step: step, env: env),
                rhs: expectedNextObservation,
                tolerance: tolerance,
                context: "\(context) nextObservation step \(step) env \(env)"
            )

            let expectedReward = rolloutStep.rewards[env]
            if abs(storage.reward(step: step, env: env) - expectedReward) > tolerance {
                throw EnvProjectError.validationFailed(
                    message: "\(context) reward mismatch at step \(step), env \(env): expected \(expectedReward), got \(storage.reward(step: step, env: env))."
                )
            }

            let expectedDone = rolloutStep.dones[env]
            if storage.done(step: step, env: env) != expectedDone {
                throw EnvProjectError.validationFailed(
                    message: "\(context) done mismatch at step \(step), env \(env): expected \(expectedDone), got \(storage.done(step: step, env: env))."
                )
            }

            let expectedResetCount = rolloutStep.resetCounts[env]
            if storage.resetCount(step: step, env: env) != expectedResetCount {
                throw EnvProjectError.validationFailed(
                    message: "\(context) resetCount mismatch at step \(step), env \(env): expected \(expectedResetCount), got \(storage.resetCount(step: step, env: env))."
                )
            }
        }
    }
}

func runValidatedRollout(
    driver: CartPoleVectorEnvDriver,
    cartPoleParams: CartPoleParams,
    resetSeed: UInt32,
    initialStates: [CartPoleState],
    horizon: Int,
    tolerance: Float
) throws -> CartPoleRolloutResult {
    driver.setResetSeed(resetSeed)
    _ = try driver.reset()

    let count = initialStates.count
    var cpuStates = initialStates
    var cpuResetCounts = Array(repeating: UInt32(0), count: count)

    for step in 0..<horizon {
        let actions = makeCartPoleActions(step: step, count: count)
        let expectedStepped = zip(cpuStates, actions).map { state, action in
            stepCartPoleReference(state: state, action: action, params: cartPoleParams)
        }

        let steppedInterface = try driver.step(actions: actions)
        let gpuStepped = driver.debugReadStates()

        for index in 0..<count {
            try validateCartPoleState(
                actual: gpuStepped[index],
                expected: expectedStepped[index],
                phase: "step",
                step: step,
                index: index,
                tolerance: tolerance
            )
        }
        try validateCartPoleOutputs(
            observations: steppedInterface.observations,
            rewards: steppedInterface.rewards,
            dones: steppedInterface.dones,
            expectedStates: expectedStepped,
            phase: "step-outputs",
            step: step,
            tolerance: tolerance
        )

        var expectedPostReset = expectedStepped
        for index in 0..<count {
            expectedPostReset[index] = resetDoneReference(
                state: expectedStepped[index],
                lane: UInt32(index),
                resetCount: &cpuResetCounts[index],
                params: ResetParams(envCount: UInt32(count), baseSeed: resetSeed)
            )
        }

        let resetInterface = try driver.resetDone()
        let gpuPostReset = driver.debugReadStates()
        let gpuResetCounts = resetInterface.resetCounts

        for index in 0..<count {
            try validateCartPoleState(
                actual: gpuPostReset[index],
                expected: expectedPostReset[index],
                phase: "reset",
                step: step,
                index: index,
                tolerance: tolerance
            )

            if gpuResetCounts[index] != cpuResetCounts[index] {
                throw EnvProjectError.validationFailed(
                    message: "Reset-count validation failed at step \(step), lane \(index): expected \(cpuResetCounts[index]), got \(gpuResetCounts[index])."
                )
            }
        }
        try validateCartPoleOutputs(
            observations: resetInterface.observations,
            rewards: resetInterface.rewards,
            dones: resetInterface.dones,
            expectedStates: expectedPostReset,
            phase: "reset-outputs",
            step: step,
            tolerance: tolerance
        )

        cpuStates = expectedPostReset
    }

    let finalStates = driver.debugReadStates()
    let finalResetCounts = driver.debugReadResetCounts()
    let totalResets = finalResetCounts.reduce(0) { $0 + Int($1) }

    if totalResets == 0 {
        throw EnvProjectError.validationFailed(message: "The rollout never triggered a reset, so reset semantics were not actually exercised.")
    }

    let sampleLines = (0..<4).map { index in
        let state = finalStates[index]
        return String(
            format: "env %d final(x=% .5f xDot=% .5f theta=% .5f thetaDot=% .5f reward=%.1f done=%u resets=%u)",
            index,
            state.x,
            state.xDot,
            state.theta,
            state.thetaDot,
            state.reward,
            state.done,
            finalResetCounts[index]
        )
    }

    return CartPoleRolloutResult(
        finalStates: finalStates,
        resetCounts: finalResetCounts,
        totalResets: totalResets,
        sampleLines: sampleLines
    )
}

func compareReplay(lhs: CartPoleRolloutResult, rhs: CartPoleRolloutResult, tolerance: Float) throws {
    for index in lhs.finalStates.indices {
        let left = lhs.finalStates[index]
        let right = rhs.finalStates[index]

        func checkFloat(_ field: String, _ expected: Float, _ actual: Float) throws {
            if abs(expected - actual) > tolerance {
                throw EnvProjectError.validationFailed(
                    message: "Same-seed replay mismatch at lane \(index), field \(field): expected \(expected), got \(actual)."
                )
            }
        }

        try checkFloat("x", left.x, right.x)
        try checkFloat("xDot", left.xDot, right.xDot)
        try checkFloat("theta", left.theta, right.theta)
        try checkFloat("thetaDot", left.thetaDot, right.thetaDot)
        try checkFloat("reward", left.reward, right.reward)

        if left.done != right.done {
            throw EnvProjectError.validationFailed(
                message: "Same-seed replay mismatch at lane \(index), field done: expected \(left.done), got \(right.done)."
            )
        }

        if lhs.resetCounts[index] != rhs.resetCounts[index] {
            throw EnvProjectError.validationFailed(
                message: "Same-seed replay mismatch at lane \(index), field resetCount: expected \(lhs.resetCounts[index]), got \(rhs.resetCounts[index])."
            )
        }
    }
}

func ensureDifferentSeedChangesResetState(reference: CartPoleRolloutResult, alternate: CartPoleRolloutResult, tolerance: Float) throws {
    for index in reference.finalStates.indices where reference.resetCounts[index] > 0 || alternate.resetCounts[index] > 0 {
        let left = reference.finalStates[index]
        let right = alternate.finalStates[index]

        if abs(left.x - right.x) > tolerance ||
            abs(left.xDot - right.xDot) > tolerance ||
            abs(left.theta - right.theta) > tolerance ||
            abs(left.thetaDot - right.thetaDot) > tolerance ||
            reference.resetCounts[index] != alternate.resetCounts[index] {
            return
        }
    }

    throw EnvProjectError.validationFailed(message: "Different seeded rollout produced no observable change in reset states.")
}

func runValidationHarness() throws {
    guard let device = MTLCreateSystemDefaultDevice() else {
        throw EnvProjectError.noMetalDevice
    }

    try checkLayout(CartPoleState.self, name: "CartPoleState", expectedSize: 24, expectedStride: 24, expectedAlignment: 4)
    try checkLayout(CartPoleParams.self, name: "CartPoleParams", expectedSize: 36, expectedStride: 36, expectedAlignment: 4)
    try checkLayout(ResetParams.self, name: "ResetParams", expectedSize: 8, expectedStride: 8, expectedAlignment: 4)

    let env = ProcessInfo.processInfo.environment
    let rootDir = env["METAL_SMOKE_ROOT"] ?? FileManager.default.currentDirectoryPath
    let count = 256
    let horizon = 24
    let tolerance: Float = 1e-5
    let replayTolerance: Float = 0.0
    let resetSeed: UInt32 = 0x1234_5678
    let actionSeed: UInt32 = 0x89AB_CDEF
    let cartPoleParams = CartPoleParams(
        envCount: UInt32(count),
        dt: 0.02,
        gravity: 9.8,
        massCart: 1.0,
        massPole: 0.1,
        halfPoleLength: 0.5,
        forceMag: 10.0,
        xThreshold: 2.4,
        thetaThresholdRadians: 12.0 * .pi / 180.0
    )
    let initialStates = makeCartPoleInitialStates(count: count)
    let gaeConfig = GAEConfig(gamma: 0.99, lambda: 0.95)
    let ppoConfig = PPOConfig(clipEpsilon: 0.2, valueCoefficient: 0.5, entropyCoefficient: 0.01)

    try validateGAESyntheticCase(tolerance: 1e-4)
    try validatePPOSyntheticCase(tolerance: 1e-4)

    let driver = try CartPoleVectorEnvDriver(
        device: device,
        rootDir: rootDir,
        envCount: count,
        cartPoleParams: cartPoleParams,
        resetSeed: resetSeed,
        initialStates: initialStates
    )
    let linearPolicy = try makeReferenceLinearPolicy(
        observationSpec: driver.observationSpec,
        actionSpec: driver.actionSpec
    )
    let alternateLinearPolicy = try makeAlternateLinearPolicy(
        observationSpec: driver.observationSpec,
        actionSpec: driver.actionSpec
    )
    let metalLinearPolicy = try makeReferenceMetalLinearPolicy(
        device: device,
        rootDir: rootDir,
        envCount: count,
        observationSpec: driver.observationSpec,
        actionSpec: driver.actionSpec
    )
    let alternateMetalLinearPolicy = try makeAlternateMetalLinearPolicy(
        device: device,
        rootDir: rootDir,
        envCount: count,
        observationSpec: driver.observationSpec,
        actionSpec: driver.actionSpec
    )
    let mlpPolicy = try makeReferenceMLPPolicy(
        observationSpec: driver.observationSpec,
        actionSpec: driver.actionSpec
    )
    let alternateMLPPolicy = try makeAlternateMLPPolicy(
        observationSpec: driver.observationSpec,
        actionSpec: driver.actionSpec
    )
    let metalMLPPolicy = try makeReferenceMetalMLPPolicy(
        device: device,
        rootDir: rootDir,
        envCount: count,
        observationSpec: driver.observationSpec,
        actionSpec: driver.actionSpec
    )
    let alternateMetalMLPPolicy = try makeAlternateMetalMLPPolicy(
        device: device,
        rootDir: rootDir,
        envCount: count,
        observationSpec: driver.observationSpec,
        actionSpec: driver.actionSpec
    )
    try validateSyntheticBackwardStep(
        basePolicy: mlpPolicy,
        observationSpec: driver.observationSpec,
        actionSpec: driver.actionSpec,
        ppoConfig: ppoConfig,
        tolerance: tolerance
    )
    try validateSyntheticAdamStep(
        basePolicy: mlpPolicy,
        observationSpec: driver.observationSpec,
        actionSpec: driver.actionSpec,
        ppoConfig: ppoConfig,
        tolerance: tolerance
    )

    let baseline = try runValidatedRollout(
        driver: driver,
        cartPoleParams: cartPoleParams,
        resetSeed: resetSeed,
        initialStates: initialStates,
        horizon: horizon,
        tolerance: tolerance
    )

    let replay = try runValidatedRollout(
        driver: driver,
        cartPoleParams: cartPoleParams,
        resetSeed: resetSeed,
        initialStates: initialStates,
        horizon: horizon,
        tolerance: tolerance
    )
    try compareReplay(lhs: baseline, rhs: replay, tolerance: replayTolerance)

    let alternateSeedRun = try runValidatedRollout(
        driver: driver,
        cartPoleParams: cartPoleParams,
        resetSeed: resetSeed &+ 1,
        initialStates: initialStates,
        horizon: horizon,
        tolerance: tolerance
    )
    try ensureDifferentSeedChangesResetState(reference: baseline, alternate: alternateSeedRun, tolerance: tolerance)

    driver.setResetSeed(resetSeed)
    let randomPolicyBaseline = try collectRandomPolicyRollout(
        driver: driver,
        config: RandomPolicyConfig(horizon: 8, actionSeed: actionSeed)
    )

    driver.setResetSeed(resetSeed)
    let randomPolicyReplay = try collectRandomPolicyRollout(
        driver: driver,
        config: RandomPolicyConfig(horizon: 8, actionSeed: actionSeed)
    )
    try compareRolloutActions(lhs: randomPolicyBaseline, rhs: randomPolicyReplay, tolerance: replayTolerance)
    try compareRolloutBatches(lhs: randomPolicyBaseline, rhs: randomPolicyReplay, tolerance: replayTolerance)

    driver.setResetSeed(resetSeed)
    let randomPolicyAlternate = try collectRandomPolicyRollout(
        driver: driver,
        config: RandomPolicyConfig(horizon: 8, actionSeed: actionSeed &+ 1)
    )
    try ensureDifferentActionSeedChangesRollout(reference: randomPolicyBaseline, alternate: randomPolicyAlternate, tolerance: replayTolerance)
    try validateActionBounds(rollout: randomPolicyBaseline, actionSpec: driver.actionSpec, tolerance: tolerance)

    let randomPolicyStorage = try makeRolloutStorage(from: randomPolicyBaseline, driver: driver)
    try validateStorageAgainstRollout(storage: randomPolicyStorage, rollout: randomPolicyBaseline, tolerance: tolerance, context: "random-policy storage")
    let randomPolicyStoredReplay = try collectRandomPolicyRolloutStorage(
        driver: driver,
        config: RandomPolicyConfig(horizon: 8, actionSeed: actionSeed)
    )
    try compareStorage(lhs: randomPolicyStorage, rhs: randomPolicyStoredReplay, tolerance: replayTolerance, context: "random-policy storage replay")

    let policyProbeBatch = try driver.reset()
    let cpuLinearProbeActions = try linearPolicy.actions(
        for: policyProbeBatch.observations,
        envCount: driver.envCount,
        observationSpec: driver.observationSpec,
        actionSpec: driver.actionSpec
    )
    let gpuLinearProbeActions = try metalLinearPolicy.actions(
        for: policyProbeBatch.observations,
        envCount: driver.envCount,
        observationSpec: driver.observationSpec,
        actionSpec: driver.actionSpec
    )
    try compareActionArrays(
        lhs: cpuLinearProbeActions,
        rhs: gpuLinearProbeActions,
        tolerance: tolerance,
        context: "CPU-vs-GPU linear-policy probe"
    )
    let cpuMLPProbeActions = try mlpPolicy.actions(
        for: policyProbeBatch.observations,
        envCount: driver.envCount,
        observationSpec: driver.observationSpec,
        actionSpec: driver.actionSpec
    )
    let gpuMLPProbeActions = try metalMLPPolicy.actions(
        for: policyProbeBatch.observations,
        envCount: driver.envCount,
        observationSpec: driver.observationSpec,
        actionSpec: driver.actionSpec
    )
    try compareActionArrays(
        lhs: cpuMLPProbeActions,
        rhs: gpuMLPProbeActions,
        tolerance: tolerance,
        context: "CPU-vs-GPU MLP-policy probe"
    )
    let cpuMLPProbeValues = try mlpPolicy.values(
        for: policyProbeBatch.observations,
        envCount: driver.envCount,
        observationSpec: driver.observationSpec,
        actionSpec: driver.actionSpec
    )
    let gpuMLPProbeValues = try metalMLPPolicy.values(
        for: policyProbeBatch.observations,
        envCount: driver.envCount,
        observationSpec: driver.observationSpec,
        actionSpec: driver.actionSpec
    )
    try compareActionArrays(
        lhs: cpuMLPProbeValues,
        rhs: gpuMLPProbeValues,
        tolerance: tolerance,
        context: "CPU-vs-GPU MLP-value probe"
    )

    driver.setResetSeed(resetSeed)
    let linearPolicyBaseline = try collectPolicyRollout(
        driver: driver,
        policy: linearPolicy,
        config: PolicyRolloutConfig(horizon: 8)
    )

    driver.setResetSeed(resetSeed)
    let linearPolicyReplay = try collectPolicyRollout(
        driver: driver,
        policy: linearPolicy,
        config: PolicyRolloutConfig(horizon: 8)
    )
    try compareRolloutActions(lhs: linearPolicyBaseline, rhs: linearPolicyReplay, tolerance: replayTolerance)
    try compareRolloutBatches(lhs: linearPolicyBaseline, rhs: linearPolicyReplay, tolerance: replayTolerance)
    try validateActionBounds(rollout: linearPolicyBaseline, actionSpec: driver.actionSpec, tolerance: tolerance)

    driver.setResetSeed(resetSeed)
    let linearPolicyAlternate = try collectPolicyRollout(
        driver: driver,
        policy: alternateLinearPolicy,
        config: PolicyRolloutConfig(horizon: 8)
    )
    try ensureDifferentPoliciesChangeRollout(reference: linearPolicyBaseline, alternate: linearPolicyAlternate, tolerance: replayTolerance)

    let linearPolicyStorage = try makeRolloutStorage(from: linearPolicyBaseline, driver: driver)
    try validateStorageAgainstRollout(storage: linearPolicyStorage, rollout: linearPolicyBaseline, tolerance: tolerance, context: "cpu linear-policy storage")
    let linearPolicyStoredReplay = try collectPolicyRolloutStorage(
        driver: driver,
        policy: linearPolicy,
        config: PolicyRolloutConfig(horizon: 8)
    )
    try compareStorage(lhs: linearPolicyStorage, rhs: linearPolicyStoredReplay, tolerance: replayTolerance, context: "cpu linear-policy storage replay")

    driver.setResetSeed(resetSeed)
    let metalLinearPolicyBaseline = try collectPolicyRollout(
        driver: driver,
        policy: metalLinearPolicy,
        config: PolicyRolloutConfig(horizon: 8)
    )

    driver.setResetSeed(resetSeed)
    let metalLinearPolicyReplay = try collectPolicyRollout(
        driver: driver,
        policy: metalLinearPolicy,
        config: PolicyRolloutConfig(horizon: 8)
    )
    try compareRolloutActions(lhs: metalLinearPolicyBaseline, rhs: metalLinearPolicyReplay, tolerance: replayTolerance)
    try compareRolloutBatches(lhs: metalLinearPolicyBaseline, rhs: metalLinearPolicyReplay, tolerance: replayTolerance)
    try validateActionBounds(rollout: metalLinearPolicyBaseline, actionSpec: driver.actionSpec, tolerance: tolerance)
    try compareRolloutActions(lhs: linearPolicyBaseline, rhs: metalLinearPolicyBaseline, tolerance: tolerance)
    try compareRolloutBatches(lhs: linearPolicyBaseline, rhs: metalLinearPolicyBaseline, tolerance: tolerance)

    driver.setResetSeed(resetSeed)
    let metalLinearPolicyAlternate = try collectPolicyRollout(
        driver: driver,
        policy: alternateMetalLinearPolicy,
        config: PolicyRolloutConfig(horizon: 8)
    )
    try ensureDifferentPoliciesChangeRollout(reference: metalLinearPolicyBaseline, alternate: metalLinearPolicyAlternate, tolerance: replayTolerance)

    let metalLinearPolicyStorage = try makeRolloutStorage(from: metalLinearPolicyBaseline, driver: driver)
    try validateStorageAgainstRollout(storage: metalLinearPolicyStorage, rollout: metalLinearPolicyBaseline, tolerance: tolerance, context: "gpu linear-policy storage")
    let metalLinearPolicyStoredReplay = try collectPolicyRolloutStorage(
        driver: driver,
        policy: metalLinearPolicy,
        config: PolicyRolloutConfig(horizon: 8)
    )
    try compareStorage(lhs: metalLinearPolicyStorage, rhs: metalLinearPolicyStoredReplay, tolerance: replayTolerance, context: "gpu linear-policy storage replay")

    driver.setResetSeed(resetSeed)
    let mlpPolicyBaseline = try collectPolicyRollout(
        driver: driver,
        policy: mlpPolicy,
        config: PolicyRolloutConfig(horizon: 8)
    )

    driver.setResetSeed(resetSeed)
    let mlpPolicyReplay = try collectPolicyRollout(
        driver: driver,
        policy: mlpPolicy,
        config: PolicyRolloutConfig(horizon: 8)
    )
    try compareRolloutActions(lhs: mlpPolicyBaseline, rhs: mlpPolicyReplay, tolerance: replayTolerance)
    try compareRolloutBatches(lhs: mlpPolicyBaseline, rhs: mlpPolicyReplay, tolerance: replayTolerance)
    try validateActionBounds(rollout: mlpPolicyBaseline, actionSpec: driver.actionSpec, tolerance: tolerance)

    driver.setResetSeed(resetSeed)
    let mlpPolicyAlternate = try collectPolicyRollout(
        driver: driver,
        policy: alternateMLPPolicy,
        config: PolicyRolloutConfig(horizon: 8)
    )
    try ensureDifferentPoliciesChangeRollout(reference: mlpPolicyBaseline, alternate: mlpPolicyAlternate, tolerance: replayTolerance)

    driver.setResetSeed(resetSeed)
    let metalMLPPolicyBaseline = try collectPolicyRollout(
        driver: driver,
        policy: metalMLPPolicy,
        config: PolicyRolloutConfig(horizon: 8)
    )

    driver.setResetSeed(resetSeed)
    let metalMLPPolicyReplay = try collectPolicyRollout(
        driver: driver,
        policy: metalMLPPolicy,
        config: PolicyRolloutConfig(horizon: 8)
    )
    try compareRolloutActions(lhs: metalMLPPolicyBaseline, rhs: metalMLPPolicyReplay, tolerance: replayTolerance)
    try compareRolloutBatches(lhs: metalMLPPolicyBaseline, rhs: metalMLPPolicyReplay, tolerance: replayTolerance)
    try validateActionBounds(rollout: metalMLPPolicyBaseline, actionSpec: driver.actionSpec, tolerance: tolerance)
    try compareRolloutActions(lhs: mlpPolicyBaseline, rhs: metalMLPPolicyBaseline, tolerance: tolerance)
    try compareRolloutBatches(lhs: mlpPolicyBaseline, rhs: metalMLPPolicyBaseline, tolerance: tolerance)

    driver.setResetSeed(resetSeed)
    let metalMLPPolicyAlternate = try collectPolicyRollout(
        driver: driver,
        policy: alternateMetalMLPPolicy,
        config: PolicyRolloutConfig(horizon: 8)
    )
    try ensureDifferentPoliciesChangeRollout(reference: metalMLPPolicyBaseline, alternate: metalMLPPolicyAlternate, tolerance: replayTolerance)

    let mlpPolicyStorage = try makeRolloutStorage(from: mlpPolicyBaseline, driver: driver)
    try validateStorageAgainstRollout(storage: mlpPolicyStorage, rollout: mlpPolicyBaseline, tolerance: tolerance, context: "cpu mlp-policy storage")
    let mlpPolicyStoredReplay = try collectPolicyRolloutStorage(
        driver: driver,
        policy: mlpPolicy,
        config: PolicyRolloutConfig(horizon: 8)
    )
    try compareStorage(lhs: mlpPolicyStorage, rhs: mlpPolicyStoredReplay, tolerance: replayTolerance, context: "cpu mlp-policy storage replay")

    driver.setResetSeed(resetSeed)
    let mlpActorCriticStorage = try collectActorCriticRolloutStorage(
        driver: driver,
        policy: mlpPolicy,
        config: PolicyRolloutConfig(horizon: 8)
    )
    driver.setResetSeed(resetSeed)
    let mlpActorCriticReplay = try collectActorCriticRolloutStorage(
        driver: driver,
        policy: mlpPolicy,
        config: PolicyRolloutConfig(horizon: 8)
    )
    try compareStorage(lhs: mlpActorCriticStorage, rhs: mlpActorCriticReplay, tolerance: replayTolerance, context: "cpu actor-critic storage replay")

    let metalMLPPolicyStorage = try makeRolloutStorage(from: metalMLPPolicyBaseline, driver: driver)
    try validateStorageAgainstRollout(storage: metalMLPPolicyStorage, rollout: metalMLPPolicyBaseline, tolerance: tolerance, context: "gpu mlp-policy storage")
    let metalMLPPolicyStoredReplay = try collectPolicyRolloutStorage(
        driver: driver,
        policy: metalMLPPolicy,
        config: PolicyRolloutConfig(horizon: 8)
    )
    try compareStorage(lhs: metalMLPPolicyStorage, rhs: metalMLPPolicyStoredReplay, tolerance: replayTolerance, context: "gpu mlp-policy storage replay")

    driver.setResetSeed(resetSeed)
    let metalMLPActorCriticStorage = try collectActorCriticRolloutStorage(
        driver: driver,
        policy: metalMLPPolicy,
        config: PolicyRolloutConfig(horizon: 8)
    )
    driver.setResetSeed(resetSeed)
    let metalMLPActorCriticReplay = try collectActorCriticRolloutStorage(
        driver: driver,
        policy: metalMLPPolicy,
        config: PolicyRolloutConfig(horizon: 8)
    )
    try compareStorage(lhs: metalMLPActorCriticStorage, rhs: metalMLPActorCriticReplay, tolerance: replayTolerance, context: "gpu actor-critic storage replay")
    try compareStorage(lhs: mlpActorCriticStorage, rhs: metalMLPActorCriticStorage, tolerance: tolerance, context: "cpu-vs-gpu actor-critic storage")

    driver.setResetSeed(resetSeed)
    let alternateMetalMLPActorCriticStorage = try collectActorCriticRolloutStorage(
        driver: driver,
        policy: alternateMetalMLPPolicy,
        config: PolicyRolloutConfig(horizon: 8)
    )
    try ensureDifferentStorageValuesOrActions(
        reference: metalMLPActorCriticStorage,
        alternate: alternateMetalMLPActorCriticStorage,
        tolerance: replayTolerance,
        context: "alternate metal mlp actor-critic storage"
    )

    let cpuGAE = try computeGAE(storage: mlpActorCriticStorage, config: gaeConfig)
    let cpuGAEReplay = try computeGAE(storage: mlpActorCriticReplay, config: gaeConfig)
    try compareActionArrays(lhs: cpuGAE.advantages, rhs: cpuGAEReplay.advantages, tolerance: replayTolerance, context: "cpu GAE replay advantages")
    try compareActionArrays(lhs: cpuGAE.returns, rhs: cpuGAEReplay.returns, tolerance: replayTolerance, context: "cpu GAE replay returns")
    try validateGAEConsistency(estimates: cpuGAE, storage: mlpActorCriticStorage, tolerance: tolerance, context: "cpu GAE")

    let gpuGAE = try computeGAE(storage: metalMLPActorCriticStorage, config: gaeConfig)
    let gpuGAEReplay = try computeGAE(storage: metalMLPActorCriticReplay, config: gaeConfig)
    try compareActionArrays(lhs: gpuGAE.advantages, rhs: gpuGAEReplay.advantages, tolerance: replayTolerance, context: "gpu GAE replay advantages")
    try compareActionArrays(lhs: gpuGAE.returns, rhs: gpuGAEReplay.returns, tolerance: replayTolerance, context: "gpu GAE replay returns")
    try validateGAEConsistency(estimates: gpuGAE, storage: metalMLPActorCriticStorage, tolerance: tolerance, context: "gpu GAE")
    try compareActionArrays(lhs: cpuGAE.advantages, rhs: gpuGAE.advantages, tolerance: tolerance, context: "cpu-vs-gpu GAE advantages")
    try compareActionArrays(lhs: cpuGAE.returns, rhs: gpuGAE.returns, tolerance: tolerance, context: "cpu-vs-gpu GAE returns")

    let cpuPPOLoss = try computePPOLoss(
        storage: mlpActorCriticStorage,
        estimates: cpuGAE,
        policy: mlpPolicy,
        observationSpec: driver.observationSpec,
        actionSpec: driver.actionSpec,
        config: ppoConfig
    )
    let cpuPPOLossReplay = try computePPOLoss(
        storage: mlpActorCriticReplay,
        estimates: cpuGAEReplay,
        policy: mlpPolicy,
        observationSpec: driver.observationSpec,
        actionSpec: driver.actionSpec,
        config: ppoConfig
    )
    try comparePPOLosses(lhs: cpuPPOLoss, rhs: cpuPPOLossReplay, tolerance: replayTolerance, context: "cpu PPO replay")
    if abs(cpuPPOLoss.meanRatio - 1.0) > tolerance {
        throw EnvProjectError.validationFailed(message: "cpu PPO same-policy meanRatio mismatch: expected 1.0, got \(cpuPPOLoss.meanRatio).")
    }

    let gpuPPOLoss = try computePPOLoss(
        storage: metalMLPActorCriticStorage,
        estimates: gpuGAE,
        policy: metalMLPPolicy,
        observationSpec: driver.observationSpec,
        actionSpec: driver.actionSpec,
        config: ppoConfig
    )
    let gpuPPOLossReplay = try computePPOLoss(
        storage: metalMLPActorCriticReplay,
        estimates: gpuGAEReplay,
        policy: metalMLPPolicy,
        observationSpec: driver.observationSpec,
        actionSpec: driver.actionSpec,
        config: ppoConfig
    )
    try comparePPOLosses(lhs: gpuPPOLoss, rhs: gpuPPOLossReplay, tolerance: replayTolerance, context: "gpu PPO replay")
    if abs(gpuPPOLoss.meanRatio - 1.0) > tolerance {
        throw EnvProjectError.validationFailed(message: "gpu PPO same-policy meanRatio mismatch: expected 1.0, got \(gpuPPOLoss.meanRatio).")
    }
    try comparePPOLosses(lhs: cpuPPOLoss, rhs: gpuPPOLoss, tolerance: tolerance, context: "cpu-vs-gpu PPO")

    let alternatePPOLoss = try computePPOLoss(
        storage: metalMLPActorCriticStorage,
        estimates: gpuGAE,
        policy: alternateMetalMLPPolicy,
        observationSpec: driver.observationSpec,
        actionSpec: driver.actionSpec,
        config: ppoConfig
    )
    if abs(alternatePPOLoss.totalLoss - gpuPPOLoss.totalLoss) <= replayTolerance {
        throw EnvProjectError.validationFailed(message: "alternate PPO loss produced no observable total-loss change.")
    }

    try validateRealBackwardStep(
        basePolicy: mlpPolicy,
        storage: mlpActorCriticStorage,
        estimates: cpuGAE,
        observationSpec: driver.observationSpec,
        actionSpec: driver.actionSpec,
        ppoConfig: ppoConfig,
        tolerance: tolerance
    )

    let trainingConfig = CPUTrainingLoopConfig(
        iterations: 3,
        rolloutHorizon: 8,
        epochsPerIteration: 2,
        miniBatchSize: 256,
        resetSeed: resetSeed,
        shuffleSeed: 0xA5A5_1357,
        gaeConfig: gaeConfig,
        ppoConfig: ppoConfig,
        optimizer: .adam(AdamConfig(learningRate: 0.001, beta1: 0.9, beta2: 0.999, epsilon: 1e-8))
    )
    var trainingModel = TrainableMLPActorCritic(policy: mlpPolicy)
    let trainingRun = try runCPUTrainingLoop(
        driver: driver,
        model: &trainingModel,
        config: trainingConfig
    )
    try validateTrainingRun(summary: trainingRun, tolerance: tolerance, context: "cpu training loop")

    var trainingReplayModel = TrainableMLPActorCritic(policy: mlpPolicy)
    let trainingReplay = try runCPUTrainingLoop(
        driver: driver,
        model: &trainingReplayModel,
        config: trainingConfig
    )
    try compareTrainingRuns(lhs: trainingRun, rhs: trainingReplay, tolerance: replayTolerance, context: "cpu training replay")

    var alternateTrainingModel = TrainableMLPActorCritic(policy: mlpPolicy)
    let alternateTrainingRun = try runCPUTrainingLoop(
        driver: driver,
        model: &alternateTrainingModel,
        config: CPUTrainingLoopConfig(
            iterations: trainingConfig.iterations,
            rolloutHorizon: trainingConfig.rolloutHorizon,
            epochsPerIteration: trainingConfig.epochsPerIteration,
            miniBatchSize: trainingConfig.miniBatchSize,
            resetSeed: trainingConfig.resetSeed &+ 1,
            shuffleSeed: trainingConfig.shuffleSeed,
            gaeConfig: trainingConfig.gaeConfig,
            ppoConfig: trainingConfig.ppoConfig,
            optimizer: trainingConfig.optimizer
        )
    )
    try ensureDifferentTrainingRuns(lhs: trainingRun, rhs: alternateTrainingRun, tolerance: replayTolerance, context: "alternate training loop seed")

    var hybridCpuModel = TrainableMLPActorCritic(policy: mlpPolicy)
    let hybridGpuPolicy = try makeReferenceMetalMLPPolicy(
        device: device,
        rootDir: rootDir,
        envCount: count,
        observationSpec: driver.observationSpec,
        actionSpec: driver.actionSpec
    )
    let hybridTrainingRun = try runHybridTrainingLoop(
        driver: driver,
        cpuModel: &hybridCpuModel,
        gpuPolicy: hybridGpuPolicy,
        config: trainingConfig
    )
    try validateTrainingRun(summary: hybridTrainingRun, tolerance: tolerance, context: "hybrid gpu-rollout training loop")

    var hybridReplayCpuModel = TrainableMLPActorCritic(policy: mlpPolicy)
    let hybridReplayGpuPolicy = try makeReferenceMetalMLPPolicy(
        device: device,
        rootDir: rootDir,
        envCount: count,
        observationSpec: driver.observationSpec,
        actionSpec: driver.actionSpec
    )
    let hybridTrainingReplay = try runHybridTrainingLoop(
        driver: driver,
        cpuModel: &hybridReplayCpuModel,
        gpuPolicy: hybridReplayGpuPolicy,
        config: trainingConfig
    )
    try compareTrainingRuns(lhs: hybridTrainingRun, rhs: hybridTrainingReplay, tolerance: replayTolerance, context: "hybrid training replay")

    var hybridAlternateCpuModel = TrainableMLPActorCritic(policy: mlpPolicy)
    let hybridAlternateGpuPolicy = try makeReferenceMetalMLPPolicy(
        device: device,
        rootDir: rootDir,
        envCount: count,
        observationSpec: driver.observationSpec,
        actionSpec: driver.actionSpec
    )
    let hybridAlternateRun = try runHybridTrainingLoop(
        driver: driver,
        cpuModel: &hybridAlternateCpuModel,
        gpuPolicy: hybridAlternateGpuPolicy,
        config: CPUTrainingLoopConfig(
            iterations: trainingConfig.iterations,
            rolloutHorizon: trainingConfig.rolloutHorizon,
            epochsPerIteration: trainingConfig.epochsPerIteration,
            miniBatchSize: trainingConfig.miniBatchSize,
            resetSeed: trainingConfig.resetSeed &+ 1,
            shuffleSeed: trainingConfig.shuffleSeed,
            gaeConfig: trainingConfig.gaeConfig,
            ppoConfig: trainingConfig.ppoConfig,
            optimizer: trainingConfig.optimizer
        )
    )
    try ensureDifferentTrainingRuns(lhs: hybridTrainingRun, rhs: hybridAlternateRun, tolerance: replayTolerance, context: "alternate hybrid training loop seed")

    print("CartPole validation harness passed")
    print("device: \(device.name)")
    print("environmentModule: CartPoleMetalEnvironment")
    print("vectorDriver: CartPoleVectorEnvDriver")
    print("shaderPath: src/envs/cartpole/Shaders/cartpole_kernels.metal")
    print("actionSpec: dims=\(driver.actionSpec.dimensionsPerEnv) range=[\(driver.actionSpec.minValue), \(driver.actionSpec.maxValue)]")
    print("observationSpec: elementsPerObservation=\(driver.observationSpec.elementsPerEnv)")
    print("count: \(count)")
    print("horizon: \(horizon)")
    print("CartPoleState layout: size=\(MemoryLayout<CartPoleState>.size) stride=\(MemoryLayout<CartPoleState>.stride) alignment=\(MemoryLayout<CartPoleState>.alignment)")
    print("CartPoleParams layout: size=\(MemoryLayout<CartPoleParams>.size) stride=\(MemoryLayout<CartPoleParams>.stride) alignment=\(MemoryLayout<CartPoleParams>.alignment)")
    print("ResetParams layout: size=\(MemoryLayout<ResetParams>.size) stride=\(MemoryLayout<ResetParams>.stride) alignment=\(MemoryLayout<ResetParams>.alignment)")
    print("validationTolerance: \(tolerance)")
    print("sameSeedReplayTolerance: \(replayTolerance)")
    print("resetSeed: 0x\(String(resetSeed, radix: 16, uppercase: true))")
    print("randomPolicyActionSeed: 0x\(String(actionSeed, radix: 16, uppercase: true))")
    print("totalResets: \(baseline.totalResets)")
    print("same-seed replay matched exactly")
    print("different-seed replay changed reset outcomes")
    print("random-policy same-seed rollout matched exactly")
    print("random-policy different-seed actions changed")
    print("linear-policy replay matched exactly")
    print("alternate linear policy changed actions")
    print("cpu and gpu linear-policy outputs matched")
    print("gpu linear-policy replay matched exactly")
    print("alternate gpu linear policy changed actions")
    print("mlp-policy replay matched exactly")
    print("cpu and gpu mlp-policy outputs matched")
    print("cpu and gpu mlp-value outputs matched")
    print("gpu mlp-policy replay matched exactly")
    print("alternate gpu mlp policy changed actions")
    print("actor-critic storage replay matched exactly")
    print("gae synthetic case matched expected values")
    print("cpu and gpu gae outputs matched")
    print("ppo synthetic case matched expected values")
    print("cpu and gpu ppo losses matched")
    print("manual backward/update step reduced loss")
    print("adam step reduced loss")
    print("cpu training loop replay matched exactly")
    print("hybrid gpu-rollout training loop replay matched exactly")
    print("rollout storage replay matched exactly")
    for line in baseline.sampleLines {
        print(line)
    }
}

do {
    try runValidationHarness()
} catch {
    fputs("error: \(error)\n", stderr)
    exit(1)
}
