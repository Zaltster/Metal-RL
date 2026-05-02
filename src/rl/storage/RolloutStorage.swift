import Foundation

struct VectorRolloutStorage {
    let horizon: Int
    let envCount: Int
    let observationDim: Int
    let actionDim: Int

    let observations: [Float]
    let actions: [Float]
    let rewards: [Float]
    let dones: [UInt32]
    let resetCounts: [UInt32]
    let nextObservations: [Float]
    let values: [Float]?
    let finalValues: [Float]?
    let logProbs: [Float]?

    init(
        horizon: Int,
        envCount: Int,
        observationDim: Int,
        actionDim: Int,
        observations: [Float],
        actions: [Float],
        rewards: [Float],
        dones: [UInt32],
        resetCounts: [UInt32],
        nextObservations: [Float],
        values: [Float]? = nil,
        finalValues: [Float]? = nil,
        logProbs: [Float]? = nil
    ) throws {
        let expectedObservationCount = horizon * envCount * observationDim
        let expectedActionCount = horizon * envCount * actionDim
        let expectedScalarCount = horizon * envCount

        if observations.count != expectedObservationCount {
            throw EnvProjectError.validationFailed(
                message: "RolloutStorage observation size mismatch: expected \(expectedObservationCount), got \(observations.count)."
            )
        }
        if actions.count != expectedActionCount {
            throw EnvProjectError.validationFailed(
                message: "RolloutStorage action size mismatch: expected \(expectedActionCount), got \(actions.count)."
            )
        }
        if rewards.count != expectedScalarCount {
            throw EnvProjectError.validationFailed(
                message: "RolloutStorage reward size mismatch: expected \(expectedScalarCount), got \(rewards.count)."
            )
        }
        if dones.count != expectedScalarCount {
            throw EnvProjectError.validationFailed(
                message: "RolloutStorage done size mismatch: expected \(expectedScalarCount), got \(dones.count)."
            )
        }
        if resetCounts.count != expectedScalarCount {
            throw EnvProjectError.validationFailed(
                message: "RolloutStorage reset-count size mismatch: expected \(expectedScalarCount), got \(resetCounts.count)."
            )
        }
        if nextObservations.count != expectedObservationCount {
            throw EnvProjectError.validationFailed(
                message: "RolloutStorage next-observation size mismatch: expected \(expectedObservationCount), got \(nextObservations.count)."
            )
        }
        if let values, values.count != expectedScalarCount {
            throw EnvProjectError.validationFailed(
                message: "RolloutStorage value size mismatch: expected \(expectedScalarCount), got \(values.count)."
            )
        }
        if let finalValues, finalValues.count != envCount {
            throw EnvProjectError.validationFailed(
                message: "RolloutStorage final-value size mismatch: expected \(envCount), got \(finalValues.count)."
            )
        }
        if let logProbs, logProbs.count != expectedScalarCount {
            throw EnvProjectError.validationFailed(
                message: "RolloutStorage log-prob size mismatch: expected \(expectedScalarCount), got \(logProbs.count)."
            )
        }

        self.horizon = horizon
        self.envCount = envCount
        self.observationDim = observationDim
        self.actionDim = actionDim
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.dones = dones
        self.resetCounts = resetCounts
        self.nextObservations = nextObservations
        self.values = values
        self.finalValues = finalValues
        self.logProbs = logProbs
    }

    init(
        rollout: VectorRollout,
        envCount: Int,
        observationSpec: VectorObservationSpec,
        actionSpec: VectorActionSpec
    ) throws {
        let horizon = rollout.steps.count
        let observationDim = observationSpec.elementsPerEnv
        let actionDim = actionSpec.dimensionsPerEnv

        var observations: [Float] = []
        var actions: [Float] = []
        var rewards: [Float] = []
        var dones: [UInt32] = []
        var resetCounts: [UInt32] = []
        var nextObservations: [Float] = []

        observations.reserveCapacity(horizon * envCount * observationDim)
        actions.reserveCapacity(horizon * envCount * actionDim)
        rewards.reserveCapacity(horizon * envCount)
        dones.reserveCapacity(horizon * envCount)
        resetCounts.reserveCapacity(horizon * envCount)
        nextObservations.reserveCapacity(horizon * envCount * observationDim)

        for step in rollout.steps {
            observations.append(contentsOf: step.observationsBefore)
            actions.append(contentsOf: step.actions)
            rewards.append(contentsOf: step.rewards)
            dones.append(contentsOf: step.dones)
            resetCounts.append(contentsOf: step.resetCounts)
            nextObservations.append(contentsOf: step.observationsAfterReset)
        }

        try self.init(
            horizon: horizon,
            envCount: envCount,
            observationDim: observationDim,
            actionDim: actionDim,
            observations: observations,
            actions: actions,
            rewards: rewards,
            dones: dones,
            resetCounts: resetCounts,
            nextObservations: nextObservations
        )
    }

    func observation(step: Int, env: Int) -> [Float] {
        let base = (step * envCount + env) * observationDim
        return Array(observations[base..<(base + observationDim)])
    }

    func action(step: Int, env: Int) -> [Float] {
        let base = (step * envCount + env) * actionDim
        return Array(actions[base..<(base + actionDim)])
    }

    func reward(step: Int, env: Int) -> Float {
        rewards[step * envCount + env]
    }

    func done(step: Int, env: Int) -> UInt32 {
        dones[step * envCount + env]
    }

    func resetCount(step: Int, env: Int) -> UInt32 {
        resetCounts[step * envCount + env]
    }

    func nextObservation(step: Int, env: Int) -> [Float] {
        let base = (step * envCount + env) * observationDim
        return Array(nextObservations[base..<(base + observationDim)])
    }

    func value(step: Int, env: Int) throws -> Float {
        guard let values else {
            throw EnvProjectError.validationFailed(message: "RolloutStorage does not contain value predictions.")
        }
        return values[step * envCount + env]
    }

    func finalValue(env: Int) throws -> Float {
        guard let finalValues else {
            throw EnvProjectError.validationFailed(message: "RolloutStorage does not contain final value predictions.")
        }
        return finalValues[env]
    }

    func logProb(step: Int, env: Int) throws -> Float {
        guard let logProbs else {
            throw EnvProjectError.validationFailed(message: "RolloutStorage does not contain action log-probabilities.")
        }
        return logProbs[step * envCount + env]
    }
}

func makeRolloutStorage(
    from rollout: VectorRollout,
    driver: MetalVectorEnvDriver
) throws -> VectorRolloutStorage {
    try VectorRolloutStorage(
        rollout: rollout,
        envCount: driver.envCount,
        observationSpec: driver.observationSpec,
        actionSpec: driver.actionSpec
    )
}

func collectRandomPolicyRolloutStorage(
    driver: MetalVectorEnvDriver,
    config: RandomPolicyConfig
) throws -> VectorRolloutStorage {
    try makeRolloutStorage(from: collectRandomPolicyRollout(driver: driver, config: config), driver: driver)
}

func collectPolicyRolloutStorage(
    driver: MetalVectorEnvDriver,
    policy: VectorPolicy,
    config: PolicyRolloutConfig
) throws -> VectorRolloutStorage {
    try makeRolloutStorage(from: collectPolicyRollout(driver: driver, policy: policy, config: config), driver: driver)
}

func collectActorCriticRolloutStorage(
    driver: MetalVectorEnvDriver,
    policy: VectorGaussianActorCriticPolicy,
    config: PolicyRolloutConfig
) throws -> VectorRolloutStorage {
    let initialBatch = try driver.reset()
    var currentBatch = initialBatch

    let horizon = config.horizon
    let envCount = driver.envCount
    let observationDim = driver.observationSpec.elementsPerEnv
    let actionDim = driver.actionSpec.dimensionsPerEnv

    var observations: [Float] = []
    var actions: [Float] = []
    var rewards: [Float] = []
    var dones: [UInt32] = []
    var resetCounts: [UInt32] = []
    var nextObservations: [Float] = []
    var values: [Float] = []
    var logProbs: [Float] = []

    observations.reserveCapacity(horizon * envCount * observationDim)
    actions.reserveCapacity(horizon * envCount * actionDim)
    rewards.reserveCapacity(horizon * envCount)
    dones.reserveCapacity(horizon * envCount)
    resetCounts.reserveCapacity(horizon * envCount)
    nextObservations.reserveCapacity(horizon * envCount * observationDim)
    values.reserveCapacity(horizon * envCount)
    logProbs.reserveCapacity(horizon * envCount)

    for stepIndex in 0..<horizon {
        let evaluation: GaussianPolicyOutputs
        switch config.samplingMode {
        case .deterministicMean:
            evaluation = try policy.evaluateGaussian(
                for: currentBatch.observations,
                taking: nil,
                envCount: envCount,
                observationSpec: driver.observationSpec,
                actionSpec: driver.actionSpec
            )
        case .stochasticGaussian:
            let meanEvaluation = try policy.evaluateGaussian(
                for: currentBatch.observations,
                taking: nil,
                envCount: envCount,
                observationSpec: driver.observationSpec,
                actionSpec: driver.actionSpec
            )
            let sampledActions = try sampleGaussianPolicyActions(
                actionMeans: meanEvaluation.actionMeans,
                logStd: meanEvaluation.logStd,
                actionSpec: driver.actionSpec,
                envCount: envCount,
                stepIndex: stepIndex,
                samplingSeed: config.samplingSeed
            )
            evaluation = try policy.evaluateGaussian(
                for: currentBatch.observations,
                taking: sampledActions,
                envCount: envCount,
                observationSpec: driver.observationSpec,
                actionSpec: driver.actionSpec
            )
        }

        let steppedBatch = try driver.step(actions: evaluation.actions)
        let postResetBatch = try driver.resetDone()

        observations.append(contentsOf: currentBatch.observations)
        actions.append(contentsOf: evaluation.actions)
        rewards.append(contentsOf: steppedBatch.rewards)
        dones.append(contentsOf: steppedBatch.dones)
        resetCounts.append(contentsOf: postResetBatch.resetCounts)
        nextObservations.append(contentsOf: postResetBatch.observations)
        values.append(contentsOf: evaluation.values)
        logProbs.append(contentsOf: evaluation.logProbs)

        currentBatch = postResetBatch
    }

    let finalValues = try policy.values(
        for: currentBatch.observations,
        envCount: envCount,
        observationSpec: driver.observationSpec,
        actionSpec: driver.actionSpec
    )

    return try VectorRolloutStorage(
        horizon: horizon,
        envCount: envCount,
        observationDim: observationDim,
        actionDim: actionDim,
        observations: observations,
        actions: actions,
        rewards: rewards,
        dones: dones,
        resetCounts: resetCounts,
        nextObservations: nextObservations,
        values: values,
        finalValues: finalValues,
        logProbs: logProbs
    )
}
