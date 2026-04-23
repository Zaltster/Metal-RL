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
        nextObservations: [Float]
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
