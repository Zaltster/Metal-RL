import Foundation

struct PolicyRolloutConfig {
    let horizon: Int
}

func collectPolicyRollout(
    driver: MetalVectorEnvDriver,
    policy: VectorPolicy,
    config: PolicyRolloutConfig
) throws -> VectorRollout {
    let initialBatch = try driver.reset()
    var currentBatch = initialBatch
    var steps: [VectorRolloutStep] = []
    steps.reserveCapacity(config.horizon)

    for _ in 0..<config.horizon {
        let actions = try policy.actions(
            for: currentBatch.observations,
            envCount: driver.envCount,
            observationSpec: driver.observationSpec,
            actionSpec: driver.actionSpec
        )

        let steppedBatch = try driver.step(actions: actions)
        let postResetBatch = try driver.resetDone()

        steps.append(
            VectorRolloutStep(
                observationsBefore: currentBatch.observations,
                actions: actions,
                rewards: steppedBatch.rewards,
                dones: steppedBatch.dones,
                observationsAfterReset: postResetBatch.observations,
                resetCounts: postResetBatch.resetCounts
            )
        )

        currentBatch = postResetBatch
    }

    return VectorRollout(initialBatch: initialBatch, steps: steps)
}
