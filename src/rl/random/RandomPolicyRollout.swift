import Foundation

struct RandomPolicyConfig {
    let horizon: Int
    let actionSeed: UInt32
}

struct VectorRolloutStep {
    let observationsBefore: [Float]
    let actions: [Float]
    let rewards: [Float]
    let dones: [UInt32]
    let observationsAfterReset: [Float]
    let resetCounts: [UInt32]
}

struct VectorRollout {
    let initialBatch: VectorEnvBatch
    let steps: [VectorRolloutStep]
}

private func randomPolicyMixBits(_ value: UInt32) -> UInt32 {
    var x = value &+ 0x9E37_79B9
    x = (x ^ (x >> 16)) &* 0x85EB_CA6B
    x = (x ^ (x >> 13)) &* 0xC2B2_AE35
    return x ^ (x >> 16)
}

private func randomPolicyUniform01(_ value: UInt32) -> Float {
    Float(value & 0x00FF_FFFF) / 16_777_215.0
}

func sampleRandomPolicyActions(
    actionSpec: VectorActionSpec,
    envCount: Int,
    stepIndex: Int,
    actionSeed: UInt32
) -> [Float] {
    let width = actionSpec.maxValue - actionSpec.minValue
    let total = envCount * actionSpec.dimensionsPerEnv

    return (0..<total).map { flatIndex in
        let stepBits = UInt32(truncatingIfNeeded: stepIndex)
        let flatBits = UInt32(truncatingIfNeeded: flatIndex)
        let hash = randomPolicyMixBits(
            actionSeed ^
            (stepBits &* 0x27D4_EB2D) ^
            (flatBits &* 0x1656_67B1)
        )
        return actionSpec.minValue + randomPolicyUniform01(hash) * width
    }
}

func collectRandomPolicyRollout(
    driver: MetalVectorEnvDriver,
    config: RandomPolicyConfig
) throws -> VectorRollout {
    let initialBatch = try driver.reset()
    var currentBatch = initialBatch
    var steps: [VectorRolloutStep] = []
    steps.reserveCapacity(config.horizon)

    for stepIndex in 0..<config.horizon {
        let actions = sampleRandomPolicyActions(
            actionSpec: driver.actionSpec,
            envCount: driver.envCount,
            stepIndex: stepIndex,
            actionSeed: config.actionSeed
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
