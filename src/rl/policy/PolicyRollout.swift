import Foundation

enum PolicySamplingMode: String {
    case deterministicMean = "deterministic-mean"
    case stochasticGaussian = "stochastic-gaussian"

    static func parse(_ rawValue: String) throws -> PolicySamplingMode {
        guard let mode = PolicySamplingMode(rawValue: rawValue) else {
            throw EnvProjectError.validationFailed(
                message: "Unsupported policy sampling mode '\(rawValue)'. Use deterministic-mean or stochastic-gaussian."
            )
        }
        return mode
    }
}

struct PolicyRolloutConfig {
    let horizon: Int
    let samplingMode: PolicySamplingMode
    let samplingSeed: UInt32

    init(
        horizon: Int,
        samplingMode: PolicySamplingMode = .deterministicMean,
        samplingSeed: UInt32 = 0xC0FF_EE11
    ) {
        self.horizon = horizon
        self.samplingMode = samplingMode
        self.samplingSeed = samplingSeed
    }
}

private func policySamplingMixBits(_ value: UInt32) -> UInt32 {
    var x = value &+ 0x9E37_79B9
    x = (x ^ (x >> 16)) &* 0x85EB_CA6B
    x = (x ^ (x >> 13)) &* 0xC2B2_AE35
    return x ^ (x >> 16)
}

private func policySamplingUniform01(_ value: UInt32) -> Float {
    Float(value & 0x00FF_FFFF) / 16_777_215.0
}

func sampleGaussianPolicyActions(
    actionMeans: [Float],
    logStd: [Float],
    actionSpec: VectorActionSpec,
    envCount: Int,
    stepIndex: Int,
    samplingSeed: UInt32
) throws -> [Float] {
    let actionDim = actionSpec.dimensionsPerEnv
    let total = envCount * actionDim

    if actionMeans.count != total {
        throw EnvProjectError.validationFailed(
            message: "Gaussian policy sampling mean size mismatch: expected \(total), got \(actionMeans.count)."
        )
    }
    if logStd.count != actionDim {
        throw EnvProjectError.validationFailed(
            message: "Gaussian policy sampling logStd size mismatch: expected \(actionDim), got \(logStd.count)."
        )
    }

    var sampled = Array(repeating: Float.zero, count: total)
    let stepBits = UInt32(truncatingIfNeeded: stepIndex)
    let twoPi = Float(2.0 * Double.pi)

    for flatIndex in 0..<total {
        let flatBits = UInt32(truncatingIfNeeded: flatIndex)
        let u1Bits = policySamplingMixBits(
            samplingSeed ^
            (stepBits &* 0x27D4_EB2D) ^
            (flatBits &* 0x1656_67B1)
        )
        let u2Bits = policySamplingMixBits(
            samplingSeed ^
            0xA511_E9B3 ^
            (stepBits &* 0x9E37_79B9) ^
            (flatBits &* 0x85EB_CA6B)
        )
        let u1 = max(policySamplingUniform01(u1Bits), 1.0e-7)
        let u2 = policySamplingUniform01(u2Bits)
        let normal = sqrt(-2.0 * log(u1)) * cos(twoPi * u2)
        let actionIndex = flatIndex % actionDim
        let rawAction = actionMeans[flatIndex] + exp(logStd[actionIndex]) * normal
        sampled[flatIndex] = min(max(rawAction, actionSpec.minValue), actionSpec.maxValue)
    }

    return sampled
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
        let actions: [Float]
        switch config.samplingMode {
        case .deterministicMean:
            actions = try policy.actions(
                for: currentBatch.observations,
                envCount: driver.envCount,
                observationSpec: driver.observationSpec,
                actionSpec: driver.actionSpec
            )
        case .stochasticGaussian:
            guard let gaussianPolicy = policy as? VectorGaussianActorCriticPolicy else {
                throw EnvProjectError.validationFailed(
                    message: "stochastic-gaussian policy sampling requires a Gaussian actor-critic policy."
                )
            }
            let evaluation = try gaussianPolicy.evaluateGaussian(
                for: currentBatch.observations,
                taking: nil,
                envCount: driver.envCount,
                observationSpec: driver.observationSpec,
                actionSpec: driver.actionSpec
            )
            actions = try sampleGaussianPolicyActions(
                actionMeans: evaluation.actionMeans,
                logStd: evaluation.logStd,
                actionSpec: driver.actionSpec,
                envCount: driver.envCount,
                stepIndex: steps.count,
                samplingSeed: config.samplingSeed
            )
        }

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
