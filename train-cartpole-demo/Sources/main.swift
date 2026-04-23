import Foundation
import Metal

func trainingEnvInt(_ name: String, default defaultValue: Int) -> Int {
    if let value = ProcessInfo.processInfo.environment[name], let parsed = Int(value) {
        return parsed
    }
    return defaultValue
}

func trainingEnvUInt32(_ name: String, default defaultValue: UInt32) -> UInt32 {
    if let value = ProcessInfo.processInfo.environment[name] {
        if let parsed = UInt32(value) {
            return parsed
        }
        if value.hasPrefix("0x") || value.hasPrefix("0X") {
            return UInt32(value.dropFirst(2), radix: 16) ?? defaultValue
        }
    }
    return defaultValue
}

func runTrainingDemo() throws {
    guard let device = MTLCreateSystemDefaultDevice() else {
        throw EnvProjectError.noMetalDevice
    }

    let env = ProcessInfo.processInfo.environment
    let rootDir = env["METAL_SMOKE_ROOT"] ?? FileManager.default.currentDirectoryPath
    let envCount = trainingEnvInt("TRAIN_ENV_COUNT", default: 256)
    let iterations = trainingEnvInt("TRAIN_ITERS", default: 8)
    let horizon = trainingEnvInt("TRAIN_HORIZON", default: 16)
    let epochs = trainingEnvInt("TRAIN_EPOCHS", default: 2)
    let miniBatchSize = trainingEnvInt("TRAIN_MINIBATCH", default: 256)
    let resetSeed = trainingEnvUInt32("TRAIN_RESET_SEED", default: 0x1234_5678)
    let shuffleSeed = trainingEnvUInt32("TRAIN_SHUFFLE_SEED", default: 0xA5A5_1357)

    let cartPoleParams = CartPoleParams(
        envCount: UInt32(envCount),
        dt: 0.02,
        gravity: 9.8,
        massCart: 1.0,
        massPole: 0.1,
        halfPoleLength: 0.5,
        forceMag: 10.0,
        xThreshold: 2.4,
        thetaThresholdRadians: 12.0 * .pi / 180.0
    )

    let driver = try CartPoleVectorEnvDriver(
        device: device,
        rootDir: rootDir,
        envCount: envCount,
        cartPoleParams: cartPoleParams,
        resetSeed: resetSeed,
        initialStates: makeCartPoleInitialStates(count: envCount)
    )
    var cpuModel = TrainableMLPActorCritic(
        policy: try makeReferenceMLPPolicy(
            observationSpec: driver.observationSpec,
            actionSpec: driver.actionSpec
        )
    )
    let gpuPolicy = try makeReferenceMetalMLPPolicy(
        device: device,
        rootDir: rootDir,
        envCount: envCount,
        observationSpec: driver.observationSpec,
        actionSpec: driver.actionSpec
    )

    let config = CPUTrainingLoopConfig(
        iterations: iterations,
        rolloutHorizon: horizon,
        epochsPerIteration: epochs,
        miniBatchSize: miniBatchSize,
        resetSeed: resetSeed,
        shuffleSeed: shuffleSeed,
        gaeConfig: GAEConfig(gamma: 0.99, lambda: 0.95),
        ppoConfig: PPOConfig(clipEpsilon: 0.2, valueCoefficient: 0.5, entropyCoefficient: 0.01),
        optimizer: .adam(AdamConfig(learningRate: 0.001, beta1: 0.9, beta2: 0.999, epsilon: 1e-8))
    )

    let summary = try runHybridTrainingLoop(
        driver: driver,
        cpuModel: &cpuModel,
        gpuPolicy: gpuPolicy,
        config: config
    )

    print("CartPole training demo completed")
    print("device: \(device.name)")
    print("envCount: \(envCount)")
    print("iterations: \(iterations)")
    print("rolloutHorizon: \(horizon)")
    print("epochsPerIteration: \(epochs)")
    print("miniBatchSize: \(miniBatchSize)")
    print("resetSeed: 0x\(String(resetSeed, radix: 16, uppercase: true))")
    print("shuffleSeed: 0x\(String(shuffleSeed, radix: 16, uppercase: true))")
    print("initialMeanReward: \(summary.initialMeanReward)")
    print("finalMeanReward: \(summary.finalMeanReward)")
    print("totalParameterDeltaL1: \(summary.totalParameterDeltaL1)")
    for line in makeTrainingSummaryLines(summary: summary) {
        print(line)
    }
}

do {
    try runTrainingDemo()
} catch {
    fputs("error: \(error)\n", stderr)
    exit(1)
}
