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

func trainingEnvString(_ name: String, default defaultValue: String) -> String {
    if let value = ProcessInfo.processInfo.environment[name] {
        return value
    }
    return defaultValue
}

enum CartPoleTrainingBackend: String {
    case persistentGPUAdam = "persistent-gpu-adam"
    case hybridCPUAdam = "hybrid-cpu-adam"

    static func parse(_ rawValue: String) throws -> CartPoleTrainingBackend {
        guard let backend = CartPoleTrainingBackend(rawValue: rawValue) else {
            throw EnvProjectError.validationFailed(
                message: "Unsupported TRAIN_BACKEND '\(rawValue)'. Use persistent-gpu-adam or hybrid-cpu-adam."
            )
        }
        return backend
    }
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
    let logEvery = trainingEnvInt("TRAIN_LOG_EVERY", default: 1)
    let replayHorizon = trainingEnvInt("TRAIN_REPLAY_HORIZON", default: 240)
    let replayEnvIndex = trainingEnvInt("TRAIN_REPLAY_ENV", default: 0)
    let resetSeed = trainingEnvUInt32("TRAIN_RESET_SEED", default: 0x1234_5678)
    let shuffleSeed = trainingEnvUInt32("TRAIN_SHUFFLE_SEED", default: 0xA5A5_1357)
    let policySamplingSeed = trainingEnvUInt32("TRAIN_POLICY_SEED", default: 0xC0FF_EE11)
    let replaySeed = trainingEnvUInt32("TRAIN_REPLAY_SEED", default: resetSeed &+ 0x00C0_FFEE)
    let backend = try CartPoleTrainingBackend.parse(env["TRAIN_BACKEND"] ?? CartPoleTrainingBackend.persistentGPUAdam.rawValue)
    let policySamplingMode = try PolicySamplingMode.parse(
        env["TRAIN_POLICY_SAMPLING"] ?? PolicySamplingMode.deterministicMean.rawValue
    )
    let defaultReplayPath = URL(fileURLWithPath: rootDir)
        .appending(path: "train-cartpole-demo/.build/cartpole_replay.html")
        .path()
    let replayPath = trainingEnvString("TRAIN_REPLAY_PATH", default: defaultReplayPath)

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
    var trainableModel = TrainableMLPActorCritic(
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
        policySamplingMode: policySamplingMode,
        policySamplingSeed: policySamplingSeed,
        epochsPerIteration: epochs,
        miniBatchSize: miniBatchSize,
        resetSeed: resetSeed,
        shuffleSeed: shuffleSeed,
        gaeConfig: GAEConfig(gamma: 0.99, lambda: 0.95),
        ppoConfig: PPOConfig(clipEpsilon: 0.2, valueCoefficient: 0.5, entropyCoefficient: 0.01),
        optimizer: .adam(AdamConfig(learningRate: 0.001, beta1: 0.9, beta2: 0.999, epsilon: 1e-8))
    )

    let progressHandler: ((TrainingProgressSnapshot) -> Void)? = logEvery > 0
        ? { snapshot in
            let completedIteration = snapshot.iteration + 1
            if completedIteration == 1 ||
                completedIteration == snapshot.totalIterations ||
                completedIteration % logEvery == 0 {
                print(makeTrainingProgressLine(snapshot: snapshot))
            }
        }
        : nil

    print("CartPole training demo started")
    print("device: \(device.name)")
    print("backend: \(backend.rawValue)")
    print("policySampling: \(policySamplingMode.rawValue)")
    print("envCount: \(envCount)")
    print("iterations: \(iterations)")
    print("rolloutHorizon: \(horizon)")
    print("epochsPerIteration: \(epochs)")
    print("miniBatchSize: \(miniBatchSize)")
    print("logEvery: \(logEvery)")
    print("replayHorizon: \(replayHorizon)")
    print("replayEnv: \(replayEnvIndex)")
    print("resetSeed: 0x\(String(resetSeed, radix: 16, uppercase: true))")
    print("shuffleSeed: 0x\(String(shuffleSeed, radix: 16, uppercase: true))")
    print("policySeed: 0x\(String(policySamplingSeed, radix: 16, uppercase: true))")
    print("replaySeed: 0x\(String(replaySeed, radix: 16, uppercase: true))")

    let summary: CPUTrainingRunSummary
    switch backend {
    case .persistentGPUAdam:
        summary = try runPersistentMetalTrainingLoop(
            device: device,
            rootDir: rootDir,
            driver: driver,
            model: &trainableModel,
            gpuPolicy: gpuPolicy,
            config: config,
            progressHandler: progressHandler
        )
        try gpuPolicy.load(model: trainableModel)
    case .hybridCPUAdam:
        summary = try runHybridTrainingLoop(
            driver: driver,
            cpuModel: &trainableModel,
            gpuPolicy: gpuPolicy,
            config: config,
            progressHandler: progressHandler
        )
        try gpuPolicy.load(model: trainableModel)
    }

    var checkpointPath: String? = nil
    if let rawPath = env["TRAIN_CHECKPOINT_PATH"], !rawPath.isEmpty {
        let checkpointURL = URL(fileURLWithPath: rawPath)
        let checkpoint = try MLPActorCriticCheckpoint(
            model: trainableModel,
            observationSpec: driver.observationSpec,
            actionSpec: driver.actionSpec
        )
        try saveCheckpoint(checkpoint, to: checkpointURL)
        checkpointPath = checkpointURL.path
    }

    var replayOutputPath: String? = nil
    if !replayPath.isEmpty && replayHorizon > 0 {
        driver.setResetSeed(replaySeed)
        let replayRollout = try collectPolicyRollout(
            driver: driver,
            policy: gpuPolicy,
            config: PolicyRolloutConfig(
                horizon: replayHorizon,
                samplingMode: policySamplingMode,
                samplingSeed: policySamplingSeed &+ 0x51A7_E000
            )
        )
        let frames = try makeCartPoleReplayFrames(
            from: replayRollout,
            envIndex: replayEnvIndex,
            observationSpec: driver.observationSpec
        )
        let replayURL = URL(fileURLWithPath: replayPath)
        try writeCartPoleHTMLReplay(
            frames: frames,
            to: replayURL,
            title: "CartPole Replay",
            dt: cartPoleParams.dt,
            xThreshold: cartPoleParams.xThreshold,
            halfPoleLength: cartPoleParams.halfPoleLength
        )
        replayOutputPath = replayURL.path
    }

    print("CartPole training demo completed")
    print("initialMeanReward: \(summary.initialMeanReward)")
    print("finalMeanReward: \(summary.finalMeanReward)")
    print("totalParameterDeltaL1: \(summary.totalParameterDeltaL1)")
    if let checkpointPath {
        print("checkpointPath: \(checkpointPath)")
    }
    if let replayOutputPath {
        print("replayPath: \(replayOutputPath)")
    }
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
