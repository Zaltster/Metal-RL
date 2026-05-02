import Foundation

struct CPUTrainingLoopConfig {
    let iterations: Int
    let rolloutHorizon: Int
    let policySamplingMode: PolicySamplingMode
    let policySamplingSeed: UInt32
    let epochsPerIteration: Int
    let miniBatchSize: Int
    let resetSeed: UInt32
    let shuffleSeed: UInt32
    let gaeConfig: GAEConfig
    let ppoConfig: PPOConfig
    let optimizer: CPUOptimizerConfig

    init(
        iterations: Int,
        rolloutHorizon: Int,
        policySamplingMode: PolicySamplingMode = .deterministicMean,
        policySamplingSeed: UInt32 = 0xC0FF_EE11,
        epochsPerIteration: Int,
        miniBatchSize: Int,
        resetSeed: UInt32,
        shuffleSeed: UInt32,
        gaeConfig: GAEConfig,
        ppoConfig: PPOConfig,
        optimizer: CPUOptimizerConfig
    ) {
        self.iterations = iterations
        self.rolloutHorizon = rolloutHorizon
        self.policySamplingMode = policySamplingMode
        self.policySamplingSeed = policySamplingSeed
        self.epochsPerIteration = epochsPerIteration
        self.miniBatchSize = miniBatchSize
        self.resetSeed = resetSeed
        self.shuffleSeed = shuffleSeed
        self.gaeConfig = gaeConfig
        self.ppoConfig = ppoConfig
        self.optimizer = optimizer
    }
}

struct CPUTrainingIterationSummary {
    let iteration: Int
    let meanReward: Float
    let doneCount: Int
    let preLoss: PPOLossBreakdown
    let postLoss: PPOLossBreakdown
    let parameterDeltaL1: Float
}

struct CPUTrainingRunSummary {
    let iterations: [CPUTrainingIterationSummary]
    let totalParameterDeltaL1: Float

    var initialMeanReward: Float {
        iterations.first?.meanReward ?? 0.0
    }

    var finalMeanReward: Float {
        iterations.last?.meanReward ?? 0.0
    }
}

struct TrainingProgressSnapshot {
    let iteration: Int
    let totalIterations: Int
    let elapsedSeconds: TimeInterval
    let etaSeconds: TimeInterval
    let completedEnvSteps: Int
    let totalEnvSteps: Int
    let envStepsPerSecond: Double
    let summary: CPUTrainingIterationSummary
}

func runCPUTrainingLoop(
    driver: CartPoleVectorEnvDriver,
    model: inout TrainableMLPActorCritic,
    config: CPUTrainingLoopConfig,
    progressHandler: ((TrainingProgressSnapshot) -> Void)? = nil
) throws -> CPUTrainingRunSummary {
    if config.iterations <= 0 {
        throw EnvProjectError.validationFailed(message: "CPU training loop requires at least one iteration.")
    }
    if config.epochsPerIteration <= 0 {
        throw EnvProjectError.validationFailed(message: "CPU training loop requires at least one epoch per iteration.")
    }
    if config.miniBatchSize <= 0 {
        throw EnvProjectError.validationFailed(message: "CPU training loop requires a positive miniBatchSize.")
    }

    var summaries: [CPUTrainingIterationSummary] = []
    summaries.reserveCapacity(config.iterations)
    var totalParameterDeltaL1: Float = 0.0
    var adamState: AdamState? = nil
    let startTime = Date()

    for iteration in 0..<config.iterations {
        driver.setResetSeed(config.resetSeed &+ UInt32(iteration))

        let storage = try collectActorCriticRolloutStorage(
            driver: driver,
            policy: model,
            config: PolicyRolloutConfig(
                horizon: config.rolloutHorizon,
                samplingMode: config.policySamplingMode,
                samplingSeed: config.policySamplingSeed &+ UInt32(iteration)
            )
        )
        let estimates = try computeGAE(storage: storage, config: config.gaeConfig)
        let batch = try makePPOBatch(storage: storage, estimates: estimates)

        let meanReward = storage.rewards.reduce(0.0, +) / Float(storage.rewards.count)
        let doneCount = storage.dones.reduce(0) { $0 + ($1 == 0 ? 0 : 1) }

        let initialLoss = try computePPOLoss(
            storage: storage,
            estimates: estimates,
            policy: model,
            observationSpec: driver.observationSpec,
            actionSpec: driver.actionSpec,
            config: config.ppoConfig
        )

        var postLoss = initialLoss
        var iterationDelta: Float = 0.0
        for epoch in 0..<config.epochsPerIteration {
            let epochSeed = config.shuffleSeed &+ UInt32(iteration * config.epochsPerIteration + epoch)
            let permutation = makeDeterministicPermutation(count: batch.sampleCount, seed: epochSeed)

            var start = 0
            while start < batch.sampleCount {
                let end = min(start + config.miniBatchSize, batch.sampleCount)
                let miniBatchIndices = Array(permutation[start..<end])
                let miniBatch = try makeMiniBatch(from: batch, indices: miniBatchIndices)
                let stepSummary = try model.applyOptimizerStep(
                    batch: miniBatch,
                    observationSpec: driver.observationSpec,
                    actionSpec: driver.actionSpec,
                    ppoConfig: config.ppoConfig,
                    optimizer: config.optimizer,
                    adamState: &adamState
                )
                postLoss = stepSummary.postLoss
                iterationDelta += stepSummary.parameterDeltaL1
                start = end
            }
        }

        let iterationSummary = CPUTrainingIterationSummary(
            iteration: iteration,
            meanReward: meanReward,
            doneCount: doneCount,
            preLoss: initialLoss,
            postLoss: postLoss,
            parameterDeltaL1: iterationDelta
        )
        summaries.append(iterationSummary)
        totalParameterDeltaL1 += iterationDelta
        emitTrainingProgress(
            progressHandler,
            summary: iterationSummary,
            config: config,
            envCount: driver.envCount,
            elapsedSeconds: Date().timeIntervalSince(startTime)
        )
    }

    return CPUTrainingRunSummary(iterations: summaries, totalParameterDeltaL1: totalParameterDeltaL1)
}

func emitTrainingProgress(
    _ progressHandler: ((TrainingProgressSnapshot) -> Void)?,
    summary: CPUTrainingIterationSummary,
    config: CPUTrainingLoopConfig,
    envCount: Int,
    elapsedSeconds: TimeInterval
) {
    guard let progressHandler else {
        return
    }

    let completedIterations = summary.iteration + 1
    let stepsPerIteration = config.rolloutHorizon * envCount
    let completedEnvSteps = completedIterations * stepsPerIteration
    let totalEnvSteps = config.iterations * stepsPerIteration
    let envStepsPerSecond = elapsedSeconds > 0
        ? Double(completedEnvSteps) / elapsedSeconds
        : 0.0
    let remainingEnvSteps = max(0, totalEnvSteps - completedEnvSteps)
    let etaSeconds = envStepsPerSecond > 0
        ? Double(remainingEnvSteps) / envStepsPerSecond
        : 0.0

    progressHandler(
        TrainingProgressSnapshot(
            iteration: summary.iteration,
            totalIterations: config.iterations,
            elapsedSeconds: elapsedSeconds,
            etaSeconds: etaSeconds,
            completedEnvSteps: completedEnvSteps,
            totalEnvSteps: totalEnvSteps,
            envStepsPerSecond: envStepsPerSecond,
            summary: summary
        )
    )
}

func makeMiniBatch(from batch: PPOBatch, indices: [Int]) throws -> PPOBatch {
    var observations: [Float] = []
    var actions: [Float] = []
    var oldLogProbs: [Float] = []
    var advantages: [Float] = []
    var returns: [Float] = []

    observations.reserveCapacity(indices.count * batch.observationDim)
    actions.reserveCapacity(indices.count * batch.actionDim)
    oldLogProbs.reserveCapacity(indices.count)
    advantages.reserveCapacity(indices.count)
    returns.reserveCapacity(indices.count)

    for sampleIndex in indices {
        let obsBase = sampleIndex * batch.observationDim
        let actionBase = sampleIndex * batch.actionDim
        observations.append(contentsOf: batch.observations[obsBase..<(obsBase + batch.observationDim)])
        actions.append(contentsOf: batch.actions[actionBase..<(actionBase + batch.actionDim)])
        oldLogProbs.append(batch.oldLogProbs[sampleIndex])
        advantages.append(batch.advantages[sampleIndex])
        returns.append(batch.returns[sampleIndex])
    }

    return try PPOBatch(
        sampleCount: indices.count,
        observationDim: batch.observationDim,
        actionDim: batch.actionDim,
        observations: observations,
        actions: actions,
        oldLogProbs: oldLogProbs,
        advantages: advantages,
        returns: returns
    )
}

func makeDeterministicPermutation(count: Int, seed: UInt32) -> [Int] {
    var indices = Array(0..<count)
    var state = UInt64(seed) | 1

    if count <= 1 {
        return indices
    }

    for i in stride(from: count - 1, through: 1, by: -1) {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        let j = Int(state % UInt64(i + 1))
        indices.swapAt(i, j)
    }

    return indices
}
