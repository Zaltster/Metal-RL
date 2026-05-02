import Foundation
import Metal

func runPersistentMetalTrainingLoop(
    device: MTLDevice,
    rootDir: String,
    driver: CartPoleVectorEnvDriver,
    model: inout TrainableMLPActorCritic,
    gpuPolicy: MetalMLPPolicy,
    config: CPUTrainingLoopConfig,
    progressHandler: ((TrainingProgressSnapshot) -> Void)? = nil
) throws -> CPUTrainingRunSummary {
    if config.iterations <= 0 {
        throw EnvProjectError.validationFailed(message: "Persistent Metal training loop requires at least one iteration.")
    }
    if config.epochsPerIteration <= 0 {
        throw EnvProjectError.validationFailed(message: "Persistent Metal training loop requires at least one epoch per iteration.")
    }
    if config.miniBatchSize <= 0 {
        throw EnvProjectError.validationFailed(message: "Persistent Metal training loop requires a positive miniBatchSize.")
    }

    let metalModel = try MetalTrainableMLPActorCritic(
        device: device,
        rootDir: rootDir,
        model: model,
        observationSpec: driver.observationSpec,
        actionSpec: driver.actionSpec
    )
    var summaries: [CPUTrainingIterationSummary] = []
    summaries.reserveCapacity(config.iterations)
    var totalParameterDeltaL1: Float = 0.0
    let startTime = Date()

    for iteration in 0..<config.iterations {
        try gpuPolicy.load(from: metalModel)
        driver.setResetSeed(config.resetSeed &+ UInt32(iteration))

        let storage = try collectActorCriticRolloutStorage(
            driver: driver,
            policy: gpuPolicy,
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
            policy: gpuPolicy,
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
                let step: MetalOptimizerStepResult
                switch config.optimizer {
                case let .sgd(sgdConfig):
                    step = try metalModel.applySGDStep(
                        batch: miniBatch,
                        ppoConfig: config.ppoConfig,
                        sgdConfig: sgdConfig
                    )
                case let .adam(adamConfig):
                    step = try metalModel.applyAdamStep(
                        batch: miniBatch,
                        ppoConfig: config.ppoConfig,
                        adamConfig: adamConfig
                    )
                }
                postLoss = try computeTrainableModelPPOLoss(
                    model: step.model,
                    batch: miniBatch,
                    observationSpec: driver.observationSpec,
                    actionSpec: driver.actionSpec,
                    ppoConfig: config.ppoConfig
                )
                iterationDelta += step.parameterDeltaL1
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

    model = metalModel.readModel()
    return CPUTrainingRunSummary(iterations: summaries, totalParameterDeltaL1: totalParameterDeltaL1)
}
