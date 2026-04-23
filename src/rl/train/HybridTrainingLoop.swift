import Foundation

func runHybridTrainingLoop(
    driver: CartPoleVectorEnvDriver,
    cpuModel: inout TrainableMLPActorCritic,
    gpuPolicy: MetalMLPPolicy,
    config: CPUTrainingLoopConfig
) throws -> CPUTrainingRunSummary {
    if config.iterations <= 0 {
        throw EnvProjectError.validationFailed(message: "Hybrid training loop requires at least one iteration.")
    }
    if config.epochsPerIteration <= 0 {
        throw EnvProjectError.validationFailed(message: "Hybrid training loop requires at least one epoch per iteration.")
    }
    if config.miniBatchSize <= 0 {
        throw EnvProjectError.validationFailed(message: "Hybrid training loop requires a positive miniBatchSize.")
    }

    var summaries: [CPUTrainingIterationSummary] = []
    summaries.reserveCapacity(config.iterations)
    var totalParameterDeltaL1: Float = 0.0
    var adamState: AdamState? = nil

    for iteration in 0..<config.iterations {
        try gpuPolicy.load(model: cpuModel)
        driver.setResetSeed(config.resetSeed &+ UInt32(iteration))

        let storage = try collectActorCriticRolloutStorage(
            driver: driver,
            policy: gpuPolicy,
            config: PolicyRolloutConfig(horizon: config.rolloutHorizon)
        )
        let estimates = try computeGAE(storage: storage, config: config.gaeConfig)
        let batch = try makePPOBatch(storage: storage, estimates: estimates)

        let meanReward = storage.rewards.reduce(0.0, +) / Float(storage.rewards.count)
        let doneCount = storage.dones.reduce(0) { $0 + ($1 == 0 ? 0 : 1) }

        let initialLoss = try computePPOLoss(
            storage: storage,
            estimates: estimates,
            policy: cpuModel,
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
                let stepSummary = try cpuModel.applyOptimizerStep(
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

        summaries.append(
            CPUTrainingIterationSummary(
                iteration: iteration,
                meanReward: meanReward,
                doneCount: doneCount,
                preLoss: initialLoss,
                postLoss: postLoss,
                parameterDeltaL1: iterationDelta
            )
        )
        totalParameterDeltaL1 += iterationDelta
    }

    return CPUTrainingRunSummary(iterations: summaries, totalParameterDeltaL1: totalParameterDeltaL1)
}

func makeTrainingSummaryLines(summary: CPUTrainingRunSummary) -> [String] {
    summary.iterations.map { iteration in
        String(
            format: "iter %d meanReward=% .5f doneCount=%d preLoss=% .6f postLoss=% .6f paramDelta=% .6f",
            iteration.iteration,
            iteration.meanReward,
            iteration.doneCount,
            iteration.preLoss.totalLoss,
            iteration.postLoss.totalLoss,
            iteration.parameterDeltaL1
        )
    }
}
