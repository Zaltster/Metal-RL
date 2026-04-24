import Foundation

struct MetalSGDTrainingStepSummary {
    let updatedModel: TrainableMLPActorCritic
    let preLoss: PPOLossBreakdown
    let postLoss: PPOLossBreakdown
    let parameterDeltaL1: Float
}

func runMetalSGDTrainingStep(
    gradientComputer: MetalMLPGradientComputer,
    model: TrainableMLPActorCritic,
    batch: PPOBatch,
    observationSpec: VectorObservationSpec,
    actionSpec: VectorActionSpec,
    ppoConfig: PPOConfig,
    sgdConfig: SGDConfig
) throws -> MetalSGDTrainingStepSummary {
    let preLoss = try computeTrainableModelPPOLoss(
        model: model,
        batch: batch,
        observationSpec: observationSpec,
        actionSpec: actionSpec,
        ppoConfig: ppoConfig
    )
    let step = try gradientComputer.applySGDStep(
        model: model,
        batch: batch,
        observationSpec: observationSpec,
        actionSpec: actionSpec,
        ppoConfig: ppoConfig,
        sgdConfig: sgdConfig
    )
    let postLoss = try computeTrainableModelPPOLoss(
        model: step.model,
        batch: batch,
        observationSpec: observationSpec,
        actionSpec: actionSpec,
        ppoConfig: ppoConfig
    )

    return MetalSGDTrainingStepSummary(
        updatedModel: step.model,
        preLoss: preLoss,
        postLoss: postLoss,
        parameterDeltaL1: step.parameterDeltaL1
    )
}

func computeTrainableModelPPOLoss(
    model: TrainableMLPActorCritic,
    batch: PPOBatch,
    observationSpec: VectorObservationSpec,
    actionSpec: VectorActionSpec,
    ppoConfig: PPOConfig
) throws -> PPOLossBreakdown {
    let evaluation = try model.evaluateGaussian(
        for: batch.observations,
        taking: batch.actions,
        envCount: batch.sampleCount,
        observationSpec: observationSpec,
        actionSpec: actionSpec
    )

    return try computePPOLoss(
        oldLogProbs: batch.oldLogProbs,
        newLogProbs: evaluation.logProbs,
        advantages: batch.advantages,
        returns: batch.returns,
        newValues: evaluation.values,
        entropies: evaluation.entropies,
        config: ppoConfig
    )
}
