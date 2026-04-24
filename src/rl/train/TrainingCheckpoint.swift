import Foundation

struct MLPActorCriticCheckpoint: Codable {
    let schemaVersion: Int
    let observationDim: Int
    let actionDim: Int
    let hiddenDim: Int
    let inputWeights: [Float]
    let inputBias: [Float]
    let outputWeights: [Float]
    let outputBias: [Float]
    let valueWeights: [Float]
    let valueBias: Float
    let logStd: [Float]

    init(
        model: TrainableMLPActorCritic,
        observationSpec: VectorObservationSpec,
        actionSpec: VectorActionSpec
    ) throws {
        self.schemaVersion = 1
        self.observationDim = observationSpec.elementsPerEnv
        self.actionDim = actionSpec.dimensionsPerEnv
        self.hiddenDim = model.hiddenDim
        self.inputWeights = model.inputWeights
        self.inputBias = model.inputBias
        self.outputWeights = model.outputWeights
        self.outputBias = model.outputBias
        self.valueWeights = model.valueWeights
        self.valueBias = model.valueBias
        self.logStd = model.logStd

        try validate()
    }

    func restoreModel() throws -> TrainableMLPActorCritic {
        try validate()
        let policy = MLPPolicy(
            inputWeights: inputWeights,
            inputBias: inputBias,
            outputWeights: outputWeights,
            outputBias: outputBias,
            valueWeights: valueWeights,
            valueBias: valueBias
        )
        return TrainableMLPActorCritic(policy: policy, logStd: logStd)
    }

    func validate() throws {
        if schemaVersion != 1 {
            throw EnvProjectError.validationFailed(
                message: "Unsupported MLPActorCriticCheckpoint schemaVersion: \(schemaVersion)."
            )
        }
        if observationDim <= 0 || actionDim <= 0 || hiddenDim <= 0 {
            throw EnvProjectError.validationFailed(message: "Checkpoint dimensions must be positive.")
        }
        if inputWeights.count != hiddenDim * observationDim {
            throw EnvProjectError.validationFailed(
                message: "Checkpoint inputWeights size mismatch: expected \(hiddenDim * observationDim), got \(inputWeights.count)."
            )
        }
        if inputBias.count != hiddenDim {
            throw EnvProjectError.validationFailed(
                message: "Checkpoint inputBias size mismatch: expected \(hiddenDim), got \(inputBias.count)."
            )
        }
        if outputWeights.count != actionDim * hiddenDim {
            throw EnvProjectError.validationFailed(
                message: "Checkpoint outputWeights size mismatch: expected \(actionDim * hiddenDim), got \(outputWeights.count)."
            )
        }
        if outputBias.count != actionDim {
            throw EnvProjectError.validationFailed(
                message: "Checkpoint outputBias size mismatch: expected \(actionDim), got \(outputBias.count)."
            )
        }
        if valueWeights.count != hiddenDim {
            throw EnvProjectError.validationFailed(
                message: "Checkpoint valueWeights size mismatch: expected \(hiddenDim), got \(valueWeights.count)."
            )
        }
        if logStd.count != actionDim {
            throw EnvProjectError.validationFailed(
                message: "Checkpoint logStd size mismatch: expected \(actionDim), got \(logStd.count)."
            )
        }
    }
}

struct AdamStateCheckpoint: Codable {
    let schemaVersion: Int
    let timestep: Int
    let inputWeightsM: [Float]
    let inputWeightsV: [Float]
    let inputBiasM: [Float]
    let inputBiasV: [Float]
    let outputWeightsM: [Float]
    let outputWeightsV: [Float]
    let outputBiasM: [Float]
    let outputBiasV: [Float]
    let valueWeightsM: [Float]
    let valueWeightsV: [Float]
    let valueBiasM: Float
    let valueBiasV: Float

    init(state: AdamState, model: TrainableMLPActorCritic) throws {
        self.schemaVersion = 1
        self.timestep = state.timestep
        self.inputWeightsM = state.inputWeightsM
        self.inputWeightsV = state.inputWeightsV
        self.inputBiasM = state.inputBiasM
        self.inputBiasV = state.inputBiasV
        self.outputWeightsM = state.outputWeightsM
        self.outputWeightsV = state.outputWeightsV
        self.outputBiasM = state.outputBiasM
        self.outputBiasV = state.outputBiasV
        self.valueWeightsM = state.valueWeightsM
        self.valueWeightsV = state.valueWeightsV
        self.valueBiasM = state.valueBiasM
        self.valueBiasV = state.valueBiasV

        try validate(model: model)
    }

    func restoreState(model: TrainableMLPActorCritic) throws -> AdamState {
        try validate(model: model)
        var state = AdamState(model: model)
        state.timestep = timestep
        state.inputWeightsM = inputWeightsM
        state.inputWeightsV = inputWeightsV
        state.inputBiasM = inputBiasM
        state.inputBiasV = inputBiasV
        state.outputWeightsM = outputWeightsM
        state.outputWeightsV = outputWeightsV
        state.outputBiasM = outputBiasM
        state.outputBiasV = outputBiasV
        state.valueWeightsM = valueWeightsM
        state.valueWeightsV = valueWeightsV
        state.valueBiasM = valueBiasM
        state.valueBiasV = valueBiasV
        return state
    }

    func validate(model: TrainableMLPActorCritic) throws {
        if schemaVersion != 1 {
            throw EnvProjectError.validationFailed(
                message: "Unsupported AdamStateCheckpoint schemaVersion: \(schemaVersion)."
            )
        }
        if timestep < 0 {
            throw EnvProjectError.validationFailed(message: "AdamStateCheckpoint timestep must be non-negative.")
        }
        if inputWeightsM.count != model.inputWeights.count || inputWeightsV.count != model.inputWeights.count {
            throw EnvProjectError.validationFailed(message: "AdamStateCheckpoint input-weight size mismatch.")
        }
        if inputBiasM.count != model.inputBias.count || inputBiasV.count != model.inputBias.count {
            throw EnvProjectError.validationFailed(message: "AdamStateCheckpoint input-bias size mismatch.")
        }
        if outputWeightsM.count != model.outputWeights.count || outputWeightsV.count != model.outputWeights.count {
            throw EnvProjectError.validationFailed(message: "AdamStateCheckpoint output-weight size mismatch.")
        }
        if outputBiasM.count != model.outputBias.count || outputBiasV.count != model.outputBias.count {
            throw EnvProjectError.validationFailed(message: "AdamStateCheckpoint output-bias size mismatch.")
        }
        if valueWeightsM.count != model.valueWeights.count || valueWeightsV.count != model.valueWeights.count {
            throw EnvProjectError.validationFailed(message: "AdamStateCheckpoint value-weight size mismatch.")
        }
    }
}

struct MLPActorCriticTrainingStateCheckpoint: Codable {
    let schemaVersion: Int
    let model: MLPActorCriticCheckpoint
    let adamState: AdamStateCheckpoint

    init(
        model: TrainableMLPActorCritic,
        adamState: AdamState,
        observationSpec: VectorObservationSpec,
        actionSpec: VectorActionSpec
    ) throws {
        self.schemaVersion = 1
        self.model = try MLPActorCriticCheckpoint(
            model: model,
            observationSpec: observationSpec,
            actionSpec: actionSpec
        )
        self.adamState = try AdamStateCheckpoint(state: adamState, model: model)

        try validate()
    }

    init(
        metalModel: MetalTrainableMLPActorCritic,
        observationSpec: VectorObservationSpec,
        actionSpec: VectorActionSpec
    ) throws {
        let model = metalModel.readModel()
        let adamState = metalModel.readAdamState()
        try self.init(
            model: model,
            adamState: adamState,
            observationSpec: observationSpec,
            actionSpec: actionSpec
        )
    }

    func restoreModelAndAdamState() throws -> (model: TrainableMLPActorCritic, adamState: AdamState) {
        try validate()
        let restoredModel = try model.restoreModel()
        let restoredState = try adamState.restoreState(model: restoredModel)
        return (model: restoredModel, adamState: restoredState)
    }

    func validate() throws {
        if schemaVersion != 1 {
            throw EnvProjectError.validationFailed(
                message: "Unsupported MLPActorCriticTrainingStateCheckpoint schemaVersion: \(schemaVersion)."
            )
        }
        try model.validate()
        let restoredModel = try model.restoreModel()
        try adamState.validate(model: restoredModel)
    }
}

func saveCheckpoint(_ checkpoint: MLPActorCriticCheckpoint, to url: URL) throws {
    try checkpoint.validate()
    let parent = url.deletingLastPathComponent()
    try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)

    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let data = try encoder.encode(checkpoint)
    try data.write(to: url, options: [.atomic])
}

func saveTrainingStateCheckpoint(_ checkpoint: MLPActorCriticTrainingStateCheckpoint, to url: URL) throws {
    try checkpoint.validate()
    let parent = url.deletingLastPathComponent()
    try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)

    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let data = try encoder.encode(checkpoint)
    try data.write(to: url, options: [.atomic])
}

func loadMLPActorCriticCheckpoint(from url: URL) throws -> MLPActorCriticCheckpoint {
    let data = try Data(contentsOf: url)
    let checkpoint = try JSONDecoder().decode(MLPActorCriticCheckpoint.self, from: data)
    try checkpoint.validate()
    return checkpoint
}

func loadMLPActorCriticTrainingStateCheckpoint(from url: URL) throws -> MLPActorCriticTrainingStateCheckpoint {
    let data = try Data(contentsOf: url)
    let checkpoint = try JSONDecoder().decode(MLPActorCriticTrainingStateCheckpoint.self, from: data)
    try checkpoint.validate()
    return checkpoint
}
