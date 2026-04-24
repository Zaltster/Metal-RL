import Foundation
import Metal

struct MetalMLPTrainableParameterBuffers {
    let observationDim: Int
    let hiddenDim: Int
    let actionDim: Int
    let inputWeightBuffer: MTLBuffer
    let inputBiasBuffer: MTLBuffer
    let outputWeightBuffer: MTLBuffer
    let outputBiasBuffer: MTLBuffer
    let valueWeightBuffer: MTLBuffer
    let valueBiasBuffer: MTLBuffer
    let inputWeightCount: Int
    let inputBiasCount: Int
    let outputWeightCount: Int
    let outputBiasCount: Int
    let valueWeightCount: Int
}

final class MetalTrainableMLPActorCritic {
    private let commandQueue: MTLCommandQueue
    private let perSamplePipeline: MTLComputePipelineState
    private let reductionPipeline: MTLComputePipelineState
    private let sgdUpdatePipeline: MTLComputePipelineState
    private let adamUpdatePipeline: MTLComputePipelineState

    private let observationDim: Int
    private let hiddenDim: Int
    private let actionDim: Int

    private let inputWeightBuffer: MTLBuffer
    private let inputBiasBuffer: MTLBuffer
    private let outputWeightBuffer: MTLBuffer
    private let outputBiasBuffer: MTLBuffer
    private let valueWeightBuffer: MTLBuffer
    private let valueBiasBuffer: MTLBuffer
    private let logStdBuffer: MTLBuffer
    private let logStd: [Float]

    private let adamBuffers: PersistentAdamBuffers
    private var adamTimestep = 0
    private var currentModelSnapshot: TrainableMLPActorCritic

    init(
        device: MTLDevice,
        rootDir: String,
        model: TrainableMLPActorCritic,
        adamState: AdamState? = nil,
        observationSpec: VectorObservationSpec,
        actionSpec: VectorActionSpec
    ) throws {
        guard let commandQueue = device.makeCommandQueue() else {
            throw EnvProjectError.commandQueueUnavailable
        }
        self.commandQueue = commandQueue

        let shaderPath = URL(fileURLWithPath: rootDir)
            .appending(path: "src/rl/train/Shaders/mlp_gradients.metal")
            .path()
        let library = try makeLibrary(device: device, shaderPath: shaderPath)
        self.perSamplePipeline = try makePipeline(device: device, library: library, name: "mlp_ppo_per_sample_gradients")
        self.reductionPipeline = try makePipeline(device: device, library: library, name: "mlp_reduce_per_sample_gradients")
        self.sgdUpdatePipeline = try makePipeline(device: device, library: library, name: "mlp_sgd_update")
        self.adamUpdatePipeline = try makePipeline(device: device, library: library, name: "mlp_adam_update")

        self.observationDim = observationSpec.elementsPerEnv
        self.hiddenDim = model.hiddenDim
        self.actionDim = actionSpec.dimensionsPerEnv
        self.logStd = model.logStd
        self.currentModelSnapshot = model

        try Self.validateModel(
            model,
            observationDim: observationSpec.elementsPerEnv,
            actionDim: actionSpec.dimensionsPerEnv,
            context: "MetalTrainableMLPActorCritic"
        )

        self.inputWeightBuffer = try Self.makeFloatBuffer(device: device, values: model.inputWeights, name: "persistent-input-weights")
        self.inputBiasBuffer = try Self.makeFloatBuffer(device: device, values: model.inputBias, name: "persistent-input-bias")
        self.outputWeightBuffer = try Self.makeFloatBuffer(device: device, values: model.outputWeights, name: "persistent-output-weights")
        self.outputBiasBuffer = try Self.makeFloatBuffer(device: device, values: model.outputBias, name: "persistent-output-bias")
        self.valueWeightBuffer = try Self.makeFloatBuffer(device: device, values: model.valueWeights, name: "persistent-value-weights")
        self.valueBiasBuffer = try Self.makeFloatBuffer(device: device, values: [model.valueBias], name: "persistent-value-bias")
        self.logStdBuffer = try Self.makeFloatBuffer(device: device, values: model.logStd, name: "persistent-log-std")
        self.adamBuffers = try Self.makeAdamBuffers(device: device, model: model, state: adamState)
        self.adamTimestep = adamState?.timestep ?? 0
    }

    func readModel() -> TrainableMLPActorCritic {
        TrainableMLPActorCritic(
            policy: MLPPolicy(
                inputWeights: readArray(from: inputWeightBuffer, count: currentModelSnapshot.inputWeights.count),
                inputBias: readArray(from: inputBiasBuffer, count: currentModelSnapshot.inputBias.count),
                outputWeights: readArray(from: outputWeightBuffer, count: currentModelSnapshot.outputWeights.count),
                outputBias: readArray(from: outputBiasBuffer, count: currentModelSnapshot.outputBias.count),
                valueWeights: readArray(from: valueWeightBuffer, count: currentModelSnapshot.valueWeights.count),
                valueBias: readArray(from: valueBiasBuffer, count: 1)[0]
            ),
            logStd: logStd
        )
    }

    func parameterBuffers() -> MetalMLPTrainableParameterBuffers {
        MetalMLPTrainableParameterBuffers(
            observationDim: observationDim,
            hiddenDim: hiddenDim,
            actionDim: actionDim,
            inputWeightBuffer: inputWeightBuffer,
            inputBiasBuffer: inputBiasBuffer,
            outputWeightBuffer: outputWeightBuffer,
            outputBiasBuffer: outputBiasBuffer,
            valueWeightBuffer: valueWeightBuffer,
            valueBiasBuffer: valueBiasBuffer,
            inputWeightCount: currentModelSnapshot.inputWeights.count,
            inputBiasCount: currentModelSnapshot.inputBias.count,
            outputWeightCount: currentModelSnapshot.outputWeights.count,
            outputBiasCount: currentModelSnapshot.outputBias.count,
            valueWeightCount: currentModelSnapshot.valueWeights.count
        )
    }

    func readAdamState() -> AdamState {
        var state = AdamState(model: currentModelSnapshot)
        state.timestep = adamTimestep
        state.inputWeightsM = readArray(from: adamBuffers.inputWeightM, count: currentModelSnapshot.inputWeights.count)
        state.inputWeightsV = readArray(from: adamBuffers.inputWeightV, count: currentModelSnapshot.inputWeights.count)
        state.inputBiasM = readArray(from: adamBuffers.inputBiasM, count: currentModelSnapshot.inputBias.count)
        state.inputBiasV = readArray(from: adamBuffers.inputBiasV, count: currentModelSnapshot.inputBias.count)
        state.outputWeightsM = readArray(from: adamBuffers.outputWeightM, count: currentModelSnapshot.outputWeights.count)
        state.outputWeightsV = readArray(from: adamBuffers.outputWeightV, count: currentModelSnapshot.outputWeights.count)
        state.outputBiasM = readArray(from: adamBuffers.outputBiasM, count: currentModelSnapshot.outputBias.count)
        state.outputBiasV = readArray(from: adamBuffers.outputBiasV, count: currentModelSnapshot.outputBias.count)
        state.valueWeightsM = readArray(from: adamBuffers.valueWeightM, count: currentModelSnapshot.valueWeights.count)
        state.valueWeightsV = readArray(from: adamBuffers.valueWeightV, count: currentModelSnapshot.valueWeights.count)
        state.valueBiasM = readArray(from: adamBuffers.valueBiasM, count: 1)[0]
        state.valueBiasV = readArray(from: adamBuffers.valueBiasV, count: 1)[0]
        return state
    }

    func applySGDStep(
        batch: PPOBatch,
        ppoConfig: PPOConfig,
        sgdConfig: SGDConfig
    ) throws -> MetalOptimizerStepResult {
        if !sgdConfig.learningRate.isFinite || sgdConfig.learningRate <= 0.0 {
            throw EnvProjectError.validationFailed(
                message: "MetalTrainableMLPActorCritic requires a positive finite SGD learning rate."
            )
        }
        try validateBatch(batch)

        let before = currentModelSnapshot
        let gradients = try computeReducedGradientBuffers(batch: batch, ppoConfig: ppoConfig)
        for group in parameterGradientGroups(gradients: gradients) {
            try applySGDUpdate(
                parameterBuffer: group.parameterBuffer,
                gradientBuffer: group.gradientBuffer,
                parameterCount: group.parameterCount,
                learningRate: sgdConfig.learningRate
            )
        }

        let after = readModel()
        currentModelSnapshot = after
        return MetalOptimizerStepResult(model: after, parameterDeltaL1: Self.parameterDeltaL1(before: before, after: after))
    }

    func applyAdamStep(
        batch: PPOBatch,
        ppoConfig: PPOConfig,
        adamConfig: AdamConfig
    ) throws -> MetalOptimizerStepResult {
        try validateAdamConfig(adamConfig)
        try validateBatch(batch)

        let before = currentModelSnapshot
        let gradients = try computeReducedGradientBuffers(batch: batch, ppoConfig: ppoConfig)
        adamTimestep += 1
        let t = Float(adamTimestep)
        let biasCorrection1 = 1.0 - pow(adamConfig.beta1, t)
        let biasCorrection2 = 1.0 - pow(adamConfig.beta2, t)

        for group in parameterGradientAdamGroups(gradients: gradients) {
            try applyAdamUpdate(
                parameterBuffer: group.parameterBuffer,
                gradientBuffer: group.gradientBuffer,
                momentumBuffer: group.momentumBuffer,
                velocityBuffer: group.velocityBuffer,
                parameterCount: group.parameterCount,
                adamConfig: adamConfig,
                biasCorrection1: biasCorrection1,
                biasCorrection2: biasCorrection2
            )
        }

        let after = readModel()
        currentModelSnapshot = after
        return MetalOptimizerStepResult(model: after, parameterDeltaL1: Self.parameterDeltaL1(before: before, after: after))
    }

    private func computeReducedGradientBuffers(
        batch: PPOBatch,
        ppoConfig: PPOConfig
    ) throws -> PersistentGradientBuffers {
        let device = perSamplePipeline.device
        let sampleCount = batch.sampleCount
        let inputWeightCount = currentModelSnapshot.inputWeights.count
        let inputBiasCount = currentModelSnapshot.inputBias.count
        let outputWeightCount = currentModelSnapshot.outputWeights.count
        let outputBiasCount = currentModelSnapshot.outputBias.count
        let valueWeightCount = currentModelSnapshot.valueWeights.count

        let observationBuffer = try Self.makeFloatBuffer(device: device, values: batch.observations, name: "persistent-gradient-observations")
        let actionBuffer = try Self.makeFloatBuffer(device: device, values: batch.actions, name: "persistent-gradient-actions")
        let oldLogProbBuffer = try Self.makeFloatBuffer(device: device, values: batch.oldLogProbs, name: "persistent-gradient-old-logprobs")
        let advantageBuffer = try Self.makeFloatBuffer(device: device, values: batch.advantages, name: "persistent-gradient-advantages")
        let returnBuffer = try Self.makeFloatBuffer(device: device, values: batch.returns, name: "persistent-gradient-returns")

        let inputWeightPerSampleGradientBuffer = try Self.makeZeroedFloatBuffer(
            device: device,
            count: sampleCount * inputWeightCount,
            name: "persistent-input-weight-per-sample"
        )
        let inputBiasPerSampleGradientBuffer = try Self.makeZeroedFloatBuffer(
            device: device,
            count: sampleCount * inputBiasCount,
            name: "persistent-input-bias-per-sample"
        )
        let outputWeightPerSampleGradientBuffer = try Self.makeZeroedFloatBuffer(
            device: device,
            count: sampleCount * outputWeightCount,
            name: "persistent-output-weight-per-sample"
        )
        let outputBiasPerSampleGradientBuffer = try Self.makeZeroedFloatBuffer(
            device: device,
            count: sampleCount * outputBiasCount,
            name: "persistent-output-bias-per-sample"
        )
        let valueWeightPerSampleGradientBuffer = try Self.makeZeroedFloatBuffer(
            device: device,
            count: sampleCount * valueWeightCount,
            name: "persistent-value-weight-per-sample"
        )
        let valueBiasPerSampleGradientBuffer = try Self.makeZeroedFloatBuffer(
            device: device,
            count: sampleCount,
            name: "persistent-value-bias-per-sample"
        )

        var params = MLPGradientParams(
            sampleCount: UInt32(sampleCount),
            observationDim: UInt32(observationDim),
            hiddenDim: UInt32(hiddenDim),
            actionDim: UInt32(actionDim),
            clipEpsilon: ppoConfig.clipEpsilon,
            valueCoefficient: ppoConfig.valueCoefficient
        )
        guard let paramsBuffer = device.makeBuffer(length: MemoryLayout<MLPGradientParams>.stride, options: .storageModeShared) else {
            throw EnvProjectError.bufferAllocationFailed("persistent-gradient-params")
        }
        writeValue(&params, to: paramsBuffer)

        try runComputePass(commandQueue: commandQueue, pipeline: perSamplePipeline, count: sampleCount) { encoder in
            encoder.setBuffer(observationBuffer, offset: 0, index: 0)
            encoder.setBuffer(actionBuffer, offset: 0, index: 1)
            encoder.setBuffer(oldLogProbBuffer, offset: 0, index: 2)
            encoder.setBuffer(advantageBuffer, offset: 0, index: 3)
            encoder.setBuffer(returnBuffer, offset: 0, index: 4)
            encoder.setBuffer(inputWeightBuffer, offset: 0, index: 5)
            encoder.setBuffer(inputBiasBuffer, offset: 0, index: 6)
            encoder.setBuffer(outputWeightBuffer, offset: 0, index: 7)
            encoder.setBuffer(outputBiasBuffer, offset: 0, index: 8)
            encoder.setBuffer(valueWeightBuffer, offset: 0, index: 9)
            encoder.setBuffer(valueBiasBuffer, offset: 0, index: 10)
            encoder.setBuffer(logStdBuffer, offset: 0, index: 11)
            encoder.setBuffer(inputWeightPerSampleGradientBuffer, offset: 0, index: 12)
            encoder.setBuffer(inputBiasPerSampleGradientBuffer, offset: 0, index: 13)
            encoder.setBuffer(outputWeightPerSampleGradientBuffer, offset: 0, index: 14)
            encoder.setBuffer(outputBiasPerSampleGradientBuffer, offset: 0, index: 15)
            encoder.setBuffer(valueWeightPerSampleGradientBuffer, offset: 0, index: 16)
            encoder.setBuffer(valueBiasPerSampleGradientBuffer, offset: 0, index: 17)
            encoder.setBuffer(paramsBuffer, offset: 0, index: 18)
        }

        return PersistentGradientBuffers(
            inputWeightGradientBuffer: try reducePerSampleGradientsOnGPU(
                inputWeightPerSampleGradientBuffer,
                sampleCount: sampleCount,
                parameterCount: inputWeightCount,
                name: "persistent-input-weight-reduced"
            ),
            inputBiasGradientBuffer: try reducePerSampleGradientsOnGPU(
                inputBiasPerSampleGradientBuffer,
                sampleCount: sampleCount,
                parameterCount: inputBiasCount,
                name: "persistent-input-bias-reduced"
            ),
            outputWeightGradientBuffer: try reducePerSampleGradientsOnGPU(
                outputWeightPerSampleGradientBuffer,
                sampleCount: sampleCount,
                parameterCount: outputWeightCount,
                name: "persistent-output-weight-reduced"
            ),
            outputBiasGradientBuffer: try reducePerSampleGradientsOnGPU(
                outputBiasPerSampleGradientBuffer,
                sampleCount: sampleCount,
                parameterCount: outputBiasCount,
                name: "persistent-output-bias-reduced"
            ),
            valueWeightGradientBuffer: try reducePerSampleGradientsOnGPU(
                valueWeightPerSampleGradientBuffer,
                sampleCount: sampleCount,
                parameterCount: valueWeightCount,
                name: "persistent-value-weight-reduced"
            ),
            valueBiasGradientBuffer: try reducePerSampleGradientsOnGPU(
                valueBiasPerSampleGradientBuffer,
                sampleCount: sampleCount,
                parameterCount: 1,
                name: "persistent-value-bias-reduced"
            )
        )
    }

    private func reducePerSampleGradientsOnGPU(
        _ perSampleBuffer: MTLBuffer,
        sampleCount: Int,
        parameterCount: Int,
        name: String
    ) throws -> MTLBuffer {
        let device = reductionPipeline.device
        let reducedBuffer = try Self.makeZeroedFloatBuffer(device: device, count: parameterCount, name: name)
        var params = MLPGradientReductionParams(sampleCount: UInt32(sampleCount), parameterCount: UInt32(parameterCount))
        guard let paramsBuffer = device.makeBuffer(length: MemoryLayout<MLPGradientReductionParams>.stride, options: .storageModeShared) else {
            throw EnvProjectError.bufferAllocationFailed("\(name)-params")
        }
        writeValue(&params, to: paramsBuffer)

        try runComputePass(commandQueue: commandQueue, pipeline: reductionPipeline, count: parameterCount) { encoder in
            encoder.setBuffer(perSampleBuffer, offset: 0, index: 0)
            encoder.setBuffer(reducedBuffer, offset: 0, index: 1)
            encoder.setBuffer(paramsBuffer, offset: 0, index: 2)
        }
        return reducedBuffer
    }

    private func applySGDUpdate(
        parameterBuffer: MTLBuffer,
        gradientBuffer: MTLBuffer,
        parameterCount: Int,
        learningRate: Float
    ) throws {
        let device = sgdUpdatePipeline.device
        var params = MLPSGDUpdateParams(learningRate: learningRate, parameterCount: UInt32(parameterCount))
        guard let paramsBuffer = device.makeBuffer(length: MemoryLayout<MLPSGDUpdateParams>.stride, options: .storageModeShared) else {
            throw EnvProjectError.bufferAllocationFailed("persistent-sgd-update-params")
        }
        writeValue(&params, to: paramsBuffer)

        try runComputePass(commandQueue: commandQueue, pipeline: sgdUpdatePipeline, count: parameterCount) { encoder in
            encoder.setBuffer(parameterBuffer, offset: 0, index: 0)
            encoder.setBuffer(gradientBuffer, offset: 0, index: 1)
            encoder.setBuffer(paramsBuffer, offset: 0, index: 2)
        }
    }

    private func applyAdamUpdate(
        parameterBuffer: MTLBuffer,
        gradientBuffer: MTLBuffer,
        momentumBuffer: MTLBuffer,
        velocityBuffer: MTLBuffer,
        parameterCount: Int,
        adamConfig: AdamConfig,
        biasCorrection1: Float,
        biasCorrection2: Float
    ) throws {
        let device = adamUpdatePipeline.device
        var params = MLPAdamUpdateParams(
            learningRate: adamConfig.learningRate,
            beta1: adamConfig.beta1,
            beta2: adamConfig.beta2,
            epsilon: adamConfig.epsilon,
            biasCorrection1: biasCorrection1,
            biasCorrection2: biasCorrection2,
            parameterCount: UInt32(parameterCount)
        )
        guard let paramsBuffer = device.makeBuffer(length: MemoryLayout<MLPAdamUpdateParams>.stride, options: .storageModeShared) else {
            throw EnvProjectError.bufferAllocationFailed("persistent-adam-update-params")
        }
        writeValue(&params, to: paramsBuffer)

        try runComputePass(commandQueue: commandQueue, pipeline: adamUpdatePipeline, count: parameterCount) { encoder in
            encoder.setBuffer(parameterBuffer, offset: 0, index: 0)
            encoder.setBuffer(gradientBuffer, offset: 0, index: 1)
            encoder.setBuffer(momentumBuffer, offset: 0, index: 2)
            encoder.setBuffer(velocityBuffer, offset: 0, index: 3)
            encoder.setBuffer(paramsBuffer, offset: 0, index: 4)
        }
    }

    private func parameterGradientGroups(
        gradients: PersistentGradientBuffers
    ) -> [(parameterBuffer: MTLBuffer, gradientBuffer: MTLBuffer, parameterCount: Int)] {
        [
            (inputWeightBuffer, gradients.inputWeightGradientBuffer, currentModelSnapshot.inputWeights.count),
            (inputBiasBuffer, gradients.inputBiasGradientBuffer, currentModelSnapshot.inputBias.count),
            (outputWeightBuffer, gradients.outputWeightGradientBuffer, currentModelSnapshot.outputWeights.count),
            (outputBiasBuffer, gradients.outputBiasGradientBuffer, currentModelSnapshot.outputBias.count),
            (valueWeightBuffer, gradients.valueWeightGradientBuffer, currentModelSnapshot.valueWeights.count),
            (valueBiasBuffer, gradients.valueBiasGradientBuffer, 1),
        ]
    }

    private func parameterGradientAdamGroups(
        gradients: PersistentGradientBuffers
    ) -> [(parameterBuffer: MTLBuffer, gradientBuffer: MTLBuffer, momentumBuffer: MTLBuffer, velocityBuffer: MTLBuffer, parameterCount: Int)] {
        [
            (
                inputWeightBuffer,
                gradients.inputWeightGradientBuffer,
                adamBuffers.inputWeightM,
                adamBuffers.inputWeightV,
                currentModelSnapshot.inputWeights.count
            ),
            (
                inputBiasBuffer,
                gradients.inputBiasGradientBuffer,
                adamBuffers.inputBiasM,
                adamBuffers.inputBiasV,
                currentModelSnapshot.inputBias.count
            ),
            (
                outputWeightBuffer,
                gradients.outputWeightGradientBuffer,
                adamBuffers.outputWeightM,
                adamBuffers.outputWeightV,
                currentModelSnapshot.outputWeights.count
            ),
            (
                outputBiasBuffer,
                gradients.outputBiasGradientBuffer,
                adamBuffers.outputBiasM,
                adamBuffers.outputBiasV,
                currentModelSnapshot.outputBias.count
            ),
            (
                valueWeightBuffer,
                gradients.valueWeightGradientBuffer,
                adamBuffers.valueWeightM,
                adamBuffers.valueWeightV,
                currentModelSnapshot.valueWeights.count
            ),
            (
                valueBiasBuffer,
                gradients.valueBiasGradientBuffer,
                adamBuffers.valueBiasM,
                adamBuffers.valueBiasV,
                1
            ),
        ]
    }

    private func validateBatch(_ batch: PPOBatch) throws {
        if batch.sampleCount <= 0 {
            throw EnvProjectError.validationFailed(message: "MetalTrainableMLPActorCritic requires at least one batch sample.")
        }
        if batch.observationDim != observationDim || batch.actionDim != actionDim {
            throw EnvProjectError.validationFailed(message: "MetalTrainableMLPActorCritic batch spec mismatch.")
        }
    }

    private func validateAdamConfig(_ config: AdamConfig) throws {
        if !config.learningRate.isFinite || config.learningRate <= 0.0 {
            throw EnvProjectError.validationFailed(
                message: "MetalTrainableMLPActorCritic requires a positive finite Adam learning rate."
            )
        }
        if !config.beta1.isFinite || config.beta1 < 0.0 || config.beta1 >= 1.0 {
            throw EnvProjectError.validationFailed(message: "MetalTrainableMLPActorCritic requires Adam beta1 in [0, 1).")
        }
        if !config.beta2.isFinite || config.beta2 < 0.0 || config.beta2 >= 1.0 {
            throw EnvProjectError.validationFailed(message: "MetalTrainableMLPActorCritic requires Adam beta2 in [0, 1).")
        }
        if !config.epsilon.isFinite || config.epsilon <= 0.0 {
            throw EnvProjectError.validationFailed(message: "MetalTrainableMLPActorCritic requires a positive finite Adam epsilon.")
        }
    }

    private static func validateModel(
        _ model: TrainableMLPActorCritic,
        observationDim: Int,
        actionDim: Int,
        context: String
    ) throws {
        let hiddenDim = model.hiddenDim
        if hiddenDim <= 0 || observationDim <= 0 || actionDim <= 0 {
            throw EnvProjectError.validationFailed(message: "\(context) dimensions must be positive.")
        }
        if model.inputWeights.count != hiddenDim * observationDim {
            throw EnvProjectError.validationFailed(message: "\(context) input-weight size mismatch.")
        }
        if model.outputWeights.count != actionDim * hiddenDim {
            throw EnvProjectError.validationFailed(message: "\(context) output-weight size mismatch.")
        }
        if model.outputBias.count != actionDim {
            throw EnvProjectError.validationFailed(message: "\(context) output-bias size mismatch.")
        }
        if model.valueWeights.count != hiddenDim {
            throw EnvProjectError.validationFailed(message: "\(context) value-weight size mismatch.")
        }
        if model.logStd.count != actionDim {
            throw EnvProjectError.validationFailed(message: "\(context) logStd size mismatch.")
        }
    }

    private static func makeFloatBuffer(device: MTLDevice, values: [Float], name: String) throws -> MTLBuffer {
        guard let buffer = device.makeBuffer(length: MemoryLayout<Float>.stride * values.count, options: .storageModeShared) else {
            throw EnvProjectError.bufferAllocationFailed(name)
        }
        copyArray(values, to: buffer)
        return buffer
    }

    private static func makeZeroedFloatBuffer(device: MTLDevice, count: Int, name: String) throws -> MTLBuffer {
        try makeFloatBuffer(device: device, values: Array(repeating: 0.0, count: count), name: name)
    }

    private static func makeAdamBuffers(
        device: MTLDevice,
        model: TrainableMLPActorCritic,
        state: AdamState?
    ) throws -> PersistentAdamBuffers {
        if let state {
            try validateAdamState(state, model: model)
        }

        return PersistentAdamBuffers(
            inputWeightM: try makeFloatBuffer(
                device: device,
                values: state?.inputWeightsM ?? Array(repeating: 0.0, count: model.inputWeights.count),
                name: "persistent-adam-input-weight-m"
            ),
            inputWeightV: try makeFloatBuffer(
                device: device,
                values: state?.inputWeightsV ?? Array(repeating: 0.0, count: model.inputWeights.count),
                name: "persistent-adam-input-weight-v"
            ),
            inputBiasM: try makeFloatBuffer(
                device: device,
                values: state?.inputBiasM ?? Array(repeating: 0.0, count: model.inputBias.count),
                name: "persistent-adam-input-bias-m"
            ),
            inputBiasV: try makeFloatBuffer(
                device: device,
                values: state?.inputBiasV ?? Array(repeating: 0.0, count: model.inputBias.count),
                name: "persistent-adam-input-bias-v"
            ),
            outputWeightM: try makeFloatBuffer(
                device: device,
                values: state?.outputWeightsM ?? Array(repeating: 0.0, count: model.outputWeights.count),
                name: "persistent-adam-output-weight-m"
            ),
            outputWeightV: try makeFloatBuffer(
                device: device,
                values: state?.outputWeightsV ?? Array(repeating: 0.0, count: model.outputWeights.count),
                name: "persistent-adam-output-weight-v"
            ),
            outputBiasM: try makeFloatBuffer(
                device: device,
                values: state?.outputBiasM ?? Array(repeating: 0.0, count: model.outputBias.count),
                name: "persistent-adam-output-bias-m"
            ),
            outputBiasV: try makeFloatBuffer(
                device: device,
                values: state?.outputBiasV ?? Array(repeating: 0.0, count: model.outputBias.count),
                name: "persistent-adam-output-bias-v"
            ),
            valueWeightM: try makeFloatBuffer(
                device: device,
                values: state?.valueWeightsM ?? Array(repeating: 0.0, count: model.valueWeights.count),
                name: "persistent-adam-value-weight-m"
            ),
            valueWeightV: try makeFloatBuffer(
                device: device,
                values: state?.valueWeightsV ?? Array(repeating: 0.0, count: model.valueWeights.count),
                name: "persistent-adam-value-weight-v"
            ),
            valueBiasM: try makeFloatBuffer(
                device: device,
                values: [state?.valueBiasM ?? 0.0],
                name: "persistent-adam-value-bias-m"
            ),
            valueBiasV: try makeFloatBuffer(
                device: device,
                values: [state?.valueBiasV ?? 0.0],
                name: "persistent-adam-value-bias-v"
            )
        )
    }

    private static func validateAdamState(_ state: AdamState, model: TrainableMLPActorCritic) throws {
        if state.timestep < 0 {
            throw EnvProjectError.validationFailed(message: "Persistent Metal Adam state timestep must be non-negative.")
        }
        if state.inputWeightsM.count != model.inputWeights.count || state.inputWeightsV.count != model.inputWeights.count {
            throw EnvProjectError.validationFailed(message: "Persistent Metal Adam input-weight state size mismatch.")
        }
        if state.inputBiasM.count != model.inputBias.count || state.inputBiasV.count != model.inputBias.count {
            throw EnvProjectError.validationFailed(message: "Persistent Metal Adam input-bias state size mismatch.")
        }
        if state.outputWeightsM.count != model.outputWeights.count || state.outputWeightsV.count != model.outputWeights.count {
            throw EnvProjectError.validationFailed(message: "Persistent Metal Adam output-weight state size mismatch.")
        }
        if state.outputBiasM.count != model.outputBias.count || state.outputBiasV.count != model.outputBias.count {
            throw EnvProjectError.validationFailed(message: "Persistent Metal Adam output-bias state size mismatch.")
        }
        if state.valueWeightsM.count != model.valueWeights.count || state.valueWeightsV.count != model.valueWeights.count {
            throw EnvProjectError.validationFailed(message: "Persistent Metal Adam value-weight state size mismatch.")
        }
    }

    private static func parameterDeltaL1(before: TrainableMLPActorCritic, after: TrainableMLPActorCritic) -> Float {
        l1Delta(before.inputWeights, after.inputWeights) +
            l1Delta(before.inputBias, after.inputBias) +
            l1Delta(before.outputWeights, after.outputWeights) +
            l1Delta(before.outputBias, after.outputBias) +
            l1Delta(before.valueWeights, after.valueWeights) +
            abs(before.valueBias - after.valueBias)
    }

    private static func l1Delta(_ lhs: [Float], _ rhs: [Float]) -> Float {
        zip(lhs, rhs).reduce(Float.zero) { partial, values in
            partial + abs(values.0 - values.1)
        }
    }
}

private struct PersistentGradientBuffers {
    let inputWeightGradientBuffer: MTLBuffer
    let inputBiasGradientBuffer: MTLBuffer
    let outputWeightGradientBuffer: MTLBuffer
    let outputBiasGradientBuffer: MTLBuffer
    let valueWeightGradientBuffer: MTLBuffer
    let valueBiasGradientBuffer: MTLBuffer
}

private struct PersistentAdamBuffers {
    let inputWeightM: MTLBuffer
    let inputWeightV: MTLBuffer
    let inputBiasM: MTLBuffer
    let inputBiasV: MTLBuffer
    let outputWeightM: MTLBuffer
    let outputWeightV: MTLBuffer
    let outputBiasM: MTLBuffer
    let outputBiasV: MTLBuffer
    let valueWeightM: MTLBuffer
    let valueWeightV: MTLBuffer
    let valueBiasM: MTLBuffer
    let valueBiasV: MTLBuffer
}
