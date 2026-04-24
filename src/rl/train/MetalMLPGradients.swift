import Foundation
import Metal

struct MLPGradientParams {
    var sampleCount: UInt32
    var observationDim: UInt32
    var hiddenDim: UInt32
    var actionDim: UInt32
    var clipEpsilon: Float
    var valueCoefficient: Float
}

struct MLPGradientReductionParams {
    var sampleCount: UInt32
    var parameterCount: UInt32
}

struct MLPSGDUpdateParams {
    var learningRate: Float
    var parameterCount: UInt32
}

struct MLPAdamUpdateParams {
    var learningRate: Float
    var beta1: Float
    var beta2: Float
    var epsilon: Float
    var biasCorrection1: Float
    var biasCorrection2: Float
    var parameterCount: UInt32
}

struct MetalOptimizerStepResult {
    let model: TrainableMLPActorCritic
    let parameterDeltaL1: Float
}

private struct MetalMLPGradientBuffers {
    let inputWeightBuffer: MTLBuffer
    let inputBiasBuffer: MTLBuffer
    let outputWeightBuffer: MTLBuffer
    let outputBiasBuffer: MTLBuffer
    let valueWeightBuffer: MTLBuffer
    let valueBiasBuffer: MTLBuffer

    let inputWeightGradientBuffer: MTLBuffer
    let inputBiasGradientBuffer: MTLBuffer
    let outputWeightGradientBuffer: MTLBuffer
    let outputBiasGradientBuffer: MTLBuffer
    let valueWeightGradientBuffer: MTLBuffer
    let valueBiasGradientBuffer: MTLBuffer

    let inputWeightCount: Int
    let inputBiasCount: Int
    let outputWeightCount: Int
    let outputBiasCount: Int
    let valueWeightCount: Int
}

final class MetalMLPGradientComputer {
    private let commandQueue: MTLCommandQueue
    private let perSamplePipeline: MTLComputePipelineState
    private let reductionPipeline: MTLComputePipelineState
    private let sgdUpdatePipeline: MTLComputePipelineState

    init(device: MTLDevice, rootDir: String) throws {
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
    }

    func computeGradients(
        model: TrainableMLPActorCritic,
        batch: PPOBatch,
        observationSpec: VectorObservationSpec,
        actionSpec: VectorActionSpec,
        ppoConfig: PPOConfig
    ) throws -> MLPGradients {
        let buffers = try computeReducedGradientBuffers(
            model: model,
            batch: batch,
            observationSpec: observationSpec,
            actionSpec: actionSpec,
            ppoConfig: ppoConfig
        )

        var gradients = MLPGradients(model: model)
        gradients.inputWeights = readArray(from: buffers.inputWeightGradientBuffer, count: buffers.inputWeightCount)
        gradients.inputBias = readArray(from: buffers.inputBiasGradientBuffer, count: buffers.inputBiasCount)
        gradients.outputWeights = readArray(from: buffers.outputWeightGradientBuffer, count: buffers.outputWeightCount)
        gradients.outputBias = readArray(from: buffers.outputBiasGradientBuffer, count: buffers.outputBiasCount)
        gradients.valueWeights = readArray(from: buffers.valueWeightGradientBuffer, count: buffers.valueWeightCount)
        gradients.valueBias = readArray(from: buffers.valueBiasGradientBuffer, count: 1)[0]
        return gradients
    }

    func applySGDStep(
        model: TrainableMLPActorCritic,
        batch: PPOBatch,
        observationSpec: VectorObservationSpec,
        actionSpec: VectorActionSpec,
        ppoConfig: PPOConfig,
        sgdConfig: SGDConfig
    ) throws -> MetalOptimizerStepResult {
        if !sgdConfig.learningRate.isFinite || sgdConfig.learningRate <= 0.0 {
            throw EnvProjectError.validationFailed(
                message: "MetalMLPGradientComputer requires a positive finite SGD learning rate."
            )
        }

        let buffers = try computeReducedGradientBuffers(
            model: model,
            batch: batch,
            observationSpec: observationSpec,
            actionSpec: actionSpec,
            ppoConfig: ppoConfig
        )

        for group in parameterGradientGroups(buffers: buffers) {
            try applySGDUpdate(
                parameterBuffer: group.parameterBuffer,
                gradientBuffer: group.gradientBuffer,
                parameterCount: group.parameterCount,
                learningRate: sgdConfig.learningRate
            )
        }

        let updatedModel = modelFromBuffers(buffers, logStd: model.logStd)

        return MetalOptimizerStepResult(
            model: updatedModel,
            parameterDeltaL1: parameterDeltaL1(before: model, after: updatedModel)
        )
    }

    private func computeReducedGradientBuffers(
        model: TrainableMLPActorCritic,
        batch: PPOBatch,
        observationSpec: VectorObservationSpec,
        actionSpec: VectorActionSpec,
        ppoConfig: PPOConfig
    ) throws -> MetalMLPGradientBuffers {
        let obsDim = observationSpec.elementsPerEnv
        let actionDim = actionSpec.dimensionsPerEnv
        let hiddenDim = model.hiddenDim

        try validateModel(model, observationDim: obsDim, actionDim: actionDim)
        if batch.sampleCount <= 0 {
            throw EnvProjectError.validationFailed(message: "MetalMLPGradientComputer requires at least one batch sample.")
        }
        if obsDim != batch.observationDim || actionDim != batch.actionDim {
            throw EnvProjectError.validationFailed(message: "MetalMLPGradientComputer batch spec mismatch.")
        }
        if hiddenDim > 64 || obsDim > 64 || actionDim > 8 {
            throw EnvProjectError.validationFailed(
                message: "MetalMLPGradientComputer dims exceed current kernel limits: obs=\(obsDim), hidden=\(hiddenDim), action=\(actionDim)."
            )
        }

        let device = perSamplePipeline.device
        let sampleCount = batch.sampleCount
        let inputWeightCount = model.inputWeights.count
        let inputBiasCount = model.inputBias.count
        let outputWeightCount = model.outputWeights.count
        let outputBiasCount = model.outputBias.count
        let valueWeightCount = model.valueWeights.count

        let observationBuffer = try makeFloatBuffer(device: device, values: batch.observations, name: "gradient-observations")
        let actionBuffer = try makeFloatBuffer(device: device, values: batch.actions, name: "gradient-actions")
        let oldLogProbBuffer = try makeFloatBuffer(device: device, values: batch.oldLogProbs, name: "gradient-old-logprobs")
        let advantageBuffer = try makeFloatBuffer(device: device, values: batch.advantages, name: "gradient-advantages")
        let returnBuffer = try makeFloatBuffer(device: device, values: batch.returns, name: "gradient-returns")
        let inputWeightBuffer = try makeFloatBuffer(device: device, values: model.inputWeights, name: "gradient-input-weights")
        let inputBiasBuffer = try makeFloatBuffer(device: device, values: model.inputBias, name: "gradient-input-bias")
        let outputWeightBuffer = try makeFloatBuffer(device: device, values: model.outputWeights, name: "gradient-output-weights")
        let outputBiasBuffer = try makeFloatBuffer(device: device, values: model.outputBias, name: "gradient-output-bias")
        let valueWeightBuffer = try makeFloatBuffer(device: device, values: model.valueWeights, name: "gradient-value-weights")
        let valueBiasBuffer = try makeFloatBuffer(device: device, values: [model.valueBias], name: "gradient-value-bias")
        let logStdBuffer = try makeFloatBuffer(device: device, values: model.logStd, name: "gradient-log-std")

        let inputWeightPerSampleGradientBuffer = try makeZeroedFloatBuffer(
            device: device,
            count: sampleCount * inputWeightCount,
            name: "gradient-input-weight-per-sample"
        )
        let inputBiasPerSampleGradientBuffer = try makeZeroedFloatBuffer(
            device: device,
            count: sampleCount * inputBiasCount,
            name: "gradient-input-bias-per-sample"
        )
        let outputWeightPerSampleGradientBuffer = try makeZeroedFloatBuffer(
            device: device,
            count: sampleCount * outputWeightCount,
            name: "gradient-output-weight-per-sample"
        )
        let outputBiasPerSampleGradientBuffer = try makeZeroedFloatBuffer(
            device: device,
            count: sampleCount * outputBiasCount,
            name: "gradient-output-bias-per-sample"
        )
        let valueWeightPerSampleGradientBuffer = try makeZeroedFloatBuffer(
            device: device,
            count: sampleCount * valueWeightCount,
            name: "gradient-value-weight-per-sample"
        )
        let valueBiasPerSampleGradientBuffer = try makeZeroedFloatBuffer(
            device: device,
            count: sampleCount,
            name: "gradient-value-bias-per-sample"
        )

        var params = MLPGradientParams(
            sampleCount: UInt32(sampleCount),
            observationDim: UInt32(obsDim),
            hiddenDim: UInt32(hiddenDim),
            actionDim: UInt32(actionDim),
            clipEpsilon: ppoConfig.clipEpsilon,
            valueCoefficient: ppoConfig.valueCoefficient
        )
        guard let paramsBuffer = device.makeBuffer(length: MemoryLayout<MLPGradientParams>.stride, options: .storageModeShared) else {
            throw EnvProjectError.bufferAllocationFailed("gradient-params")
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

        return MetalMLPGradientBuffers(
            inputWeightBuffer: inputWeightBuffer,
            inputBiasBuffer: inputBiasBuffer,
            outputWeightBuffer: outputWeightBuffer,
            outputBiasBuffer: outputBiasBuffer,
            valueWeightBuffer: valueWeightBuffer,
            valueBiasBuffer: valueBiasBuffer,
            inputWeightGradientBuffer: try reducePerSampleGradientsOnGPU(
                inputWeightPerSampleGradientBuffer,
                sampleCount: sampleCount,
                parameterCount: inputWeightCount,
                name: "gradient-input-weight-reduced"
            ),
            inputBiasGradientBuffer: try reducePerSampleGradientsOnGPU(
                inputBiasPerSampleGradientBuffer,
                sampleCount: sampleCount,
                parameterCount: inputBiasCount,
                name: "gradient-input-bias-reduced"
            ),
            outputWeightGradientBuffer: try reducePerSampleGradientsOnGPU(
                outputWeightPerSampleGradientBuffer,
                sampleCount: sampleCount,
                parameterCount: outputWeightCount,
                name: "gradient-output-weight-reduced"
            ),
            outputBiasGradientBuffer: try reducePerSampleGradientsOnGPU(
                outputBiasPerSampleGradientBuffer,
                sampleCount: sampleCount,
                parameterCount: outputBiasCount,
                name: "gradient-output-bias-reduced"
            ),
            valueWeightGradientBuffer: try reducePerSampleGradientsOnGPU(
                valueWeightPerSampleGradientBuffer,
                sampleCount: sampleCount,
                parameterCount: valueWeightCount,
                name: "gradient-value-weight-reduced"
            ),
            valueBiasGradientBuffer: try reducePerSampleGradientsOnGPU(
                valueBiasPerSampleGradientBuffer,
                sampleCount: sampleCount,
                parameterCount: 1,
                name: "gradient-value-bias-reduced"
            ),
            inputWeightCount: inputWeightCount,
            inputBiasCount: inputBiasCount,
            outputWeightCount: outputWeightCount,
            outputBiasCount: outputBiasCount,
            valueWeightCount: valueWeightCount
        )
    }

    private func makeFloatBuffer(device: MTLDevice, values: [Float], name: String) throws -> MTLBuffer {
        guard let buffer = device.makeBuffer(length: MemoryLayout<Float>.stride * values.count, options: .storageModeShared) else {
            throw EnvProjectError.bufferAllocationFailed(name)
        }
        copyArray(values, to: buffer)
        return buffer
    }

    private func makeZeroedFloatBuffer(device: MTLDevice, count: Int, name: String) throws -> MTLBuffer {
        try makeFloatBuffer(device: device, values: Array(repeating: 0.0, count: count), name: name)
    }

    private func reducePerSampleGradientsOnGPU(
        _ perSampleBuffer: MTLBuffer,
        sampleCount: Int,
        parameterCount: Int,
        name: String
    ) throws -> MTLBuffer {
        let device = reductionPipeline.device
        let reducedBuffer = try makeZeroedFloatBuffer(device: device, count: parameterCount, name: name)
        var params = MLPGradientReductionParams(
            sampleCount: UInt32(sampleCount),
            parameterCount: UInt32(parameterCount)
        )
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
        var params = MLPSGDUpdateParams(
            learningRate: learningRate,
            parameterCount: UInt32(parameterCount)
        )
        guard let paramsBuffer = device.makeBuffer(length: MemoryLayout<MLPSGDUpdateParams>.stride, options: .storageModeShared) else {
            throw EnvProjectError.bufferAllocationFailed("sgd-update-params")
        }
        writeValue(&params, to: paramsBuffer)

        try runComputePass(commandQueue: commandQueue, pipeline: sgdUpdatePipeline, count: parameterCount) { encoder in
            encoder.setBuffer(parameterBuffer, offset: 0, index: 0)
            encoder.setBuffer(gradientBuffer, offset: 0, index: 1)
            encoder.setBuffer(paramsBuffer, offset: 0, index: 2)
        }
    }

    private func validateModel(
        _ model: TrainableMLPActorCritic,
        observationDim: Int,
        actionDim: Int
    ) throws {
        let hiddenDim = model.hiddenDim
        if hiddenDim <= 0 || observationDim <= 0 || actionDim <= 0 {
            throw EnvProjectError.validationFailed(message: "MetalMLPGradientComputer dimensions must be positive.")
        }
        if model.inputWeights.count != hiddenDim * observationDim {
            throw EnvProjectError.validationFailed(
                message: "MetalMLPGradientComputer input-weight size mismatch: expected \(hiddenDim * observationDim), got \(model.inputWeights.count)."
            )
        }
        if model.outputWeights.count != actionDim * hiddenDim {
            throw EnvProjectError.validationFailed(
                message: "MetalMLPGradientComputer output-weight size mismatch: expected \(actionDim * hiddenDim), got \(model.outputWeights.count)."
            )
        }
        if model.outputBias.count != actionDim {
            throw EnvProjectError.validationFailed(
                message: "MetalMLPGradientComputer output-bias size mismatch: expected \(actionDim), got \(model.outputBias.count)."
            )
        }
        if model.valueWeights.count != hiddenDim {
            throw EnvProjectError.validationFailed(
                message: "MetalMLPGradientComputer value-weight size mismatch: expected \(hiddenDim), got \(model.valueWeights.count)."
            )
        }
        if model.logStd.count != actionDim {
            throw EnvProjectError.validationFailed(
                message: "MetalMLPGradientComputer logStd size mismatch: expected \(actionDim), got \(model.logStd.count)."
            )
        }
    }

    private func parameterGradientGroups(
        buffers: MetalMLPGradientBuffers
    ) -> [(parameterBuffer: MTLBuffer, gradientBuffer: MTLBuffer, parameterCount: Int)] {
        [
            (buffers.inputWeightBuffer, buffers.inputWeightGradientBuffer, buffers.inputWeightCount),
            (buffers.inputBiasBuffer, buffers.inputBiasGradientBuffer, buffers.inputBiasCount),
            (buffers.outputWeightBuffer, buffers.outputWeightGradientBuffer, buffers.outputWeightCount),
            (buffers.outputBiasBuffer, buffers.outputBiasGradientBuffer, buffers.outputBiasCount),
            (buffers.valueWeightBuffer, buffers.valueWeightGradientBuffer, buffers.valueWeightCount),
            (buffers.valueBiasBuffer, buffers.valueBiasGradientBuffer, 1),
        ]
    }

    private func modelFromBuffers(
        _ buffers: MetalMLPGradientBuffers,
        logStd: [Float]
    ) -> TrainableMLPActorCritic {
        TrainableMLPActorCritic(
            policy: MLPPolicy(
                inputWeights: readArray(from: buffers.inputWeightBuffer, count: buffers.inputWeightCount),
                inputBias: readArray(from: buffers.inputBiasBuffer, count: buffers.inputBiasCount),
                outputWeights: readArray(from: buffers.outputWeightBuffer, count: buffers.outputWeightCount),
                outputBias: readArray(from: buffers.outputBiasBuffer, count: buffers.outputBiasCount),
                valueWeights: readArray(from: buffers.valueWeightBuffer, count: buffers.valueWeightCount),
                valueBias: readArray(from: buffers.valueBiasBuffer, count: 1)[0]
            ),
            logStd: logStd
        )
    }

    private func parameterDeltaL1(before: TrainableMLPActorCritic, after: TrainableMLPActorCritic) -> Float {
        l1Delta(before.inputWeights, after.inputWeights) +
            l1Delta(before.inputBias, after.inputBias) +
            l1Delta(before.outputWeights, after.outputWeights) +
            l1Delta(before.outputBias, after.outputBias) +
            l1Delta(before.valueWeights, after.valueWeights) +
            abs(before.valueBias - after.valueBias)
    }

    private func l1Delta(_ lhs: [Float], _ rhs: [Float]) -> Float {
        zip(lhs, rhs).reduce(Float.zero) { partial, values in
            partial + abs(values.0 - values.1)
        }
    }
}
