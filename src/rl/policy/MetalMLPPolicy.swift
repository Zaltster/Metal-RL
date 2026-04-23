import Foundation
import Metal

struct MLPPolicyParams {
    var envCount: UInt32
    var observationDim: UInt32
    var hiddenDim: UInt32
    var actionDim: UInt32
    var minAction: Float
    var maxAction: Float
}

final class MetalMLPPolicy: VectorActorCriticPolicy {
    let envCount: Int
    let observationSpec: VectorObservationSpec
    let actionSpec: VectorActionSpec

    private let commandQueue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState
    private let observationBuffer: MTLBuffer
    private let inputWeightBuffer: MTLBuffer
    private let inputBiasBuffer: MTLBuffer
    private let outputWeightBuffer: MTLBuffer
    private let outputBiasBuffer: MTLBuffer
    private let valueWeightBuffer: MTLBuffer
    private let valueBiasBuffer: MTLBuffer
    private let actionBuffer: MTLBuffer
    private let valueBuffer: MTLBuffer
    private let paramsBuffer: MTLBuffer

    private var params: MLPPolicyParams

    init(
        device: MTLDevice,
        rootDir: String,
        envCount: Int,
        observationSpec: VectorObservationSpec,
        actionSpec: VectorActionSpec,
        inputWeights: [Float],
        inputBias: [Float],
        outputWeights: [Float],
        outputBias: [Float],
        valueWeights: [Float],
        valueBias: Float
    ) throws {
        let hiddenDim = inputBias.count

        if inputWeights.count != observationSpec.elementsPerEnv * hiddenDim {
            throw EnvProjectError.validationFailed(
                message: "MetalMLPPolicy input weight size mismatch: expected \(observationSpec.elementsPerEnv * hiddenDim), got \(inputWeights.count)."
            )
        }
        if outputWeights.count != actionSpec.dimensionsPerEnv * hiddenDim {
            throw EnvProjectError.validationFailed(
                message: "MetalMLPPolicy output weight size mismatch: expected \(actionSpec.dimensionsPerEnv * hiddenDim), got \(outputWeights.count)."
            )
        }
        if valueWeights.count != hiddenDim {
            throw EnvProjectError.validationFailed(
                message: "MetalMLPPolicy value weight size mismatch: expected \(hiddenDim), got \(valueWeights.count)."
            )
        }
        if outputBias.count != actionSpec.dimensionsPerEnv {
            throw EnvProjectError.validationFailed(
                message: "MetalMLPPolicy output bias size mismatch: expected \(actionSpec.dimensionsPerEnv), got \(outputBias.count)."
            )
        }

        self.envCount = envCount
        self.observationSpec = observationSpec
        self.actionSpec = actionSpec
        let shaderPath = URL(fileURLWithPath: rootDir)
            .appending(path: "src/rl/policy/Shaders/mlp_policy.metal")
            .path()
        self.params = MLPPolicyParams(
            envCount: UInt32(envCount),
            observationDim: UInt32(observationSpec.elementsPerEnv),
            hiddenDim: UInt32(hiddenDim),
            actionDim: UInt32(actionSpec.dimensionsPerEnv),
            minAction: actionSpec.minValue,
            maxAction: actionSpec.maxValue
        )

        guard let commandQueue = device.makeCommandQueue() else {
            throw EnvProjectError.commandQueueUnavailable
        }
        self.commandQueue = commandQueue

        let library = try makeLibrary(device: device, shaderPath: shaderPath)
        self.pipeline = try makePipeline(device: device, library: library, name: "mlp_policy_forward")

        let observationLength = MemoryLayout<Float>.stride * envCount * observationSpec.elementsPerEnv
        let inputWeightLength = MemoryLayout<Float>.stride * inputWeights.count
        let inputBiasLength = MemoryLayout<Float>.stride * inputBias.count
        let outputWeightLength = MemoryLayout<Float>.stride * outputWeights.count
        let outputBiasLength = MemoryLayout<Float>.stride * outputBias.count
        let valueWeightLength = MemoryLayout<Float>.stride * valueWeights.count
        let valueBiasLength = MemoryLayout<Float>.stride
        let actionLength = MemoryLayout<Float>.stride * envCount * actionSpec.dimensionsPerEnv
        let valueLength = MemoryLayout<Float>.stride * envCount

        guard let observationBuffer = device.makeBuffer(length: observationLength, options: .storageModeShared) else {
            throw EnvProjectError.bufferAllocationFailed("mlp-policy-observation")
        }
        guard let inputWeightBuffer = device.makeBuffer(length: inputWeightLength, options: .storageModeShared) else {
            throw EnvProjectError.bufferAllocationFailed("mlp-policy-input-weights")
        }
        guard let inputBiasBuffer = device.makeBuffer(length: inputBiasLength, options: .storageModeShared) else {
            throw EnvProjectError.bufferAllocationFailed("mlp-policy-input-bias")
        }
        guard let outputWeightBuffer = device.makeBuffer(length: outputWeightLength, options: .storageModeShared) else {
            throw EnvProjectError.bufferAllocationFailed("mlp-policy-output-weights")
        }
        guard let outputBiasBuffer = device.makeBuffer(length: outputBiasLength, options: .storageModeShared) else {
            throw EnvProjectError.bufferAllocationFailed("mlp-policy-output-bias")
        }
        guard let valueWeightBuffer = device.makeBuffer(length: valueWeightLength, options: .storageModeShared) else {
            throw EnvProjectError.bufferAllocationFailed("mlp-policy-value-weights")
        }
        guard let valueBiasBuffer = device.makeBuffer(length: valueBiasLength, options: .storageModeShared) else {
            throw EnvProjectError.bufferAllocationFailed("mlp-policy-value-bias")
        }
        guard let actionBuffer = device.makeBuffer(length: actionLength, options: .storageModeShared) else {
            throw EnvProjectError.bufferAllocationFailed("mlp-policy-actions")
        }
        guard let valueBuffer = device.makeBuffer(length: valueLength, options: .storageModeShared) else {
            throw EnvProjectError.bufferAllocationFailed("mlp-policy-values")
        }
        guard let paramsBuffer = device.makeBuffer(length: MemoryLayout<MLPPolicyParams>.stride, options: .storageModeShared) else {
            throw EnvProjectError.bufferAllocationFailed("mlp-policy-params")
        }

        self.observationBuffer = observationBuffer
        self.inputWeightBuffer = inputWeightBuffer
        self.inputBiasBuffer = inputBiasBuffer
        self.outputWeightBuffer = outputWeightBuffer
        self.outputBiasBuffer = outputBiasBuffer
        self.valueWeightBuffer = valueWeightBuffer
        self.valueBiasBuffer = valueBiasBuffer
        self.actionBuffer = actionBuffer
        self.valueBuffer = valueBuffer
        self.paramsBuffer = paramsBuffer

        copyArray(inputWeights, to: inputWeightBuffer)
        copyArray(inputBias, to: inputBiasBuffer)
        copyArray(outputWeights, to: outputWeightBuffer)
        copyArray(outputBias, to: outputBiasBuffer)
        copyArray(valueWeights, to: valueWeightBuffer)
        copyArray([valueBias], to: valueBiasBuffer)
        writeValue(&self.params, to: paramsBuffer)
    }

    func evaluate(
        for observations: [Float],
        envCount: Int,
        observationSpec: VectorObservationSpec,
        actionSpec: VectorActionSpec
    ) throws -> PolicyValueOutputs {
        if envCount != self.envCount {
            throw EnvProjectError.validationFailed(
                message: "MetalMLPPolicy envCount mismatch: expected \(self.envCount), got \(envCount)."
            )
        }
        if observationSpec.elementsPerEnv != self.observationSpec.elementsPerEnv {
            throw EnvProjectError.validationFailed(
                message: "MetalMLPPolicy observationSpec mismatch: expected \(self.observationSpec.elementsPerEnv), got \(observationSpec.elementsPerEnv)."
            )
        }
        if actionSpec.dimensionsPerEnv != self.actionSpec.dimensionsPerEnv ||
            actionSpec.minValue != self.actionSpec.minValue ||
            actionSpec.maxValue != self.actionSpec.maxValue {
            throw EnvProjectError.validationFailed(
                message: "MetalMLPPolicy actionSpec mismatch."
            )
        }
        if observations.count != envCount * observationSpec.elementsPerEnv {
            throw EnvProjectError.validationFailed(
                message: "MetalMLPPolicy observation size mismatch: expected \(envCount * observationSpec.elementsPerEnv), got \(observations.count)."
            )
        }

        copyArray(observations, to: observationBuffer)

        try runComputePass(
            commandQueue: commandQueue,
            pipeline: pipeline,
            count: envCount
        ) { encoder in
            encoder.setBuffer(observationBuffer, offset: 0, index: 0)
            encoder.setBuffer(inputWeightBuffer, offset: 0, index: 1)
            encoder.setBuffer(inputBiasBuffer, offset: 0, index: 2)
            encoder.setBuffer(outputWeightBuffer, offset: 0, index: 3)
            encoder.setBuffer(outputBiasBuffer, offset: 0, index: 4)
            encoder.setBuffer(valueWeightBuffer, offset: 0, index: 5)
            encoder.setBuffer(valueBiasBuffer, offset: 0, index: 6)
            encoder.setBuffer(actionBuffer, offset: 0, index: 7)
            encoder.setBuffer(valueBuffer, offset: 0, index: 8)
            encoder.setBuffer(paramsBuffer, offset: 0, index: 9)
        }

        return PolicyValueOutputs(
            actions: readArray(from: actionBuffer, count: envCount * actionSpec.dimensionsPerEnv),
            values: readArray(from: valueBuffer, count: envCount)
        )
    }
}

extension MetalMLPPolicy: VectorGaussianActorCriticPolicy {
    func evaluateGaussian(
        for observations: [Float],
        taking actions: [Float]?,
        envCount: Int,
        observationSpec: VectorObservationSpec,
        actionSpec: VectorActionSpec
    ) throws -> GaussianPolicyOutputs {
        let base = try evaluate(
            for: observations,
            envCount: envCount,
            observationSpec: observationSpec,
            actionSpec: actionSpec
        )
        let actionDim = actionSpec.dimensionsPerEnv
        let chosenActions = actions ?? base.actions

        if chosenActions.count != envCount * actionDim {
            throw EnvProjectError.validationFailed(
                message: "MetalMLPPolicy chosen-action size mismatch: expected \(envCount * actionDim), got \(chosenActions.count)."
            )
        }

        let logStd = Array(repeating: Float(-0.35), count: actionDim)
        let entropies = Array(repeating: gaussianEntropy(logStd: logStd), count: envCount)
        var logProbs = Array(repeating: Float.zero, count: envCount)

        for envIndex in 0..<envCount {
            let actionBase = envIndex * actionDim
            let actionSlice = Array(chosenActions[actionBase..<(actionBase + actionDim)])
            let meanSlice = Array(base.actions[actionBase..<(actionBase + actionDim)])
            logProbs[envIndex] = gaussianLogProb(action: actionSlice, mean: meanSlice, logStd: logStd)
        }

        return GaussianPolicyOutputs(
            actions: chosenActions,
            actionMeans: base.actions,
            values: base.values,
            logProbs: logProbs,
            entropies: entropies,
            logStd: logStd
        )
    }
}

func makeReferenceMetalMLPPolicy(
    device: MTLDevice,
    rootDir: String,
    envCount: Int,
    observationSpec: VectorObservationSpec,
    actionSpec: VectorActionSpec
) throws -> MetalMLPPolicy {
    let cpuPolicy = try makeReferenceMLPPolicy(observationSpec: observationSpec, actionSpec: actionSpec)
    return try MetalMLPPolicy(
        device: device,
        rootDir: rootDir,
        envCount: envCount,
        observationSpec: observationSpec,
        actionSpec: actionSpec,
        inputWeights: cpuPolicy.inputWeights,
        inputBias: cpuPolicy.inputBias,
        outputWeights: cpuPolicy.outputWeights,
        outputBias: cpuPolicy.outputBias,
        valueWeights: cpuPolicy.valueWeights,
        valueBias: cpuPolicy.valueBias
    )
}

func makeAlternateMetalMLPPolicy(
    device: MTLDevice,
    rootDir: String,
    envCount: Int,
    observationSpec: VectorObservationSpec,
    actionSpec: VectorActionSpec
) throws -> MetalMLPPolicy {
    let cpuPolicy = try makeAlternateMLPPolicy(observationSpec: observationSpec, actionSpec: actionSpec)
    return try MetalMLPPolicy(
        device: device,
        rootDir: rootDir,
        envCount: envCount,
        observationSpec: observationSpec,
        actionSpec: actionSpec,
        inputWeights: cpuPolicy.inputWeights,
        inputBias: cpuPolicy.inputBias,
        outputWeights: cpuPolicy.outputWeights,
        outputBias: cpuPolicy.outputBias,
        valueWeights: cpuPolicy.valueWeights,
        valueBias: cpuPolicy.valueBias
    )
}
