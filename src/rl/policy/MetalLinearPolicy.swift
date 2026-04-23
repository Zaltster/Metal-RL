import Foundation
import Metal

struct LinearPolicyParams {
    var envCount: UInt32
    var observationDim: UInt32
    var actionDim: UInt32
    var minAction: Float
    var maxAction: Float
}

final class MetalLinearPolicy: VectorPolicy {
    let envCount: Int
    let observationSpec: VectorObservationSpec
    let actionSpec: VectorActionSpec

    private let commandQueue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState
    private let observationBuffer: MTLBuffer
    private let weightBuffer: MTLBuffer
    private let biasBuffer: MTLBuffer
    private let outputBuffer: MTLBuffer
    private let paramsBuffer: MTLBuffer

    private var params: LinearPolicyParams

    init(
        device: MTLDevice,
        rootDir: String,
        envCount: Int,
        observationSpec: VectorObservationSpec,
        actionSpec: VectorActionSpec,
        weights: [Float],
        bias: [Float]
    ) throws {
        if weights.count != observationSpec.elementsPerEnv * actionSpec.dimensionsPerEnv {
            throw EnvProjectError.validationFailed(
                message: "MetalLinearPolicy weight size mismatch: expected \(observationSpec.elementsPerEnv * actionSpec.dimensionsPerEnv), got \(weights.count)."
            )
        }
        if bias.count != actionSpec.dimensionsPerEnv {
            throw EnvProjectError.validationFailed(
                message: "MetalLinearPolicy bias size mismatch: expected \(actionSpec.dimensionsPerEnv), got \(bias.count)."
            )
        }

        self.envCount = envCount
        self.observationSpec = observationSpec
        self.actionSpec = actionSpec
        let shaderPath = URL(fileURLWithPath: rootDir)
            .appending(path: "src/rl/policy/Shaders/linear_policy.metal")
            .path()
        self.params = LinearPolicyParams(
            envCount: UInt32(envCount),
            observationDim: UInt32(observationSpec.elementsPerEnv),
            actionDim: UInt32(actionSpec.dimensionsPerEnv),
            minAction: actionSpec.minValue,
            maxAction: actionSpec.maxValue
        )

        guard let commandQueue = device.makeCommandQueue() else {
            throw EnvProjectError.commandQueueUnavailable
        }
        self.commandQueue = commandQueue

        let library = try makeLibrary(device: device, shaderPath: shaderPath)
        self.pipeline = try makePipeline(device: device, library: library, name: "linear_policy_forward")

        let observationLength = MemoryLayout<Float>.stride * envCount * observationSpec.elementsPerEnv
        let weightLength = MemoryLayout<Float>.stride * weights.count
        let biasLength = MemoryLayout<Float>.stride * bias.count
        let outputLength = MemoryLayout<Float>.stride * envCount * actionSpec.dimensionsPerEnv

        guard let observationBuffer = device.makeBuffer(length: observationLength, options: .storageModeShared) else {
            throw EnvProjectError.bufferAllocationFailed("policy-observation")
        }
        guard let weightBuffer = device.makeBuffer(length: weightLength, options: .storageModeShared) else {
            throw EnvProjectError.bufferAllocationFailed("policy-weights")
        }
        guard let biasBuffer = device.makeBuffer(length: biasLength, options: .storageModeShared) else {
            throw EnvProjectError.bufferAllocationFailed("policy-bias")
        }
        guard let outputBuffer = device.makeBuffer(length: outputLength, options: .storageModeShared) else {
            throw EnvProjectError.bufferAllocationFailed("policy-output")
        }
        guard let paramsBuffer = device.makeBuffer(length: MemoryLayout<LinearPolicyParams>.stride, options: .storageModeShared) else {
            throw EnvProjectError.bufferAllocationFailed("policy-params")
        }

        self.observationBuffer = observationBuffer
        self.weightBuffer = weightBuffer
        self.biasBuffer = biasBuffer
        self.outputBuffer = outputBuffer
        self.paramsBuffer = paramsBuffer

        copyArray(weights, to: weightBuffer)
        copyArray(bias, to: biasBuffer)
        writeValue(&self.params, to: paramsBuffer)
    }

    func actions(
        for observations: [Float],
        envCount: Int,
        observationSpec: VectorObservationSpec,
        actionSpec: VectorActionSpec
    ) throws -> [Float] {
        if envCount != self.envCount {
            throw EnvProjectError.validationFailed(
                message: "MetalLinearPolicy envCount mismatch: expected \(self.envCount), got \(envCount)."
            )
        }
        if observationSpec.elementsPerEnv != self.observationSpec.elementsPerEnv {
            throw EnvProjectError.validationFailed(
                message: "MetalLinearPolicy observationSpec mismatch: expected \(self.observationSpec.elementsPerEnv), got \(observationSpec.elementsPerEnv)."
            )
        }
        if actionSpec.dimensionsPerEnv != self.actionSpec.dimensionsPerEnv ||
            actionSpec.minValue != self.actionSpec.minValue ||
            actionSpec.maxValue != self.actionSpec.maxValue {
            throw EnvProjectError.validationFailed(
                message: "MetalLinearPolicy actionSpec mismatch."
            )
        }
        if observations.count != envCount * observationSpec.elementsPerEnv {
            throw EnvProjectError.validationFailed(
                message: "MetalLinearPolicy observation size mismatch: expected \(envCount * observationSpec.elementsPerEnv), got \(observations.count)."
            )
        }

        copyArray(observations, to: observationBuffer)

        try runComputePass(
            commandQueue: commandQueue,
            pipeline: pipeline,
            count: envCount * actionSpec.dimensionsPerEnv
        ) { encoder in
            encoder.setBuffer(observationBuffer, offset: 0, index: 0)
            encoder.setBuffer(weightBuffer, offset: 0, index: 1)
            encoder.setBuffer(biasBuffer, offset: 0, index: 2)
            encoder.setBuffer(outputBuffer, offset: 0, index: 3)
            encoder.setBuffer(paramsBuffer, offset: 0, index: 4)
        }

        return readArray(from: outputBuffer, count: envCount * actionSpec.dimensionsPerEnv)
    }
}

func makeReferenceMetalLinearPolicy(
    device: MTLDevice,
    rootDir: String,
    envCount: Int,
    observationSpec: VectorObservationSpec,
    actionSpec: VectorActionSpec
) throws -> MetalLinearPolicy {
    let cpuPolicy = try makeReferenceLinearPolicy(observationSpec: observationSpec, actionSpec: actionSpec)
    return try MetalLinearPolicy(
        device: device,
        rootDir: rootDir,
        envCount: envCount,
        observationSpec: observationSpec,
        actionSpec: actionSpec,
        weights: cpuPolicy.weights,
        bias: cpuPolicy.bias
    )
}

func makeAlternateMetalLinearPolicy(
    device: MTLDevice,
    rootDir: String,
    envCount: Int,
    observationSpec: VectorObservationSpec,
    actionSpec: VectorActionSpec
) throws -> MetalLinearPolicy {
    let cpuPolicy = try makeAlternateLinearPolicy(observationSpec: observationSpec, actionSpec: actionSpec)
    return try MetalLinearPolicy(
        device: device,
        rootDir: rootDir,
        envCount: envCount,
        observationSpec: observationSpec,
        actionSpec: actionSpec,
        weights: cpuPolicy.weights,
        bias: cpuPolicy.bias
    )
}
