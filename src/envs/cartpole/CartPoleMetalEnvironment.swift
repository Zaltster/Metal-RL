import Foundation
import Metal

final class CartPoleMetalEnvironment {
    let actionSpec = VectorActionSpec(dimensionsPerEnv: 1, minValue: -1.0, maxValue: 1.0)
    let observationSpec = VectorObservationSpec(elementsPerEnv: 4)

    let envCount: Int

    private let commandQueue: MTLCommandQueue
    private let stepPipeline: MTLComputePipelineState
    private let resetPipeline: MTLComputePipelineState
    private let outputPipeline: MTLComputePipelineState
    private let stateBuffer: MTLBuffer
    private let actionBuffer: MTLBuffer
    private let resetCountBuffer: MTLBuffer
    private let observationBuffer: MTLBuffer
    private let rewardBuffer: MTLBuffer
    private let doneBuffer: MTLBuffer
    private let stepParamsBuffer: MTLBuffer
    private let resetParamsBuffer: MTLBuffer

    private var stepParams: CartPoleParams
    private var resetParams: ResetParams

    init(
        device: MTLDevice,
        rootDir: String,
        envCount: Int,
        cartPoleParams: CartPoleParams,
        resetSeed: UInt32
    ) throws {
        if envCount != Int(cartPoleParams.envCount) {
            throw EnvProjectError.validationFailed(message: "envCount must match cartPoleParams.envCount.")
        }

        self.envCount = envCount
        let shaderPath = URL(fileURLWithPath: rootDir)
            .appending(path: "src/envs/cartpole/Shaders/cartpole_kernels.metal")
            .path()

        guard let commandQueue = device.makeCommandQueue() else {
            throw EnvProjectError.commandQueueUnavailable
        }
        self.commandQueue = commandQueue

        self.stepParams = cartPoleParams
        self.resetParams = ResetParams(envCount: UInt32(envCount), baseSeed: resetSeed)

        let library = try makeLibrary(device: device, shaderPath: shaderPath)
        self.stepPipeline = try makePipeline(device: device, library: library, name: "step_cartpole")
        self.resetPipeline = try makePipeline(device: device, library: library, name: "reset_done_cartpoles")
        self.outputPipeline = try makePipeline(device: device, library: library, name: "write_cartpole_outputs")

        let stateLength = MemoryLayout<CartPoleState>.stride * envCount
        let actionLength = MemoryLayout<Float>.stride * envCount
        let resetCountLength = MemoryLayout<UInt32>.stride * envCount
        let observationLength = MemoryLayout<Float>.stride * envCount * observationSpec.elementsPerEnv
        let rewardLength = MemoryLayout<Float>.stride * envCount
        let doneLength = MemoryLayout<UInt32>.stride * envCount

        guard let stateBuffer = device.makeBuffer(length: stateLength, options: .storageModeShared) else {
            throw EnvProjectError.bufferAllocationFailed("state")
        }
        guard let actionBuffer = device.makeBuffer(length: actionLength, options: .storageModeShared) else {
            throw EnvProjectError.bufferAllocationFailed("action")
        }
        guard let resetCountBuffer = device.makeBuffer(length: resetCountLength, options: .storageModeShared) else {
            throw EnvProjectError.bufferAllocationFailed("reset-count")
        }
        guard let observationBuffer = device.makeBuffer(length: observationLength, options: .storageModeShared) else {
            throw EnvProjectError.bufferAllocationFailed("observation")
        }
        guard let rewardBuffer = device.makeBuffer(length: rewardLength, options: .storageModeShared) else {
            throw EnvProjectError.bufferAllocationFailed("reward")
        }
        guard let doneBuffer = device.makeBuffer(length: doneLength, options: .storageModeShared) else {
            throw EnvProjectError.bufferAllocationFailed("done")
        }
        guard let stepParamsBuffer = device.makeBuffer(length: MemoryLayout<CartPoleParams>.stride, options: .storageModeShared) else {
            throw EnvProjectError.bufferAllocationFailed("step-params")
        }
        guard let resetParamsBuffer = device.makeBuffer(length: MemoryLayout<ResetParams>.stride, options: .storageModeShared) else {
            throw EnvProjectError.bufferAllocationFailed("reset-params")
        }

        self.stateBuffer = stateBuffer
        self.actionBuffer = actionBuffer
        self.resetCountBuffer = resetCountBuffer
        self.observationBuffer = observationBuffer
        self.rewardBuffer = rewardBuffer
        self.doneBuffer = doneBuffer
        self.stepParamsBuffer = stepParamsBuffer
        self.resetParamsBuffer = resetParamsBuffer

        writeValue(&self.stepParams, to: stepParamsBuffer)
        writeValue(&self.resetParams, to: resetParamsBuffer)
        let zeroResetCounts = Array(repeating: UInt32(0), count: envCount)
        copyArray(zeroResetCounts, to: resetCountBuffer)
    }

    func load(initialStates: [CartPoleState]) throws {
        if initialStates.count != envCount {
            throw EnvProjectError.validationFailed(message: "initialStates count must match envCount.")
        }
        copyArray(initialStates, to: stateBuffer)
        let zeroResetCounts = Array(repeating: UInt32(0), count: envCount)
        copyArray(zeroResetCounts, to: resetCountBuffer)
        try refreshOutputs()
    }

    func setResetSeed(_ seed: UInt32) {
        resetParams.baseSeed = seed
        writeValue(&resetParams, to: resetParamsBuffer)
    }

    func step(actions: [Float]) throws {
        if actions.count != envCount {
            throw EnvProjectError.validationFailed(message: "actions count must match envCount.")
        }
        copyArray(actions, to: actionBuffer)

        try runComputePass(commandQueue: commandQueue, pipeline: stepPipeline, count: envCount) { encoder in
            encoder.setBuffer(stateBuffer, offset: 0, index: 0)
            encoder.setBuffer(actionBuffer, offset: 0, index: 1)
            encoder.setBuffer(stepParamsBuffer, offset: 0, index: 2)
        }
        try refreshOutputs()
    }

    func resetDone() throws {
        try runComputePass(commandQueue: commandQueue, pipeline: resetPipeline, count: envCount) { encoder in
            encoder.setBuffer(stateBuffer, offset: 0, index: 0)
            encoder.setBuffer(resetCountBuffer, offset: 0, index: 1)
            encoder.setBuffer(resetParamsBuffer, offset: 0, index: 2)
        }
        try refreshOutputs()
    }

    func readStates() -> [CartPoleState] {
        readArray(from: stateBuffer, count: envCount)
    }

    func readResetCounts() -> [UInt32] {
        readArray(from: resetCountBuffer, count: envCount)
    }

    func readObservations() -> [Float] {
        readArray(from: observationBuffer, count: envCount * observationSpec.elementsPerEnv)
    }

    func readRewards() -> [Float] {
        readArray(from: rewardBuffer, count: envCount)
    }

    func readDones() -> [UInt32] {
        readArray(from: doneBuffer, count: envCount)
    }

    func readBatch() -> VectorEnvBatch {
        return VectorEnvBatch(
            observations: readObservations(),
            rewards: readRewards(),
            dones: readDones(),
            resetCounts: readResetCounts()
        )
    }

    private func refreshOutputs() throws {
        try runComputePass(commandQueue: commandQueue, pipeline: outputPipeline, count: envCount) { encoder in
            encoder.setBuffer(stateBuffer, offset: 0, index: 0)
            encoder.setBuffer(observationBuffer, offset: 0, index: 1)
            encoder.setBuffer(rewardBuffer, offset: 0, index: 2)
            encoder.setBuffer(doneBuffer, offset: 0, index: 3)
        }
    }
}
