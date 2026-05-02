import Foundation
import Metal

final class HumanoidMetalEnvironment {
    let envCount: Int
    let linkCount: Int
    let jointCount: Int
    let dofCount: Int
    let observationSpec: VectorObservationSpec
    let actionSpec: VectorActionSpec
    let linkNames: [String]
    let parentLinkIndices: [Int]
    let selfCollisionPairCount: Int
    let selfCollisionPairs: [(Int, Int)]
    let warnings: [String]

    private let commandQueue: MTLCommandQueue
    private let resetPipeline: MTLComputePipelineState
    private let stepPipeline: MTLComputePipelineState
    private let jointMotorPipeline: MTLComputePipelineState
    private let freeBodyPipeline: MTLComputePipelineState
    private let constraintPipeline: MTLComputePipelineState
    private let anchorErrorPipeline: MTLComputePipelineState
    private let groundContactPipeline: MTLComputePipelineState
    private let selfContactPipeline: MTLComputePipelineState
    private let groundContactSolvePipeline: MTLComputePipelineState
    private let rootSyncPipeline: MTLComputePipelineState
    private let fkPipeline: MTLComputePipelineState
    private let outputPipeline: MTLComputePipelineState

    private let linkConstantsBuffer: MTLBuffer
    private let collisionConstantsBuffer: MTLBuffer
    private let collisionPairConstantsBuffer: MTLBuffer
    private let jointConstantsBuffer: MTLBuffer
    private let paramsBuffer: MTLBuffer
    private let rootPositionBuffer: MTLBuffer
    private let rootRotationBuffer: MTLBuffer
    private let jointPositionBuffer: MTLBuffer
    private let jointVelocityBuffer: MTLBuffer
    private let actionBuffer: MTLBuffer
    private let linkPositionBuffer: MTLBuffer
    private let linkRotationBuffer: MTLBuffer
    private let linkLinearVelocityBuffer: MTLBuffer
    private let linkAngularVelocityBuffer: MTLBuffer
    private let jointAnchorImpulseBuffer: MTLBuffer
    private let jointAngularImpulseBuffer: MTLBuffer
    private let jointMotorImpulseBuffer: MTLBuffer
    private let jointLimitImpulseBuffer: MTLBuffer
    private let solverDiagnosticsBuffer: MTLBuffer
    private let jointAnchorErrorBuffer: MTLBuffer
    private let contactPointBuffer: MTLBuffer
    private let contactNormalBuffer: MTLBuffer
    private let contactPenetrationBuffer: MTLBuffer
    private let selfContactPointBuffer: MTLBuffer
    private let selfContactNormalBuffer: MTLBuffer
    private let selfContactPenetrationBuffer: MTLBuffer
    private let observationBuffer: MTLBuffer
    private let rewardBuffer: MTLBuffer
    private let doneBuffer: MTLBuffer
    private let resetCountBuffer: MTLBuffer
    private var params: HumanoidEnvParams

    init(
        device: MTLDevice,
        rootDir: String,
        envCount: Int,
        specURL: URL,
        dt: Float = 0.016666668,
        rootHeightMin: Float = 0.45
    ) throws {
        let spec = try loadHumanoidRobotSpec(from: specURL)
        let report = try validateHumanoidRobotSpec(spec)
        let layout = report.dofLayout
        let linkConstants = try makeHumanoidLinkConstants(spec: spec)
        let collisionConstants = makeHumanoidCollisionConstants(spec: spec)
        let collisionPairs = makeHumanoidCollisionPairConstants(linkCount: spec.links.count)
        let jointConstants = try makeHumanoidJointConstants(spec: spec, layout: layout)
        let defaultJointPositions = try makeHumanoidDefaultJointPositions(spec: spec, layout: layout)

        self.envCount = envCount
        self.linkCount = spec.links.count
        self.jointCount = spec.joints.count
        self.dofCount = layout.totalDoFs
        self.observationSpec = VectorObservationSpec(elementsPerEnv: 7 + 2 * layout.totalDoFs)
        self.actionSpec = VectorActionSpec(dimensionsPerEnv: layout.totalDoFs, minValue: -1.0, maxValue: 1.0)
        self.linkNames = spec.links.map(\.name)
        self.selfCollisionPairCount = collisionPairs.count
        self.selfCollisionPairs = collisionPairs.map { (Int($0.linkA), Int($0.linkB)) }
        self.parentLinkIndices = spec.joints.map { joint in
            if joint.type == .free {
                return -1
            }
            return spec.links.firstIndex(where: { $0.name == joint.parent_link }) ?? -1
        }
        self.warnings = report.warnings
        self.params = HumanoidEnvParams(
            envCount: UInt32(envCount),
            linkCount: UInt32(spec.links.count),
            selfCollisionPairCount: UInt32(collisionPairs.count),
            jointCount: UInt32(spec.joints.count),
            dofCount: UInt32(layout.totalDoFs),
            observationDim: UInt32(7 + 2 * layout.totalDoFs),
            dt: dt,
            rootHeightMin: rootHeightMin,
            gravityX: 0.0,
            gravityY: 0.0,
            gravityZ: 0.0,
            constraintIterations: 8,
            constraintBaumgarte: 0.2,
            contactBaumgarte: 0.2,
            contactFriction: 0.8,
            reserved: 0
        )

        guard let commandQueue = device.makeCommandQueue() else {
            throw EnvProjectError.commandQueueUnavailable
        }
        self.commandQueue = commandQueue

        let shaderPath = URL(fileURLWithPath: rootDir)
            .appending(path: "src/envs/humanoid/Shaders/humanoid_kernels.metal")
            .path()
        let library = try makeLibrary(device: device, shaderPath: shaderPath)
        self.resetPipeline = try makePipeline(device: device, library: library, name: "humanoid_reset")
        self.stepPipeline = try makePipeline(device: device, library: library, name: "humanoid_step_elastic")
        self.jointMotorPipeline = try makePipeline(device: device, library: library, name: "humanoid_apply_joint_motor_impulses")
        self.freeBodyPipeline = try makePipeline(device: device, library: library, name: "humanoid_integrate_free_bodies")
        self.constraintPipeline = try makePipeline(device: device, library: library, name: "humanoid_solve_joint_anchor_constraints")
        self.anchorErrorPipeline = try makePipeline(device: device, library: library, name: "humanoid_measure_joint_anchor_errors")
        self.groundContactPipeline = try makePipeline(device: device, library: library, name: "humanoid_detect_ground_contacts")
        self.selfContactPipeline = try makePipeline(device: device, library: library, name: "humanoid_detect_self_contacts")
        self.groundContactSolvePipeline = try makePipeline(device: device, library: library, name: "humanoid_solve_ground_contacts")
        self.rootSyncPipeline = try makePipeline(device: device, library: library, name: "humanoid_sync_root_from_pelvis")
        self.fkPipeline = try makePipeline(device: device, library: library, name: "humanoid_forward_kinematics")
        self.outputPipeline = try makePipeline(device: device, library: library, name: "humanoid_write_outputs")

        let linkConstantsLength = MemoryLayout<HumanoidLinkGPUConstants>.stride * linkConstants.count
        let collisionConstantsLength = MemoryLayout<HumanoidCollisionGPUConstants>.stride * collisionConstants.count
        let collisionPairConstantsLength = MemoryLayout<HumanoidCollisionPairGPUConstants>.stride * collisionPairs.count
        let jointConstantsLength = MemoryLayout<HumanoidJointGPUConstants>.stride * jointConstants.count
        let paramsLength = MemoryLayout<HumanoidEnvParams>.stride
        let rootPositionLength = MemoryLayout<Float>.stride * envCount * 3
        let rootRotationLength = MemoryLayout<Float>.stride * envCount * 4
        let jointStateLength = MemoryLayout<Float>.stride * envCount * layout.totalDoFs
        let linkPositionLength = MemoryLayout<Float>.stride * envCount * spec.links.count * 3
        let linkRotationLength = MemoryLayout<Float>.stride * envCount * spec.links.count * 4
        let linkVelocityLength = MemoryLayout<Float>.stride * envCount * spec.links.count * 3
        let jointImpulseLength = MemoryLayout<Float>.stride * envCount * spec.joints.count * 3
        let dofImpulseLength = MemoryLayout<Float>.stride * envCount * layout.totalDoFs
        let solverDiagnosticsLength = MemoryLayout<Float>.stride * envCount * 4
        let jointErrorLength = MemoryLayout<Float>.stride * envCount * spec.joints.count
        let contactVectorLength = MemoryLayout<Float>.stride * envCount * spec.links.count * 3
        let contactScalarLength = MemoryLayout<Float>.stride * envCount * spec.links.count
        let selfContactVectorLength = MemoryLayout<Float>.stride * envCount * collisionPairs.count * 3
        let selfContactScalarLength = MemoryLayout<Float>.stride * envCount * collisionPairs.count
        let observationLength = MemoryLayout<Float>.stride * envCount * observationSpec.elementsPerEnv
        let scalarLength = MemoryLayout<Float>.stride * envCount
        let uintLength = MemoryLayout<UInt32>.stride * envCount

        guard let linkConstantsBuffer = device.makeBuffer(length: linkConstantsLength, options: .storageModeShared) else {
            throw EnvProjectError.bufferAllocationFailed("humanoid-link-constants")
        }
        guard let collisionConstantsBuffer = device.makeBuffer(length: collisionConstantsLength, options: .storageModeShared) else {
            throw EnvProjectError.bufferAllocationFailed("humanoid-collision-constants")
        }
        guard let collisionPairConstantsBuffer = device.makeBuffer(length: collisionPairConstantsLength, options: .storageModeShared) else {
            throw EnvProjectError.bufferAllocationFailed("humanoid-collision-pair-constants")
        }
        guard let jointConstantsBuffer = device.makeBuffer(length: jointConstantsLength, options: .storageModeShared) else {
            throw EnvProjectError.bufferAllocationFailed("humanoid-joint-constants")
        }
        guard let paramsBuffer = device.makeBuffer(length: paramsLength, options: .storageModeShared) else {
            throw EnvProjectError.bufferAllocationFailed("humanoid-params")
        }
        guard let rootPositionBuffer = device.makeBuffer(length: rootPositionLength, options: .storageModeShared),
              let rootRotationBuffer = device.makeBuffer(length: rootRotationLength, options: .storageModeShared),
              let jointPositionBuffer = device.makeBuffer(length: jointStateLength, options: .storageModeShared),
              let jointVelocityBuffer = device.makeBuffer(length: jointStateLength, options: .storageModeShared),
              let actionBuffer = device.makeBuffer(length: jointStateLength, options: .storageModeShared),
              let linkPositionBuffer = device.makeBuffer(length: linkPositionLength, options: .storageModeShared),
              let linkRotationBuffer = device.makeBuffer(length: linkRotationLength, options: .storageModeShared),
              let linkLinearVelocityBuffer = device.makeBuffer(length: linkVelocityLength, options: .storageModeShared),
              let linkAngularVelocityBuffer = device.makeBuffer(length: linkVelocityLength, options: .storageModeShared),
              let jointAnchorImpulseBuffer = device.makeBuffer(length: jointImpulseLength, options: .storageModeShared),
              let jointAngularImpulseBuffer = device.makeBuffer(length: jointImpulseLength, options: .storageModeShared),
              let jointMotorImpulseBuffer = device.makeBuffer(length: dofImpulseLength, options: .storageModeShared),
              let jointLimitImpulseBuffer = device.makeBuffer(length: dofImpulseLength, options: .storageModeShared),
              let solverDiagnosticsBuffer = device.makeBuffer(length: solverDiagnosticsLength, options: .storageModeShared),
              let jointAnchorErrorBuffer = device.makeBuffer(length: jointErrorLength, options: .storageModeShared),
              let contactPointBuffer = device.makeBuffer(length: contactVectorLength, options: .storageModeShared),
              let contactNormalBuffer = device.makeBuffer(length: contactVectorLength, options: .storageModeShared),
              let contactPenetrationBuffer = device.makeBuffer(length: contactScalarLength, options: .storageModeShared),
              let selfContactPointBuffer = device.makeBuffer(length: selfContactVectorLength, options: .storageModeShared),
              let selfContactNormalBuffer = device.makeBuffer(length: selfContactVectorLength, options: .storageModeShared),
              let selfContactPenetrationBuffer = device.makeBuffer(length: selfContactScalarLength, options: .storageModeShared),
              let observationBuffer = device.makeBuffer(length: observationLength, options: .storageModeShared),
              let rewardBuffer = device.makeBuffer(length: scalarLength, options: .storageModeShared),
              let doneBuffer = device.makeBuffer(length: uintLength, options: .storageModeShared),
              let resetCountBuffer = device.makeBuffer(length: uintLength, options: .storageModeShared) else {
            throw EnvProjectError.bufferAllocationFailed("humanoid-state")
        }

        self.linkConstantsBuffer = linkConstantsBuffer
        self.collisionConstantsBuffer = collisionConstantsBuffer
        self.collisionPairConstantsBuffer = collisionPairConstantsBuffer
        self.jointConstantsBuffer = jointConstantsBuffer
        self.paramsBuffer = paramsBuffer
        self.rootPositionBuffer = rootPositionBuffer
        self.rootRotationBuffer = rootRotationBuffer
        self.jointPositionBuffer = jointPositionBuffer
        self.jointVelocityBuffer = jointVelocityBuffer
        self.actionBuffer = actionBuffer
        self.linkPositionBuffer = linkPositionBuffer
        self.linkRotationBuffer = linkRotationBuffer
        self.linkLinearVelocityBuffer = linkLinearVelocityBuffer
        self.linkAngularVelocityBuffer = linkAngularVelocityBuffer
        self.jointAnchorImpulseBuffer = jointAnchorImpulseBuffer
        self.jointAngularImpulseBuffer = jointAngularImpulseBuffer
        self.jointMotorImpulseBuffer = jointMotorImpulseBuffer
        self.jointLimitImpulseBuffer = jointLimitImpulseBuffer
        self.solverDiagnosticsBuffer = solverDiagnosticsBuffer
        self.jointAnchorErrorBuffer = jointAnchorErrorBuffer
        self.contactPointBuffer = contactPointBuffer
        self.contactNormalBuffer = contactNormalBuffer
        self.contactPenetrationBuffer = contactPenetrationBuffer
        self.selfContactPointBuffer = selfContactPointBuffer
        self.selfContactNormalBuffer = selfContactNormalBuffer
        self.selfContactPenetrationBuffer = selfContactPenetrationBuffer
        self.observationBuffer = observationBuffer
        self.rewardBuffer = rewardBuffer
        self.doneBuffer = doneBuffer
        self.resetCountBuffer = resetCountBuffer

        copyArray(linkConstants, to: linkConstantsBuffer)
        copyArray(collisionConstants, to: collisionConstantsBuffer)
        copyArray(collisionPairs, to: collisionPairConstantsBuffer)
        copyArray(jointConstants, to: jointConstantsBuffer)
        writeValue(&params, to: paramsBuffer)

        let rootPositions = Array(repeating: spec.default_pose.root_position, count: envCount).flatMap { $0 }
        let rootRotations = Array(repeating: spec.default_pose.root_rotation, count: envCount).flatMap { $0 }
        let jointPositions = Array(repeating: defaultJointPositions, count: envCount).flatMap { $0 }
        let zeros = Array(repeating: Float.zero, count: envCount * layout.totalDoFs)
        copyArray(rootPositions, to: rootPositionBuffer)
        copyArray(rootRotations, to: rootRotationBuffer)
        copyArray(jointPositions, to: jointPositionBuffer)
        copyArray(zeros, to: jointVelocityBuffer)
        copyArray(zeros, to: actionBuffer)
        copyArray(Array(repeating: Float.zero, count: envCount * spec.links.count * 3), to: linkLinearVelocityBuffer)
        copyArray(Array(repeating: Float.zero, count: envCount * spec.links.count * 3), to: linkAngularVelocityBuffer)
        copyArray(Array(repeating: Float.zero, count: envCount * spec.joints.count * 3), to: jointAnchorImpulseBuffer)
        copyArray(Array(repeating: Float.zero, count: envCount * spec.joints.count * 3), to: jointAngularImpulseBuffer)
        copyArray(Array(repeating: Float.zero, count: envCount * layout.totalDoFs), to: jointMotorImpulseBuffer)
        copyArray(Array(repeating: Float.zero, count: envCount * layout.totalDoFs), to: jointLimitImpulseBuffer)
        copyArray(Array(repeating: Float.zero, count: envCount * 4), to: solverDiagnosticsBuffer)
        copyArray(Array(repeating: Float.zero, count: envCount * spec.joints.count), to: jointAnchorErrorBuffer)
        copyArray(Array(repeating: Float.zero, count: envCount * spec.links.count * 3), to: contactPointBuffer)
        copyArray(Array(repeating: Float.zero, count: envCount * spec.links.count * 3), to: contactNormalBuffer)
        copyArray(Array(repeating: Float.zero, count: envCount * spec.links.count), to: contactPenetrationBuffer)
        copyArray(Array(repeating: Float.zero, count: envCount * collisionPairs.count * 3), to: selfContactPointBuffer)
        copyArray(Array(repeating: Float.zero, count: envCount * collisionPairs.count * 3), to: selfContactNormalBuffer)
        copyArray(Array(repeating: Float.zero, count: envCount * collisionPairs.count), to: selfContactPenetrationBuffer)
        copyArray(Array(repeating: UInt32(0), count: envCount), to: resetCountBuffer)
        try runForwardKinematics()
        try refreshOutputs()
    }

    func reset() throws -> VectorEnvBatch {
        try runComputePass(commandQueue: commandQueue, pipeline: resetPipeline, count: envCount) { encoder in
            encoder.setBuffer(linkConstantsBuffer, offset: 0, index: 0)
            encoder.setBuffer(jointConstantsBuffer, offset: 0, index: 1)
            encoder.setBuffer(paramsBuffer, offset: 0, index: 2)
            encoder.setBuffer(rootPositionBuffer, offset: 0, index: 3)
            encoder.setBuffer(rootRotationBuffer, offset: 0, index: 4)
            encoder.setBuffer(jointPositionBuffer, offset: 0, index: 5)
            encoder.setBuffer(jointVelocityBuffer, offset: 0, index: 6)
            encoder.setBuffer(actionBuffer, offset: 0, index: 7)
            encoder.setBuffer(linkLinearVelocityBuffer, offset: 0, index: 8)
            encoder.setBuffer(linkAngularVelocityBuffer, offset: 0, index: 9)
            encoder.setBuffer(resetCountBuffer, offset: 0, index: 10)
            encoder.setBuffer(jointAnchorImpulseBuffer, offset: 0, index: 11)
            encoder.setBuffer(jointAngularImpulseBuffer, offset: 0, index: 12)
            encoder.setBuffer(jointMotorImpulseBuffer, offset: 0, index: 13)
            encoder.setBuffer(jointLimitImpulseBuffer, offset: 0, index: 14)
            encoder.setBuffer(solverDiagnosticsBuffer, offset: 0, index: 15)
        }
        try runForwardKinematics()
        try refreshOutputs()
        return readBatch()
    }

    func step(actions: [Float]) throws -> VectorEnvBatch {
        let expectedCount = envCount * dofCount
        if actions.count != expectedCount {
            throw EnvProjectError.validationFailed(message: "Humanoid actions size mismatch: expected \(expectedCount), got \(actions.count).")
        }
        copyArray(actions, to: actionBuffer)
        try runComputePass(commandQueue: commandQueue, pipeline: stepPipeline, count: envCount) { encoder in
            encoder.setBuffer(jointConstantsBuffer, offset: 0, index: 0)
            encoder.setBuffer(paramsBuffer, offset: 0, index: 1)
            encoder.setBuffer(jointPositionBuffer, offset: 0, index: 2)
            encoder.setBuffer(jointVelocityBuffer, offset: 0, index: 3)
            encoder.setBuffer(actionBuffer, offset: 0, index: 4)
            encoder.setBuffer(doneBuffer, offset: 0, index: 5)
        }
        try runForwardKinematics()
        try refreshOutputs()
        return readBatch()
    }

    func integrateFreeBodies(gravity: [Float], steps: Int = 1) throws -> VectorEnvBatch {
        if gravity.count != 3 {
            throw EnvProjectError.validationFailed(message: "Humanoid free-body gravity must have 3 elements.")
        }
        if steps < 0 {
            throw EnvProjectError.validationFailed(message: "Humanoid free-body integration step count must be non-negative.")
        }

        params.gravityX = gravity[0]
        params.gravityY = gravity[1]
        params.gravityZ = gravity[2]
        writeValue(&params, to: paramsBuffer)

        for _ in 0..<steps {
            try runComputePass(commandQueue: commandQueue, pipeline: freeBodyPipeline, count: envCount) { encoder in
                encoder.setBuffer(paramsBuffer, offset: 0, index: 0)
                encoder.setBuffer(linkPositionBuffer, offset: 0, index: 1)
                encoder.setBuffer(linkRotationBuffer, offset: 0, index: 2)
                encoder.setBuffer(linkLinearVelocityBuffer, offset: 0, index: 3)
                encoder.setBuffer(linkAngularVelocityBuffer, offset: 0, index: 4)
            }
        }
        try refreshOutputs()
        return readBatch()
    }

    func solveJointAnchorConstraints(iterations: UInt32 = 8, baumgarte: Float = 0.2) throws -> VectorEnvBatch {
        params.constraintIterations = iterations
        params.constraintBaumgarte = baumgarte
        writeValue(&params, to: paramsBuffer)

        try runComputePass(commandQueue: commandQueue, pipeline: constraintPipeline, count: envCount) { encoder in
            encoder.setBuffer(linkConstantsBuffer, offset: 0, index: 0)
            encoder.setBuffer(jointConstantsBuffer, offset: 0, index: 1)
            encoder.setBuffer(paramsBuffer, offset: 0, index: 2)
            encoder.setBuffer(linkPositionBuffer, offset: 0, index: 3)
            encoder.setBuffer(linkRotationBuffer, offset: 0, index: 4)
            encoder.setBuffer(linkLinearVelocityBuffer, offset: 0, index: 5)
            encoder.setBuffer(linkAngularVelocityBuffer, offset: 0, index: 6)
            encoder.setBuffer(jointAnchorImpulseBuffer, offset: 0, index: 7)
            encoder.setBuffer(jointAngularImpulseBuffer, offset: 0, index: 8)
            encoder.setBuffer(solverDiagnosticsBuffer, offset: 0, index: 9)
        }
        try refreshOutputs()
        return readBatch()
    }

    func applyJointMotorImpulses() throws -> VectorEnvBatch {
        try runComputePass(commandQueue: commandQueue, pipeline: jointMotorPipeline, count: envCount) { encoder in
            encoder.setBuffer(linkConstantsBuffer, offset: 0, index: 0)
            encoder.setBuffer(jointConstantsBuffer, offset: 0, index: 1)
            encoder.setBuffer(paramsBuffer, offset: 0, index: 2)
            encoder.setBuffer(jointPositionBuffer, offset: 0, index: 3)
            encoder.setBuffer(jointVelocityBuffer, offset: 0, index: 4)
            encoder.setBuffer(linkRotationBuffer, offset: 0, index: 5)
            encoder.setBuffer(linkLinearVelocityBuffer, offset: 0, index: 6)
            encoder.setBuffer(linkAngularVelocityBuffer, offset: 0, index: 7)
            encoder.setBuffer(jointMotorImpulseBuffer, offset: 0, index: 8)
            encoder.setBuffer(jointLimitImpulseBuffer, offset: 0, index: 9)
            encoder.setBuffer(solverDiagnosticsBuffer, offset: 0, index: 10)
        }
        try refreshOutputs()
        return readBatch()
    }

    func detectGroundContacts() throws {
        try runComputePass(commandQueue: commandQueue, pipeline: groundContactPipeline, count: envCount * linkCount) { encoder in
            encoder.setBuffer(collisionConstantsBuffer, offset: 0, index: 0)
            encoder.setBuffer(paramsBuffer, offset: 0, index: 1)
            encoder.setBuffer(linkPositionBuffer, offset: 0, index: 2)
            encoder.setBuffer(linkRotationBuffer, offset: 0, index: 3)
            encoder.setBuffer(contactPointBuffer, offset: 0, index: 4)
            encoder.setBuffer(contactNormalBuffer, offset: 0, index: 5)
            encoder.setBuffer(contactPenetrationBuffer, offset: 0, index: 6)
        }
    }

    func detectSelfContacts() throws {
        try runComputePass(commandQueue: commandQueue, pipeline: selfContactPipeline, count: envCount * selfCollisionPairCount) { encoder in
            encoder.setBuffer(collisionConstantsBuffer, offset: 0, index: 0)
            encoder.setBuffer(collisionPairConstantsBuffer, offset: 0, index: 1)
            encoder.setBuffer(paramsBuffer, offset: 0, index: 2)
            encoder.setBuffer(linkPositionBuffer, offset: 0, index: 3)
            encoder.setBuffer(linkRotationBuffer, offset: 0, index: 4)
            encoder.setBuffer(selfContactPointBuffer, offset: 0, index: 5)
            encoder.setBuffer(selfContactNormalBuffer, offset: 0, index: 6)
            encoder.setBuffer(selfContactPenetrationBuffer, offset: 0, index: 7)
        }
    }

    func solveGroundContacts(baumgarte: Float = 0.2, friction: Float = 0.8) throws -> VectorEnvBatch {
        params.contactBaumgarte = baumgarte
        params.contactFriction = friction
        writeValue(&params, to: paramsBuffer)
        try detectGroundContacts()
        try runComputePass(commandQueue: commandQueue, pipeline: groundContactSolvePipeline, count: envCount * linkCount) { encoder in
            encoder.setBuffer(paramsBuffer, offset: 0, index: 0)
            encoder.setBuffer(linkPositionBuffer, offset: 0, index: 1)
            encoder.setBuffer(linkLinearVelocityBuffer, offset: 0, index: 2)
            encoder.setBuffer(contactNormalBuffer, offset: 0, index: 3)
            encoder.setBuffer(contactPenetrationBuffer, offset: 0, index: 4)
        }
        try detectGroundContacts()
        try refreshOutputs()
        return readBatch()
    }

    func stepStanding(
        actions: [Float],
        gravity: [Float] = [0.0, 0.0, -9.8],
        jointIterations: UInt32 = 8,
        jointBaumgarte: Float = 0.2,
        contactBaumgarte: Float = 0.2,
        contactFriction: Float = 0.8
    ) throws -> VectorEnvBatch {
        let expectedCount = envCount * dofCount
        if actions.count != expectedCount {
            throw EnvProjectError.validationFailed(message: "Humanoid standing actions size mismatch: expected \(expectedCount), got \(actions.count).")
        }
        if gravity.count != 3 {
            throw EnvProjectError.validationFailed(message: "Humanoid standing gravity must have 3 elements.")
        }

        copyArray(actions, to: actionBuffer)
        params.gravityX = gravity[0]
        params.gravityY = gravity[1]
        params.gravityZ = gravity[2]
        params.constraintIterations = jointIterations
        params.constraintBaumgarte = jointBaumgarte
        params.contactBaumgarte = contactBaumgarte
        params.contactFriction = contactFriction
        writeValue(&params, to: paramsBuffer)

        try runComputePass(commandQueue: commandQueue, pipeline: stepPipeline, count: envCount) { encoder in
            encoder.setBuffer(jointConstantsBuffer, offset: 0, index: 0)
            encoder.setBuffer(paramsBuffer, offset: 0, index: 1)
            encoder.setBuffer(jointPositionBuffer, offset: 0, index: 2)
            encoder.setBuffer(jointVelocityBuffer, offset: 0, index: 3)
            encoder.setBuffer(actionBuffer, offset: 0, index: 4)
            encoder.setBuffer(doneBuffer, offset: 0, index: 5)
        }
        try runForwardKinematics()
        _ = try applyJointMotorImpulses()
        try runComputePass(commandQueue: commandQueue, pipeline: freeBodyPipeline, count: envCount) { encoder in
            encoder.setBuffer(paramsBuffer, offset: 0, index: 0)
            encoder.setBuffer(linkPositionBuffer, offset: 0, index: 1)
            encoder.setBuffer(linkRotationBuffer, offset: 0, index: 2)
            encoder.setBuffer(linkLinearVelocityBuffer, offset: 0, index: 3)
            encoder.setBuffer(linkAngularVelocityBuffer, offset: 0, index: 4)
        }
        try runComputePass(commandQueue: commandQueue, pipeline: constraintPipeline, count: envCount) { encoder in
            encoder.setBuffer(linkConstantsBuffer, offset: 0, index: 0)
            encoder.setBuffer(jointConstantsBuffer, offset: 0, index: 1)
            encoder.setBuffer(paramsBuffer, offset: 0, index: 2)
            encoder.setBuffer(linkPositionBuffer, offset: 0, index: 3)
            encoder.setBuffer(linkRotationBuffer, offset: 0, index: 4)
            encoder.setBuffer(linkLinearVelocityBuffer, offset: 0, index: 5)
            encoder.setBuffer(linkAngularVelocityBuffer, offset: 0, index: 6)
            encoder.setBuffer(jointAnchorImpulseBuffer, offset: 0, index: 7)
            encoder.setBuffer(jointAngularImpulseBuffer, offset: 0, index: 8)
            encoder.setBuffer(solverDiagnosticsBuffer, offset: 0, index: 9)
        }
        try detectGroundContacts()
        try runComputePass(commandQueue: commandQueue, pipeline: groundContactSolvePipeline, count: envCount * linkCount) { encoder in
            encoder.setBuffer(paramsBuffer, offset: 0, index: 0)
            encoder.setBuffer(linkPositionBuffer, offset: 0, index: 1)
            encoder.setBuffer(linkLinearVelocityBuffer, offset: 0, index: 2)
            encoder.setBuffer(contactNormalBuffer, offset: 0, index: 3)
            encoder.setBuffer(contactPenetrationBuffer, offset: 0, index: 4)
        }
        try detectGroundContacts()
        try syncRootFromPelvis()
        try refreshOutputs()
        return readBatch()
    }

    // Validation/repro setup only. This seeds GPU buffers for controlled
    // integrator tests; it is not a CPU simulation path or runtime fallback.
    func loadLinkVelocitiesForValidation(linear: [Float], angular: [Float]) throws {
        let expectedCount = envCount * linkCount * 3
        if linear.count != expectedCount || angular.count != expectedCount {
            throw EnvProjectError.validationFailed(
                message: "Humanoid validation link velocity size mismatch: expected \(expectedCount)."
            )
        }
        copyArray(linear, to: linkLinearVelocityBuffer)
        copyArray(angular, to: linkAngularVelocityBuffer)
        clearJointConstraintImpulses()
    }

    // Validation/repro setup only. This injects deliberately broken GPU link
    // states for constraint tests; it is not a CPU simulator or runtime fallback.
    func loadLinkStateForValidation(positions: [Float], rotations: [Float]) throws {
        let expectedPositions = envCount * linkCount * 3
        let expectedRotations = envCount * linkCount * 4
        if positions.count != expectedPositions || rotations.count != expectedRotations {
            throw EnvProjectError.validationFailed(
                message: "Humanoid validation link state size mismatch: expected \(expectedPositions) positions and \(expectedRotations) rotations."
            )
        }
        copyArray(positions, to: linkPositionBuffer)
        copyArray(rotations, to: linkRotationBuffer)
        clearJointConstraintImpulses()
    }

    // Validation/repro setup only. This seeds GPU joint DoF buffers for one-step
    // motor/limit tests; it is not a CPU simulator or runtime fallback.
    func loadJointStateForValidation(positions: [Float], velocities: [Float]) throws {
        let expectedCount = envCount * dofCount
        if positions.count != expectedCount || velocities.count != expectedCount {
            throw EnvProjectError.validationFailed(
                message: "Humanoid validation joint state size mismatch: expected \(expectedCount)."
            )
        }
        copyArray(positions, to: jointPositionBuffer)
        copyArray(velocities, to: jointVelocityBuffer)
        clearJointConstraintImpulses()
        try runForwardKinematics()
        try refreshOutputs()
    }

    func readBatch() -> VectorEnvBatch {
        VectorEnvBatch(
            observations: readObservations(),
            rewards: readRewards(),
            dones: readDones(),
            resetCounts: readResetCounts()
        )
    }

    func readJointPositions() -> [Float] {
        readArray(from: jointPositionBuffer, count: envCount * dofCount)
    }

    func readJointVelocities() -> [Float] {
        readArray(from: jointVelocityBuffer, count: envCount * dofCount)
    }

    func readLinkPositions() -> [Float] {
        readArray(from: linkPositionBuffer, count: envCount * linkCount * 3)
    }

    func readLinkRotations() -> [Float] {
        readArray(from: linkRotationBuffer, count: envCount * linkCount * 4)
    }

    func readLinkLinearVelocities() -> [Float] {
        readArray(from: linkLinearVelocityBuffer, count: envCount * linkCount * 3)
    }

    func readLinkAngularVelocities() -> [Float] {
        readArray(from: linkAngularVelocityBuffer, count: envCount * linkCount * 3)
    }

    func readLinkConstants() -> [HumanoidLinkGPUConstants] {
        readArray(from: linkConstantsBuffer, count: linkCount)
    }

    func readJointAnchorErrors() throws -> [Float] {
        try runComputePass(commandQueue: commandQueue, pipeline: anchorErrorPipeline, count: envCount) { encoder in
            encoder.setBuffer(jointConstantsBuffer, offset: 0, index: 0)
            encoder.setBuffer(paramsBuffer, offset: 0, index: 1)
            encoder.setBuffer(linkPositionBuffer, offset: 0, index: 2)
            encoder.setBuffer(linkRotationBuffer, offset: 0, index: 3)
            encoder.setBuffer(jointAnchorErrorBuffer, offset: 0, index: 4)
        }
        return readArray(from: jointAnchorErrorBuffer, count: envCount * jointCount)
    }

    func readJointAnchorImpulses() -> [Float] {
        readArray(from: jointAnchorImpulseBuffer, count: envCount * jointCount * 3)
    }

    func readJointAngularImpulses() -> [Float] {
        readArray(from: jointAngularImpulseBuffer, count: envCount * jointCount * 3)
    }

    func readJointMotorImpulses() -> [Float] {
        readArray(from: jointMotorImpulseBuffer, count: envCount * dofCount)
    }

    func readJointLimitImpulses() -> [Float] {
        readArray(from: jointLimitImpulseBuffer, count: envCount * dofCount)
    }

    func readSolverDiagnostics() -> [Float] {
        readArray(from: solverDiagnosticsBuffer, count: envCount * 4)
    }

    func readContactPoints() -> [Float] {
        readArray(from: contactPointBuffer, count: envCount * linkCount * 3)
    }

    func readContactNormals() -> [Float] {
        readArray(from: contactNormalBuffer, count: envCount * linkCount * 3)
    }

    func readContactPenetrations() -> [Float] {
        readArray(from: contactPenetrationBuffer, count: envCount * linkCount)
    }

    func readSelfContactPoints() -> [Float] {
        readArray(from: selfContactPointBuffer, count: envCount * selfCollisionPairCount * 3)
    }

    func readSelfContactNormals() -> [Float] {
        readArray(from: selfContactNormalBuffer, count: envCount * selfCollisionPairCount * 3)
    }

    func readSelfContactPenetrations() -> [Float] {
        readArray(from: selfContactPenetrationBuffer, count: envCount * selfCollisionPairCount)
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

    func readResetCounts() -> [UInt32] {
        readArray(from: resetCountBuffer, count: envCount)
    }

    func makeReplayFrame(envIndex: Int) throws -> HumanoidReplayFrame {
        if envIndex < 0 || envIndex >= envCount {
            throw EnvProjectError.validationFailed(message: "Humanoid replay env index \(envIndex) out of bounds.")
        }
        let linkPositions = readLinkPositions()
        let jointPositions = readJointPositions()
        let rewards = readRewards()
        let dones = readDones()
        let resetCounts = readResetCounts()
        let linkBase = envIndex * linkCount * 3
        let jointBase = envIndex * dofCount
        return HumanoidReplayFrame(
            linkPositions: Array(linkPositions[linkBase..<(linkBase + linkCount * 3)]),
            jointPositions: Array(jointPositions[jointBase..<(jointBase + dofCount)]),
            reward: rewards[envIndex],
            done: dones[envIndex],
            resetCount: resetCounts[envIndex]
        )
    }

    private func runForwardKinematics() throws {
        try runComputePass(commandQueue: commandQueue, pipeline: fkPipeline, count: envCount) { encoder in
            encoder.setBuffer(jointConstantsBuffer, offset: 0, index: 0)
            encoder.setBuffer(paramsBuffer, offset: 0, index: 1)
            encoder.setBuffer(rootPositionBuffer, offset: 0, index: 2)
            encoder.setBuffer(rootRotationBuffer, offset: 0, index: 3)
            encoder.setBuffer(jointPositionBuffer, offset: 0, index: 4)
            encoder.setBuffer(linkPositionBuffer, offset: 0, index: 5)
            encoder.setBuffer(linkRotationBuffer, offset: 0, index: 6)
        }
    }

    private func syncRootFromPelvis() throws {
        try runComputePass(commandQueue: commandQueue, pipeline: rootSyncPipeline, count: envCount) { encoder in
            encoder.setBuffer(paramsBuffer, offset: 0, index: 0)
            encoder.setBuffer(linkPositionBuffer, offset: 0, index: 1)
            encoder.setBuffer(linkRotationBuffer, offset: 0, index: 2)
            encoder.setBuffer(rootPositionBuffer, offset: 0, index: 3)
            encoder.setBuffer(rootRotationBuffer, offset: 0, index: 4)
        }
    }

    private func refreshOutputs() throws {
        try runComputePass(commandQueue: commandQueue, pipeline: outputPipeline, count: envCount) { encoder in
            encoder.setBuffer(paramsBuffer, offset: 0, index: 0)
            encoder.setBuffer(rootPositionBuffer, offset: 0, index: 1)
            encoder.setBuffer(rootRotationBuffer, offset: 0, index: 2)
            encoder.setBuffer(jointPositionBuffer, offset: 0, index: 3)
            encoder.setBuffer(jointVelocityBuffer, offset: 0, index: 4)
            encoder.setBuffer(linkPositionBuffer, offset: 0, index: 5)
            encoder.setBuffer(observationBuffer, offset: 0, index: 6)
            encoder.setBuffer(rewardBuffer, offset: 0, index: 7)
            encoder.setBuffer(doneBuffer, offset: 0, index: 8)
        }
    }

    private func clearJointConstraintImpulses() {
        copyArray(Array(repeating: Float.zero, count: envCount * jointCount * 3), to: jointAnchorImpulseBuffer)
        copyArray(Array(repeating: Float.zero, count: envCount * jointCount * 3), to: jointAngularImpulseBuffer)
        copyArray(Array(repeating: Float.zero, count: envCount * dofCount), to: jointMotorImpulseBuffer)
        copyArray(Array(repeating: Float.zero, count: envCount * dofCount), to: jointLimitImpulseBuffer)
        copyArray(Array(repeating: Float.zero, count: envCount * 4), to: solverDiagnosticsBuffer)
    }
}
