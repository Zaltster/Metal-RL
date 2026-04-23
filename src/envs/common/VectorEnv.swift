import Foundation

struct VectorActionSpec {
    let dimensionsPerEnv: Int
    let minValue: Float
    let maxValue: Float
}

struct VectorObservationSpec {
    let elementsPerEnv: Int
}

struct VectorEnvBatch {
    let observations: [Float]
    let rewards: [Float]
    let dones: [UInt32]
    let resetCounts: [UInt32]
}

protocol MetalVectorEnvDriver {
    var envCount: Int { get }
    var actionSpec: VectorActionSpec { get }
    var observationSpec: VectorObservationSpec { get }

    func reset() throws -> VectorEnvBatch
    func step(actions: [Float]) throws -> VectorEnvBatch
    func resetDone() throws -> VectorEnvBatch
}
