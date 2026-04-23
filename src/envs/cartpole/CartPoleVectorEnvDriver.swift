import Foundation
import Metal

final class CartPoleVectorEnvDriver: MetalVectorEnvDriver {
    let environment: CartPoleMetalEnvironment
    let envCount: Int
    let actionSpec: VectorActionSpec
    let observationSpec: VectorObservationSpec

    private let initialStates: [CartPoleState]

    init(
        environment: CartPoleMetalEnvironment,
        initialStates: [CartPoleState]
    ) throws {
        if initialStates.count != environment.envCount {
            throw EnvProjectError.validationFailed(message: "initialStates count must match environment.envCount.")
        }

        self.environment = environment
        self.envCount = environment.envCount
        self.actionSpec = environment.actionSpec
        self.observationSpec = environment.observationSpec
        self.initialStates = initialStates
    }

    convenience init(
        device: MTLDevice,
        rootDir: String,
        envCount: Int,
        cartPoleParams: CartPoleParams,
        resetSeed: UInt32,
        initialStates: [CartPoleState]? = nil
    ) throws {
        let environment = try CartPoleMetalEnvironment(
            device: device,
            rootDir: rootDir,
            envCount: envCount,
            cartPoleParams: cartPoleParams,
            resetSeed: resetSeed
        )
        let states = initialStates ?? makeCartPoleInitialStates(count: envCount)
        try self.init(environment: environment, initialStates: states)
    }

    func setResetSeed(_ seed: UInt32) {
        environment.setResetSeed(seed)
    }

    func reset() throws -> VectorEnvBatch {
        try environment.load(initialStates: initialStates)
        return environment.readBatch()
    }

    func step(actions: [Float]) throws -> VectorEnvBatch {
        try environment.step(actions: actions)
        return environment.readBatch()
    }

    func resetDone() throws -> VectorEnvBatch {
        try environment.resetDone()
        return environment.readBatch()
    }

    func debugReadStates() -> [CartPoleState] {
        environment.readStates()
    }

    func debugReadResetCounts() -> [UInt32] {
        environment.readResetCounts()
    }
}
