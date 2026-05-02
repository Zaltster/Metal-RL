import Foundation
import Metal

final class HumanoidStandingVectorEnvDriver: MetalVectorEnvDriver {
    let environment: HumanoidMetalEnvironment
    let gravity: [Float]
    let jointIterations: UInt32
    let jointBaumgarte: Float
    let contactBaumgarte: Float
    let contactFriction: Float
    let contactRestitution: Float
    let contactIterations: Int
    let solveSelfContact: Bool

    var envCount: Int { environment.envCount }
    var actionSpec: VectorActionSpec { environment.actionSpec }
    var observationSpec: VectorObservationSpec { environment.observationSpec }

    init(
        environment: HumanoidMetalEnvironment,
        gravity: [Float] = [0.0, 0.0, -9.8],
        jointIterations: UInt32 = 8,
        jointBaumgarte: Float = 0.2,
        contactBaumgarte: Float = 0.2,
        contactFriction: Float = 0.8,
        contactRestitution: Float = 0.0,
        contactIterations: Int = 1,
        solveSelfContact: Bool = true
    ) {
        self.environment = environment
        self.gravity = gravity
        self.jointIterations = jointIterations
        self.jointBaumgarte = jointBaumgarte
        self.contactBaumgarte = contactBaumgarte
        self.contactFriction = contactFriction
        self.contactRestitution = contactRestitution
        self.contactIterations = contactIterations
        self.solveSelfContact = solveSelfContact
    }

    func reset() throws -> VectorEnvBatch {
        try environment.reset()
    }

    func step(actions: [Float]) throws -> VectorEnvBatch {
        try environment.stepStanding(
            actions: actions,
            gravity: gravity,
            jointIterations: jointIterations,
            jointBaumgarte: jointBaumgarte,
            contactBaumgarte: contactBaumgarte,
            contactFriction: contactFriction,
            contactRestitution: contactRestitution,
            contactIterations: contactIterations,
            solveSelfContact: solveSelfContact
        )
    }

    func resetDone() throws -> VectorEnvBatch {
        environment.readBatch()
    }
}
