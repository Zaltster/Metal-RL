import Foundation

enum HumanoidJointType: String, Codable {
    case free
    case fixed
    case revolute
    case prismatic
    case spherical

    var gpuCode: UInt32 {
        switch self {
        case .free:
            return 0
        case .fixed:
            return 1
        case .revolute:
            return 2
        case .prismatic:
            return 3
        case .spherical:
            return 4
        }
    }

    var dofCount: Int {
        switch self {
        case .free, .fixed:
            return 0
        case .revolute, .prismatic:
            return 1
        case .spherical:
            return 3
        }
    }
}

enum HumanoidJSONValue: Codable {
    case number(Float)
    case array([Float])
    case string(String)
    case object([String: HumanoidJSONValue])
    case null

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let value = try? container.decode(Float.self) {
            self = .number(value)
        } else if let value = try? container.decode([Float].self) {
            self = .array(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else {
            self = .object(try container.decode([String: HumanoidJSONValue].self))
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case let .number(value):
            try container.encode(value)
        case let .array(value):
            try container.encode(value)
        case let .string(value):
            try container.encode(value)
        case let .object(value):
            try container.encode(value)
        case .null:
            try container.encodeNil()
        }
    }

    var floatValue: Float? {
        if case let .number(value) = self {
            return value
        }
        return nil
    }

    var floatArrayValue: [Float]? {
        if case let .array(value) = self {
            return value
        }
        return nil
    }
}

struct HumanoidTransformSpec: Codable {
    let translation: [Float]
    let rotation: [Float]
}

struct HumanoidShapeSpec: Codable {
    let type: String
    let params: [String: HumanoidJSONValue]?
    let transform: HumanoidTransformSpec
    let material: HumanoidMaterialSpec?
}

struct HumanoidMaterialSpec: Codable {
    let friction: Float
    let restitution: Float
}

struct HumanoidInertialSpec: Codable {
    let mass: Float
    let com: [Float]
    let inertia: [Float]
}

struct HumanoidLinkSpec: Codable {
    let name: String
    let inertial: HumanoidInertialSpec
    let visual: [HumanoidShapeSpec]
    let collision: [HumanoidShapeSpec]
}

struct HumanoidJointLimits: Codable {
    let position: [Float]?
    let swing_x: [Float]?
    let swing_y: [Float]?
    let twist_z: [Float]?
}

struct HumanoidJointDynamics: Codable {
    let damping: Float
    let armature: Float
    let friction: Float
    let stiffness: Float
}

struct HumanoidJointActuator: Codable {
    let type: String
    let max_force: HumanoidJSONValue
    let max_velocity: Float
}

struct HumanoidJointSpec: Codable {
    let name: String
    let type: HumanoidJointType
    let parent_link: String
    let child_link: String
    let anchor_in_parent: [Float]
    let anchor_in_child: [Float]
    let frame_in_parent: [Float]
    let frame_in_child: [Float]
    let limits: HumanoidJointLimits?
    let dynamics: HumanoidJointDynamics?
    let actuator: HumanoidJointActuator?
}

struct HumanoidDefaultPose: Codable {
    let root_position: [Float]
    let root_rotation: [Float]
    let joint_positions: [String: HumanoidJSONValue]
}

struct HumanoidRobotSpec: Codable {
    let schema_version: String
    let name: String
    let units: String
    let links: [HumanoidLinkSpec]
    let joints: [HumanoidJointSpec]
    let default_pose: HumanoidDefaultPose
}

struct HumanoidDoFLayoutRow {
    let name: String
    let offset: Int
    let size: Int
}

struct HumanoidDoFLayout {
    let totalDoFs: Int
    let joints: [HumanoidDoFLayoutRow]
}

struct HumanoidValidationReport {
    let warnings: [String]
    let dofLayout: HumanoidDoFLayout
}

struct HumanoidJointGPUConstants {
    var parentLink: Int32
    var childLink: Int32
    var type: UInt32
    var dofOffset: UInt32
    var dofCount: UInt32
    var reserved0: UInt32
    var reserved1: UInt32
    var reserved2: UInt32

    var anchorParentX: Float
    var anchorParentY: Float
    var anchorParentZ: Float
    var anchorChildX: Float
    var anchorChildY: Float
    var anchorChildZ: Float
    var frameParentX: Float
    var frameParentY: Float
    var frameParentZ: Float
    var frameParentW: Float
    var frameChildX: Float
    var frameChildY: Float
    var frameChildZ: Float
    var frameChildW: Float

    var limitMinX: Float
    var limitMinY: Float
    var limitMinZ: Float
    var limitMaxX: Float
    var limitMaxY: Float
    var limitMaxZ: Float

    var defaultX: Float
    var defaultY: Float
    var defaultZ: Float
    var dampingX: Float
    var dampingY: Float
    var dampingZ: Float
    var stiffnessX: Float
    var stiffnessY: Float
    var stiffnessZ: Float
    var armatureX: Float
    var armatureY: Float
    var armatureZ: Float
    var maxForceX: Float
    var maxForceY: Float
    var maxForceZ: Float
    var maxVelocityX: Float
    var maxVelocityY: Float
    var maxVelocityZ: Float
}

struct HumanoidLinkGPUConstants {
    var mass: Float
    var invMass: Float
    var comX: Float
    var comY: Float
    var comZ: Float
    var inertiaIxx: Float
    var inertiaIyy: Float
    var inertiaIzz: Float
    var inertiaIxy: Float
    var inertiaIxz: Float
    var inertiaIyz: Float
    var invInertiaIxx: Float
    var invInertiaIyy: Float
    var invInertiaIzz: Float
    var invInertiaIxy: Float
    var invInertiaIxz: Float
    var invInertiaIyz: Float
}

struct HumanoidCollisionGPUConstants {
    var type: UInt32
    var reserved0: UInt32
    var reserved1: UInt32
    var reserved2: UInt32
    var translationX: Float
    var translationY: Float
    var translationZ: Float
    var rotationX: Float
    var rotationY: Float
    var rotationZ: Float
    var rotationW: Float
    var paramX: Float
    var paramY: Float
    var paramZ: Float
}

struct HumanoidCollisionPairGPUConstants {
    var linkA: UInt32
    var linkB: UInt32
    var reserved0: UInt32
    var reserved1: UInt32
}

struct HumanoidEnvParams {
    var envCount: UInt32
    var linkCount: UInt32
    var selfCollisionPairCount: UInt32
    var jointCount: UInt32
    var dofCount: UInt32
    var observationDim: UInt32
    var dt: Float
    var rootHeightMin: Float
    var gravityX: Float
    var gravityY: Float
    var gravityZ: Float
    var constraintIterations: UInt32
    var constraintBaumgarte: Float
    var contactBaumgarte: Float
    var contactFriction: Float
    var contactRestitution: Float
    var selectedSelfCollisionPair: UInt32
}

struct HumanoidReplayFrame {
    let linkPositions: [Float]
    let jointPositions: [Float]
    let contactPoints: [Float]
    let contactNormals: [Float]
    let contactPenetrations: [Float]
    let reward: Float
    let done: UInt32
    let resetCount: UInt32
}
