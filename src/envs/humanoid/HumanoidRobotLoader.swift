import Foundation

// This file is asset loading and validation only. It must not grow into a CPU
// humanoid simulator or a training fallback; the runtime environment is Metal.

func loadHumanoidRobotSpec(from url: URL) throws -> HumanoidRobotSpec {
    let data = try Data(contentsOf: url)
    let decoder = JSONDecoder()
    return try decoder.decode(HumanoidRobotSpec.self, from: data)
}

func validateHumanoidRobotSpec(_ spec: HumanoidRobotSpec) throws -> HumanoidValidationReport {
    var warnings: [String] = []

    if spec.schema_version != "1.0" {
        warnings.append("Unexpected humanoid schema_version \(spec.schema_version); expected 1.0.")
    }
    if spec.units != "SI" {
        throw EnvProjectError.validationFailed(message: "Humanoid spec units must be SI.")
    }
    if spec.links.isEmpty {
        throw EnvProjectError.validationFailed(message: "Humanoid spec must contain links.")
    }
    if spec.joints.isEmpty {
        throw EnvProjectError.validationFailed(message: "Humanoid spec must contain joints.")
    }

    var linkNames = Set<String>()
    var linkIndex: [String: Int] = [:]
    var totalMass: Float = 0.0
    for (index, link) in spec.links.enumerated() {
        if link.name.isEmpty {
            throw EnvProjectError.validationFailed(message: "Humanoid link at index \(index) has empty name.")
        }
        if linkNames.contains(link.name) {
            throw EnvProjectError.validationFailed(message: "Duplicate humanoid link name \(link.name).")
        }
        linkNames.insert(link.name)
        linkIndex[link.name] = index

        if link.inertial.mass <= 0.0 {
            throw EnvProjectError.validationFailed(message: "Humanoid link \(link.name) mass must be positive.")
        }
        if link.inertial.com.count != 3 {
            throw EnvProjectError.validationFailed(message: "Humanoid link \(link.name) COM must have 3 elements.")
        }
        try validatePositiveDefiniteInertia(link.inertial.inertia, context: "Humanoid link \(link.name)")
        totalMass += link.inertial.mass

        try validateHumanoidShapes(link.visual, context: "Humanoid link \(link.name) visual")
        try validateHumanoidShapes(link.collision, context: "Humanoid link \(link.name) collision")
    }

    if totalMass < 60.0 || totalMass > 90.0 {
        warnings.append(String(format: "Humanoid total mass %.2f kg is outside the 75 kg +/-20%% band.", totalMass))
    }

    var jointNames = Set<String>()
    var parentCounts: [String: Int] = [:]
    var freeCount = 0
    var layoutRows: [HumanoidDoFLayoutRow] = []
    var dofOffset = 0

    for (index, joint) in spec.joints.enumerated() {
        if joint.name.isEmpty {
            throw EnvProjectError.validationFailed(message: "Humanoid joint at index \(index) has empty name.")
        }
        if jointNames.contains(joint.name) {
            throw EnvProjectError.validationFailed(message: "Duplicate humanoid joint name \(joint.name).")
        }
        jointNames.insert(joint.name)

        if joint.anchor_in_parent.count != 3 || joint.anchor_in_child.count != 3 {
            throw EnvProjectError.validationFailed(message: "Humanoid joint \(joint.name) anchors must have 3 elements.")
        }
        try validateUnitQuaternion(joint.frame_in_parent, context: "Humanoid joint \(joint.name) frame_in_parent")
        try validateUnitQuaternion(joint.frame_in_child, context: "Humanoid joint \(joint.name) frame_in_child")

        if joint.type == .free {
            freeCount += 1
            if joint.parent_link != "world" {
                throw EnvProjectError.validationFailed(message: "Humanoid free joint \(joint.name) must use parent_link world.")
            }
        } else if linkIndex[joint.parent_link] == nil {
            throw EnvProjectError.validationFailed(message: "Humanoid joint \(joint.name) parent_link \(joint.parent_link) is missing.")
        }

        guard let childIndex = linkIndex[joint.child_link] else {
            throw EnvProjectError.validationFailed(message: "Humanoid joint \(joint.name) child_link \(joint.child_link) is missing.")
        }
        parentCounts[joint.child_link, default: 0] += 1

        if joint.type != .free, let parentIndex = linkIndex[joint.parent_link], parentIndex >= childIndex {
            throw EnvProjectError.validationFailed(
                message: "Humanoid joint \(joint.name) violates parent-before-child link ordering."
            )
        }

        try validateJointFields(joint)
        layoutRows.append(HumanoidDoFLayoutRow(name: joint.name, offset: dofOffset, size: joint.type.dofCount))
        dofOffset += joint.type.dofCount
    }

    if freeCount != 1 {
        throw EnvProjectError.validationFailed(message: "Humanoid spec requires exactly one free root joint; found \(freeCount).")
    }
    for link in spec.links {
        let parentCount = parentCounts[link.name, default: 0]
        if parentCount != 1 {
            throw EnvProjectError.validationFailed(
                message: "Humanoid link \(link.name) must have exactly one parent joint; found \(parentCount)."
            )
        }
    }

    try validateUnitQuaternion(spec.default_pose.root_rotation, context: "Humanoid default root_rotation")
    if spec.default_pose.root_position.count != 3 {
        throw EnvProjectError.validationFailed(message: "Humanoid default root_position must have 3 elements.")
    }
    try validateDefaultPose(spec: spec, jointNames: jointNames)

    return HumanoidValidationReport(
        warnings: warnings,
        dofLayout: HumanoidDoFLayout(totalDoFs: dofOffset, joints: layoutRows)
    )
}

func makeHumanoidJointConstants(
    spec: HumanoidRobotSpec,
    layout: HumanoidDoFLayout
) throws -> [HumanoidJointGPUConstants] {
    let linkIndex = Dictionary(uniqueKeysWithValues: spec.links.enumerated().map { ($1.name, $0) })
    let pose = spec.default_pose.joint_positions

    return try spec.joints.enumerated().map { index, joint in
        let row = layout.joints[index]
        let defaults = try defaultValues(for: joint, pose: pose)
        let limits = jointLimitTriples(joint)
        let dynamics = joint.dynamics
        let maxForce = actuatorMaxForceTriple(joint)
        let maxVelocity = actuatorMaxVelocityTriple(joint)

        return HumanoidJointGPUConstants(
            parentLink: Int32(linkIndex[joint.parent_link] ?? -1),
            childLink: Int32(linkIndex[joint.child_link] ?? -1),
            type: joint.type.gpuCode,
            dofOffset: UInt32(row.offset),
            dofCount: UInt32(row.size),
            reserved0: 0,
            reserved1: 0,
            reserved2: 0,
            anchorParentX: joint.anchor_in_parent[0],
            anchorParentY: joint.anchor_in_parent[1],
            anchorParentZ: joint.anchor_in_parent[2],
            anchorChildX: joint.anchor_in_child[0],
            anchorChildY: joint.anchor_in_child[1],
            anchorChildZ: joint.anchor_in_child[2],
            frameParentX: joint.frame_in_parent[0],
            frameParentY: joint.frame_in_parent[1],
            frameParentZ: joint.frame_in_parent[2],
            frameParentW: joint.frame_in_parent[3],
            frameChildX: joint.frame_in_child[0],
            frameChildY: joint.frame_in_child[1],
            frameChildZ: joint.frame_in_child[2],
            frameChildW: joint.frame_in_child[3],
            limitMinX: limits.min[0],
            limitMinY: limits.min[1],
            limitMinZ: limits.min[2],
            limitMaxX: limits.max[0],
            limitMaxY: limits.max[1],
            limitMaxZ: limits.max[2],
            defaultX: defaults[0],
            defaultY: defaults[1],
            defaultZ: defaults[2],
            dampingX: dynamics?.damping ?? 0.0,
            dampingY: dynamics?.damping ?? 0.0,
            dampingZ: dynamics?.damping ?? 0.0,
            stiffnessX: dynamics?.stiffness ?? 0.0,
            stiffnessY: dynamics?.stiffness ?? 0.0,
            stiffnessZ: dynamics?.stiffness ?? 0.0,
            armatureX: dynamics?.armature ?? 1.0,
            armatureY: dynamics?.armature ?? 1.0,
            armatureZ: dynamics?.armature ?? 1.0,
            maxForceX: maxForce[0],
            maxForceY: maxForce[1],
            maxForceZ: maxForce[2],
            maxVelocityX: maxVelocity[0],
            maxVelocityY: maxVelocity[1],
            maxVelocityZ: maxVelocity[2]
        )
    }
}

func makeHumanoidLinkConstants(spec: HumanoidRobotSpec) throws -> [HumanoidLinkGPUConstants] {
    try spec.links.map { link in
        let inertia = link.inertial.inertia
        let inverse = try inverseInertia(inertia, context: "Humanoid link \(link.name)")
        return HumanoidLinkGPUConstants(
            mass: link.inertial.mass,
            invMass: 1.0 / link.inertial.mass,
            comX: link.inertial.com[0],
            comY: link.inertial.com[1],
            comZ: link.inertial.com[2],
            inertiaIxx: inertia[0],
            inertiaIyy: inertia[1],
            inertiaIzz: inertia[2],
            inertiaIxy: inertia[3],
            inertiaIxz: inertia[4],
            inertiaIyz: inertia[5],
            invInertiaIxx: inverse[0],
            invInertiaIyy: inverse[1],
            invInertiaIzz: inverse[2],
            invInertiaIxy: inverse[3],
            invInertiaIxz: inverse[4],
            invInertiaIyz: inverse[5]
        )
    }
}

func makeHumanoidCollisionConstants(spec: HumanoidRobotSpec) -> [HumanoidCollisionGPUConstants] {
    spec.links.map { link in
        guard let shape = link.collision.first else {
            return HumanoidCollisionGPUConstants(
                type: 0,
                reserved0: 0,
                reserved1: 0,
                reserved2: 0,
                translationX: 0.0,
                translationY: 0.0,
                translationZ: 0.0,
                rotationX: 0.0,
                rotationY: 0.0,
                rotationZ: 0.0,
                rotationW: 1.0,
                paramX: 0.0,
                paramY: 0.0,
                paramZ: 0.0
            )
        }
        let params = shape.params ?? [:]
        let typeCode: UInt32
        let dimensions: [Float]
        switch shape.type {
        case "box":
            typeCode = 1
            dimensions = params["half_extents"]?.floatArrayValue ?? [0.0, 0.0, 0.0]
        case "sphere":
            typeCode = 2
            let radius = params["radius"]?.floatValue ?? 0.0
            dimensions = [radius, 0.0, 0.0]
        case "capsule":
            typeCode = 3
            dimensions = [
                params["radius"]?.floatValue ?? 0.0,
                params["half_length"]?.floatValue ?? 0.0,
                0.0,
            ]
        case "cylinder":
            typeCode = 4
            dimensions = [
                params["radius"]?.floatValue ?? 0.0,
                params["half_length"]?.floatValue ?? 0.0,
                0.0,
            ]
        default:
            typeCode = 0
            dimensions = [0.0, 0.0, 0.0]
        }
        return HumanoidCollisionGPUConstants(
            type: typeCode,
            reserved0: 0,
            reserved1: 0,
            reserved2: 0,
            translationX: shape.transform.translation[0],
            translationY: shape.transform.translation[1],
            translationZ: shape.transform.translation[2],
            rotationX: shape.transform.rotation[0],
            rotationY: shape.transform.rotation[1],
            rotationZ: shape.transform.rotation[2],
            rotationW: shape.transform.rotation[3],
            paramX: dimensions.indices.contains(0) ? dimensions[0] : 0.0,
            paramY: dimensions.indices.contains(1) ? dimensions[1] : 0.0,
            paramZ: dimensions.indices.contains(2) ? dimensions[2] : 0.0
        )
    }
}

func makeHumanoidCollisionPairConstants(linkCount: Int) -> [HumanoidCollisionPairGPUConstants] {
    var pairs: [HumanoidCollisionPairGPUConstants] = []
    for linkA in 0..<linkCount {
        for linkB in (linkA + 1)..<linkCount {
            pairs.append(HumanoidCollisionPairGPUConstants(
                linkA: UInt32(linkA),
                linkB: UInt32(linkB),
                reserved0: 0,
                reserved1: 0
            ))
        }
    }
    return pairs
}

func makeHumanoidDefaultJointPositions(spec: HumanoidRobotSpec, layout: HumanoidDoFLayout) throws -> [Float] {
    var values = Array(repeating: Float.zero, count: layout.totalDoFs)
    for (index, joint) in spec.joints.enumerated() {
        let row = layout.joints[index]
        let defaults = try defaultValues(for: joint, pose: spec.default_pose.joint_positions)
        for axis in 0..<row.size {
            values[row.offset + axis] = defaults[axis]
        }
    }
    return values
}

private func validateHumanoidShapes(_ shapes: [HumanoidShapeSpec], context: String) throws {
    let allowed = Set(["box", "sphere", "capsule", "cylinder", "convex_hull", "mesh"])
    for (index, shape) in shapes.enumerated() {
        if !allowed.contains(shape.type) {
            throw EnvProjectError.validationFailed(message: "\(context)[\(index)] has unsupported type \(shape.type).")
        }
        if shape.transform.translation.count != 3 {
            throw EnvProjectError.validationFailed(message: "\(context)[\(index)] translation must have 3 elements.")
        }
        try validateUnitQuaternion(shape.transform.rotation, context: "\(context)[\(index)] rotation")
        if shape.type != "mesh" {
            try validateShapeParams(shape, context: "\(context)[\(index)]")
        }
    }
}

private func validateShapeParams(_ shape: HumanoidShapeSpec, context: String) throws {
    let params = shape.params ?? [:]
    switch shape.type {
    case "box":
        guard let values = params["half_extents"]?.floatArrayValue, values.count == 3, values.allSatisfy({ $0 > 0 }) else {
            throw EnvProjectError.validationFailed(message: "\(context) box half_extents must be 3 positive numbers.")
        }
    case "sphere":
        guard let radius = params["radius"]?.floatValue, radius > 0 else {
            throw EnvProjectError.validationFailed(message: "\(context) sphere radius must be positive.")
        }
    case "capsule", "cylinder":
        guard let radius = params["radius"]?.floatValue, radius > 0,
              let halfLength = params["half_length"]?.floatValue, halfLength > 0 else {
            throw EnvProjectError.validationFailed(message: "\(context) \(shape.type) radius and half_length must be positive.")
        }
    case "convex_hull":
        guard case .string? = params["vertices_file"] else {
            throw EnvProjectError.validationFailed(message: "\(context) convex_hull requires vertices_file.")
        }
    default:
        break
    }
}

private func validateJointFields(_ joint: HumanoidJointSpec) throws {
    switch joint.type {
    case .free:
        if joint.limits != nil || joint.dynamics != nil || joint.actuator != nil {
            throw EnvProjectError.validationFailed(message: "Humanoid free joint \(joint.name) must have null limits/dynamics/actuator.")
        }
    case .fixed:
        break
    case .revolute, .prismatic:
        guard let position = joint.limits?.position, position.count == 2, position[0] <= position[1] else {
            throw EnvProjectError.validationFailed(message: "Humanoid joint \(joint.name) requires ordered position limits.")
        }
        guard joint.dynamics != nil, joint.actuator != nil else {
            throw EnvProjectError.validationFailed(message: "Humanoid joint \(joint.name) requires dynamics and actuator.")
        }
    case .spherical:
        guard let limits = joint.limits,
              let swingX = limits.swing_x, swingX.count == 2, swingX[0] <= swingX[1],
              let swingY = limits.swing_y, swingY.count == 2, swingY[0] <= swingY[1],
              let twistZ = limits.twist_z, twistZ.count == 2, twistZ[0] <= twistZ[1] else {
            throw EnvProjectError.validationFailed(message: "Humanoid spherical joint \(joint.name) requires ordered swing/twist limits.")
        }
        guard joint.dynamics != nil, joint.actuator != nil else {
            throw EnvProjectError.validationFailed(message: "Humanoid joint \(joint.name) requires dynamics and actuator.")
        }
    }
}

private func validateDefaultPose(spec: HumanoidRobotSpec, jointNames: Set<String>) throws {
    for (name, value) in spec.default_pose.joint_positions {
        guard jointNames.contains(name), let joint = spec.joints.first(where: { $0.name == name }) else {
            throw EnvProjectError.validationFailed(message: "Humanoid default_pose references unknown joint \(name).")
        }
        let values = try defaultValues(for: joint, pose: [name: value])
        let limits = jointLimitTriples(joint)
        for axis in 0..<joint.type.dofCount {
            if values[axis] < limits.min[axis] || values[axis] > limits.max[axis] {
                throw EnvProjectError.validationFailed(message: "Humanoid default_pose \(name) axis \(axis) is outside joint limits.")
            }
        }
    }
}

private func defaultValues(for joint: HumanoidJointSpec, pose: [String: HumanoidJSONValue]) throws -> [Float] {
    switch joint.type {
    case .free, .fixed:
        return [0.0, 0.0, 0.0]
    case .revolute, .prismatic:
        guard let value = pose[joint.name]?.floatValue else {
            throw EnvProjectError.validationFailed(message: "Humanoid default_pose missing scalar value for joint \(joint.name).")
        }
        return [value, 0.0, 0.0]
    case .spherical:
        guard let values = pose[joint.name]?.floatArrayValue, values.count == 3 else {
            throw EnvProjectError.validationFailed(message: "Humanoid default_pose missing 3-vector value for joint \(joint.name).")
        }
        return values
    }
}

private func jointLimitTriples(_ joint: HumanoidJointSpec) -> (min: [Float], max: [Float]) {
    switch joint.type {
    case .free, .fixed:
        return ([-Float.greatestFiniteMagnitude, -Float.greatestFiniteMagnitude, -Float.greatestFiniteMagnitude],
                [Float.greatestFiniteMagnitude, Float.greatestFiniteMagnitude, Float.greatestFiniteMagnitude])
    case .revolute, .prismatic:
        let position = joint.limits?.position ?? [0.0, 0.0]
        return ([position[0], 0.0, 0.0], [position[1], 0.0, 0.0])
    case .spherical:
        let limits = joint.limits
        return (
            [limits?.swing_x?[0] ?? 0.0, limits?.swing_y?[0] ?? 0.0, limits?.twist_z?[0] ?? 0.0],
            [limits?.swing_x?[1] ?? 0.0, limits?.swing_y?[1] ?? 0.0, limits?.twist_z?[1] ?? 0.0]
        )
    }
}

private func actuatorMaxForceTriple(_ joint: HumanoidJointSpec) -> [Float] {
    guard let actuator = joint.actuator else {
        return [0.0, 0.0, 0.0]
    }
    if let scalar = actuator.max_force.floatValue {
        return [scalar, scalar, scalar]
    }
    if let values = actuator.max_force.floatArrayValue {
        return [
            values.indices.contains(0) ? values[0] : 0.0,
            values.indices.contains(1) ? values[1] : 0.0,
            values.indices.contains(2) ? values[2] : 0.0,
        ]
    }
    return [0.0, 0.0, 0.0]
}

private func actuatorMaxVelocityTriple(_ joint: HumanoidJointSpec) -> [Float] {
    guard let actuator = joint.actuator else {
        return [Float.greatestFiniteMagnitude, Float.greatestFiniteMagnitude, Float.greatestFiniteMagnitude]
    }
    return [actuator.max_velocity, actuator.max_velocity, actuator.max_velocity]
}

private func validateUnitQuaternion(_ q: [Float], context: String) throws {
    if q.count != 4 {
        throw EnvProjectError.validationFailed(message: "\(context) must have 4 elements.")
    }
    let norm = sqrt(q.reduce(Float.zero) { $0 + $1 * $1 })
    if abs(norm - 1.0) > 1e-6 {
        throw EnvProjectError.validationFailed(message: "\(context) must be normalized; norm=\(norm).")
    }
}

private func validatePositiveDefiniteInertia(_ inertia: [Float], context: String) throws {
    if inertia.count != 6 {
        throw EnvProjectError.validationFailed(message: "\(context) inertia must have 6 elements.")
    }
    let ixx = inertia[0]
    let iyy = inertia[1]
    let izz = inertia[2]
    let ixy = inertia[3]
    let ixz = inertia[4]
    let iyz = inertia[5]
    if ixx <= 0 {
        throw EnvProjectError.validationFailed(message: "\(context) inertia Ixx must be positive.")
    }
    if ixx * iyy - ixy * ixy <= 0 {
        throw EnvProjectError.validationFailed(message: "\(context) inertia leading 2x2 minor must be positive.")
    }
    let det = ixx * (iyy * izz - iyz * iyz) -
        ixy * (ixy * izz - iyz * ixz) +
        ixz * (ixy * iyz - iyy * ixz)
    if det <= 0 {
        throw EnvProjectError.validationFailed(message: "\(context) inertia tensor must be positive-definite.")
    }
}

private func inverseInertia(_ inertia: [Float], context: String) throws -> [Float] {
    if inertia.count != 6 {
        throw EnvProjectError.validationFailed(message: "\(context) inertia must have 6 elements.")
    }

    let a = inertia[0]
    let b = inertia[1]
    let c = inertia[2]
    let d = inertia[3]
    let e = inertia[4]
    let f = inertia[5]
    let det = a * (b * c - f * f) -
        d * (d * c - f * e) +
        e * (d * f - b * e)

    if det <= 0 {
        throw EnvProjectError.validationFailed(message: "\(context) inertia tensor inverse requires positive determinant.")
    }

    let invIxx = (b * c - f * f) / det
    let invIyy = (a * c - e * e) / det
    let invIzz = (a * b - d * d) / det
    let invIxy = (e * f - d * c) / det
    let invIxz = (d * f - e * b) / det
    let invIyz = (d * e - a * f) / det
    return [invIxx, invIyy, invIzz, invIxy, invIxz, invIyz]
}
