#include <metal_stdlib>
using namespace metal;

struct HumanoidLinkGPUConstants {
    float mass;
    float invMass;
    float comX;
    float comY;
    float comZ;
    float inertiaIxx;
    float inertiaIyy;
    float inertiaIzz;
    float inertiaIxy;
    float inertiaIxz;
    float inertiaIyz;
    float invInertiaIxx;
    float invInertiaIyy;
    float invInertiaIzz;
    float invInertiaIxy;
    float invInertiaIxz;
    float invInertiaIyz;
};

struct HumanoidCollisionGPUConstants {
    uint type;
    uint reserved0;
    uint reserved1;
    uint reserved2;
    float translationX;
    float translationY;
    float translationZ;
    float rotationX;
    float rotationY;
    float rotationZ;
    float rotationW;
    float paramX;
    float paramY;
    float paramZ;
};

struct HumanoidCollisionPairGPUConstants {
    uint linkA;
    uint linkB;
    uint reserved0;
    uint reserved1;
};

struct HumanoidJointGPUConstants {
    int parentLink;
    int childLink;
    uint type;
    uint dofOffset;
    uint dofCount;
    uint reserved0;
    uint reserved1;
    uint reserved2;

    float anchorParentX;
    float anchorParentY;
    float anchorParentZ;
    float anchorChildX;
    float anchorChildY;
    float anchorChildZ;
    float frameParentX;
    float frameParentY;
    float frameParentZ;
    float frameParentW;
    float frameChildX;
    float frameChildY;
    float frameChildZ;
    float frameChildW;

    float limitMinX;
    float limitMinY;
    float limitMinZ;
    float limitMaxX;
    float limitMaxY;
    float limitMaxZ;

    float defaultX;
    float defaultY;
    float defaultZ;
    float dampingX;
    float dampingY;
    float dampingZ;
    float stiffnessX;
    float stiffnessY;
    float stiffnessZ;
    float armatureX;
    float armatureY;
    float armatureZ;
    float maxForceX;
    float maxForceY;
    float maxForceZ;
    float maxVelocityX;
    float maxVelocityY;
    float maxVelocityZ;
};

struct HumanoidEnvParams {
    uint envCount;
    uint linkCount;
    uint selfCollisionPairCount;
    uint jointCount;
    uint dofCount;
    uint observationDim;
    float dt;
    float rootHeightMin;
    float gravityX;
    float gravityY;
    float gravityZ;
    uint constraintIterations;
    float constraintBaumgarte;
    float contactBaumgarte;
    float contactFriction;
    uint reserved;
};

float4 quat_mul(float4 a, float4 b) {
    return float4(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z
    );
}

float4 quat_conjugate(float4 q) {
    return float4(-q.x, -q.y, -q.z, q.w);
}

float4 quat_axis_angle(float3 axis, float angle) {
    const float halfAngle = 0.5f * angle;
    const float s = sin(halfAngle);
    return normalize(float4(axis * s, cos(halfAngle)));
}

float4 quat_from_rotation_vector(float3 rotationVector) {
    const float angle = length(rotationVector);
    if (angle <= 1.0e-7f) {
        return float4(0.0f, 0.0f, 0.0f, 1.0f);
    }
    return quat_axis_angle(rotationVector / angle, angle);
}

float3 quat_to_rotation_vector(float4 q) {
    float4 normalized = normalize(q);
    if (normalized.w < 0.0f) {
        normalized = -normalized;
    }
    const float vectorLength = length(normalized.xyz);
    if (vectorLength <= 1.0e-7f) {
        return float3(0.0f);
    }
    const float angle = 2.0f * atan2(vectorLength, normalized.w);
    return normalized.xyz * (angle / vectorLength);
}

float3 quat_rotate(float4 q, float3 v) {
    const float3 t = 2.0f * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

float4 integrate_quaternion(float4 q, float3 angularVelocity, float dt) {
    const float4 omega = float4(angularVelocity, 0.0f);
    const float4 qdot = 0.5f * quat_mul(omega, q);
    return normalize(q + dt * qdot);
}

float3 load_float3(device const float *values, uint base) {
    return float3(values[base + 0u], values[base + 1u], values[base + 2u]);
}

float4 load_quat(device const float *values, uint base) {
    return normalize(float4(values[base + 0u], values[base + 1u], values[base + 2u], values[base + 3u]));
}

void store_float3(device float *values, uint base, float3 value) {
    values[base + 0u] = value.x;
    values[base + 1u] = value.y;
    values[base + 2u] = value.z;
}

void store_quat(device float *values, uint base, float4 value) {
    const float4 q = normalize(value);
    values[base + 0u] = q.x;
    values[base + 1u] = q.y;
    values[base + 2u] = q.z;
    values[base + 3u] = q.w;
}

float3 joint_vec3(float x, float y, float z) {
    return float3(x, y, z);
}

float4 joint_frame_parent(const HumanoidJointGPUConstants joint) {
    return normalize(float4(joint.frameParentX, joint.frameParentY, joint.frameParentZ, joint.frameParentW));
}

float4 joint_frame_child(const HumanoidJointGPUConstants joint) {
    return normalize(float4(joint.frameChildX, joint.frameChildY, joint.frameChildZ, joint.frameChildW));
}

float joint_axis_value(device const float *values, uint base, uint dofCount, uint axis) {
    return axis < dofCount ? values[base + axis] : 0.0f;
}

float4 joint_rotation(const HumanoidJointGPUConstants joint, device const float *jointPositions, uint envBase) {
    const uint base = envBase + joint.dofOffset;
    if (joint.type == 2u) {
        return quat_axis_angle(float3(1.0f, 0.0f, 0.0f), jointPositions[base]);
    }
    if (joint.type == 4u) {
        const float sx = joint_axis_value(jointPositions, base, joint.dofCount, 0u);
        const float sy = joint_axis_value(jointPositions, base, joint.dofCount, 1u);
        const float tz = joint_axis_value(jointPositions, base, joint.dofCount, 2u);
        const float4 qx = quat_axis_angle(float3(1.0f, 0.0f, 0.0f), sx);
        const float4 qy = quat_axis_angle(float3(0.0f, 1.0f, 0.0f), sy);
        const float4 qz = quat_axis_angle(float3(0.0f, 0.0f, 1.0f), tz);
        return normalize(quat_mul(qz, quat_mul(qy, qx)));
    }
    return float4(0.0f, 0.0f, 0.0f, 1.0f);
}

kernel void humanoid_reset(
    device const HumanoidLinkGPUConstants *links [[buffer(0)]],
    device const HumanoidJointGPUConstants *joints [[buffer(1)]],
    constant HumanoidEnvParams &params [[buffer(2)]],
    device float *rootPositions [[buffer(3)]],
    device float *rootRotations [[buffer(4)]],
    device float *jointPositions [[buffer(5)]],
    device float *jointVelocities [[buffer(6)]],
    device float *actions [[buffer(7)]],
    device float *linkLinearVelocities [[buffer(8)]],
    device float *linkAngularVelocities [[buffer(9)]],
    device uint *resetCounts [[buffer(10)]],
    device float *jointAnchorImpulses [[buffer(11)]],
    device float *jointAngularImpulses [[buffer(12)]],
    device float *jointMotorImpulses [[buffer(13)]],
    device float *jointLimitImpulses [[buffer(14)]],
    device float *solverDiagnostics [[buffer(15)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.envCount) {
        return;
    }

    const uint rootBase3 = gid * 3u;
    rootPositions[rootBase3 + 0u] = 0.0f;
    rootPositions[rootBase3 + 1u] = 0.0f;
    rootPositions[rootBase3 + 2u] = 1.0f;

    const uint rootBase4 = gid * 4u;
    rootRotations[rootBase4 + 0u] = 0.0f;
    rootRotations[rootBase4 + 1u] = 0.0f;
    rootRotations[rootBase4 + 2u] = 0.0f;
    rootRotations[rootBase4 + 3u] = 1.0f;

    const uint envDofBase = gid * params.dofCount;
    const uint envLinkBase3 = gid * params.linkCount * 3u;
    const uint diagnosticsBase = gid * 4u;
    solverDiagnostics[diagnosticsBase + 0u] = 0.0f;
    solverDiagnostics[diagnosticsBase + 1u] = 0.0f;
    solverDiagnostics[diagnosticsBase + 2u] = 0.0f;
    solverDiagnostics[diagnosticsBase + 3u] = 0.0f;
    for (uint linkIndex = 0u; linkIndex < params.linkCount; ++linkIndex) {
        const HumanoidLinkGPUConstants link = links[linkIndex];
        const uint base = envLinkBase3 + linkIndex * 3u;
        linkLinearVelocities[base + 0u] = 0.0f * link.invMass;
        linkLinearVelocities[base + 1u] = 0.0f;
        linkLinearVelocities[base + 2u] = 0.0f;
        linkAngularVelocities[base + 0u] = 0.0f * link.invInertiaIxx;
        linkAngularVelocities[base + 1u] = 0.0f;
        linkAngularVelocities[base + 2u] = 0.0f;
    }

    for (uint jointIndex = 0u; jointIndex < params.jointCount; ++jointIndex) {
        const HumanoidJointGPUConstants joint = joints[jointIndex];
        const uint base = envDofBase + joint.dofOffset;
        const uint impulseBase = (gid * params.jointCount + jointIndex) * 3u;
        jointAnchorImpulses[impulseBase + 0u] = 0.0f;
        jointAnchorImpulses[impulseBase + 1u] = 0.0f;
        jointAnchorImpulses[impulseBase + 2u] = 0.0f;
        jointAngularImpulses[impulseBase + 0u] = 0.0f;
        jointAngularImpulses[impulseBase + 1u] = 0.0f;
        jointAngularImpulses[impulseBase + 2u] = 0.0f;
        if (joint.dofCount > 0u) {
            jointPositions[base + 0u] = joint.defaultX;
            jointVelocities[base + 0u] = 0.0f;
            actions[base + 0u] = 0.0f;
            jointMotorImpulses[base + 0u] = 0.0f;
            jointLimitImpulses[base + 0u] = 0.0f;
        }
        if (joint.dofCount > 1u) {
            jointPositions[base + 1u] = joint.defaultY;
            jointVelocities[base + 1u] = 0.0f;
            actions[base + 1u] = 0.0f;
            jointMotorImpulses[base + 1u] = 0.0f;
            jointLimitImpulses[base + 1u] = 0.0f;
        }
        if (joint.dofCount > 2u) {
            jointPositions[base + 2u] = joint.defaultZ;
            jointVelocities[base + 2u] = 0.0f;
            actions[base + 2u] = 0.0f;
            jointMotorImpulses[base + 2u] = 0.0f;
            jointLimitImpulses[base + 2u] = 0.0f;
        }
    }
    resetCounts[gid] += 1u;
}

kernel void humanoid_step_elastic(
    device const HumanoidJointGPUConstants *joints [[buffer(0)]],
    constant HumanoidEnvParams &params [[buffer(1)]],
    device float *jointPositions [[buffer(2)]],
    device float *jointVelocities [[buffer(3)]],
    device const float *actions [[buffer(4)]],
    device const uint *dones [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.envCount || dones[gid] != 0u) {
        return;
    }

    const uint envDofBase = gid * params.dofCount;
    for (uint jointIndex = 0u; jointIndex < params.jointCount; ++jointIndex) {
        const HumanoidJointGPUConstants joint = joints[jointIndex];
        const uint base = envDofBase + joint.dofOffset;
        for (uint axis = 0u; axis < joint.dofCount; ++axis) {
            const float defaults[3] = { joint.defaultX, joint.defaultY, joint.defaultZ };
            const float damping[3] = { joint.dampingX, joint.dampingY, joint.dampingZ };
            const float stiffness[3] = { joint.stiffnessX, joint.stiffnessY, joint.stiffnessZ };
            const float armature[3] = { joint.armatureX, joint.armatureY, joint.armatureZ };
            const float maxForce[3] = { joint.maxForceX, joint.maxForceY, joint.maxForceZ };
            const float maxVelocity[3] = { joint.maxVelocityX, joint.maxVelocityY, joint.maxVelocityZ };
            const float mins[3] = { joint.limitMinX, joint.limitMinY, joint.limitMinZ };
            const float maxs[3] = { joint.limitMaxX, joint.limitMaxY, joint.limitMaxZ };

            float q = jointPositions[base + axis];
            float qd = jointVelocities[base + axis];
            const float motor = clamp(actions[base + axis], -1.0f, 1.0f) * maxForce[axis];
            const float spring = -stiffness[axis] * (q - defaults[axis]);
            const float damper = -damping[axis] * qd;
            const float effectiveInertia = max(armature[axis], 1.0e-4f);
            const float qdd = (motor + spring + damper) / effectiveInertia;

            qd += params.dt * qdd;
            qd = clamp(qd, -maxVelocity[axis], maxVelocity[axis]);
            q += params.dt * qd;
            if (q < mins[axis]) {
                q = mins[axis];
                qd = min(qd, 0.0f) * -0.1f;
            }
            if (q > maxs[axis]) {
                q = maxs[axis];
                qd = max(qd, 0.0f) * -0.1f;
            }

            jointPositions[base + axis] = q;
            jointVelocities[base + axis] = qd;
        }
        if (joint.type == 4u && joint.dofCount == 3u) {
            float swingX = jointPositions[base + 0u];
            float swingY = jointPositions[base + 1u];
            const float maxSwingX = max(abs(joint.limitMinX), abs(joint.limitMaxX));
            const float maxSwingY = max(abs(joint.limitMinY), abs(joint.limitMaxY));
            const float normalizedSwing = sqrt(
                (swingX * swingX) / max(maxSwingX * maxSwingX, 1.0e-6f) +
                (swingY * swingY) / max(maxSwingY * maxSwingY, 1.0e-6f)
            );
            if (normalizedSwing > 1.0f) {
                swingX /= normalizedSwing;
                swingY /= normalizedSwing;
                jointPositions[base + 0u] = swingX;
                jointPositions[base + 1u] = swingY;
                jointVelocities[base + 0u] *= -0.1f;
                jointVelocities[base + 1u] *= -0.1f;
            }
        }
    }
}

float angular_inv_mass_for_axis(const HumanoidLinkGPUConstants link, float3 axis) {
    const float3 a = abs(axis);
    return max(
        a.x * link.invInertiaIxx + a.y * link.invInertiaIyy + a.z * link.invInertiaIzz,
        1.0e-6f
    );
}

float3 inverse_inertia_mul_local(const HumanoidLinkGPUConstants link, float3 axis) {
    return float3(
        link.invInertiaIxx * axis.x + link.invInertiaIxy * axis.y + link.invInertiaIxz * axis.z,
        link.invInertiaIxy * axis.x + link.invInertiaIyy * axis.y + link.invInertiaIyz * axis.z,
        link.invInertiaIxz * axis.x + link.invInertiaIyz * axis.y + link.invInertiaIzz * axis.z
    );
}

float3 inverse_inertia_mul_world(const HumanoidLinkGPUConstants link, float4 rotation, float3 axis) {
    const float3 localAxis = quat_rotate(quat_conjugate(rotation), axis);
    const float3 localResult = inverse_inertia_mul_local(link, localAxis);
    return quat_rotate(rotation, localResult);
}

float inverse_inertia_quadratic_world(const HumanoidLinkGPUConstants link, float4 rotation, float3 axis) {
    return max(dot(axis, inverse_inertia_mul_world(link, rotation, axis)), 0.0f);
}

float anchor_row_inverse_mass(
    const HumanoidLinkGPUConstants link,
    float4 rotation,
    float3 anchorOffset,
    float3 axis
) {
    const float3 angularAxis = cross(anchorOffset, axis);
    return link.invMass + inverse_inertia_quadratic_world(link, rotation, angularAxis);
}

kernel void humanoid_apply_joint_motor_impulses(
    device const HumanoidLinkGPUConstants *links [[buffer(0)]],
    device const HumanoidJointGPUConstants *joints [[buffer(1)]],
    constant HumanoidEnvParams &params [[buffer(2)]],
    device const float *jointPositions [[buffer(3)]],
    device const float *jointVelocities [[buffer(4)]],
    device const float *linkRotations [[buffer(5)]],
    device float *linkLinearVelocities [[buffer(6)]],
    device float *linkAngularVelocities [[buffer(7)]],
    device float *jointMotorImpulses [[buffer(8)]],
    device float *jointLimitImpulses [[buffer(9)]],
    device float *solverDiagnostics [[buffer(10)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.envCount) {
        return;
    }

    const uint envDofBase = gid * params.dofCount;
    const uint envLinkBase3 = gid * params.linkCount * 3u;
    const uint envLinkBase4 = gid * params.linkCount * 4u;
    const uint diagnosticsBase = gid * 4u;
    float maxConstraintError = solverDiagnostics[diagnosticsBase + 0u];
    float maxImpulseMagnitude = solverDiagnostics[diagnosticsBase + 1u];
    float nonFiniteCount = solverDiagnostics[diagnosticsBase + 2u];
    float rowFailureCount = solverDiagnostics[diagnosticsBase + 3u];
    for (uint jointIndex = 0u; jointIndex < params.jointCount; ++jointIndex) {
        const HumanoidJointGPUConstants joint = joints[jointIndex];
        if (joint.dofCount == 0u || joint.parentLink < 0 || joint.childLink < 0) {
            continue;
        }

        const uint parent = uint(joint.parentLink);
        const uint child = uint(joint.childLink);
        const uint parentBase3 = envLinkBase3 + parent * 3u;
        const uint childBase3 = envLinkBase3 + child * 3u;
        const uint parentBase4 = envLinkBase4 + parent * 4u;
        const uint childBase4 = envLinkBase4 + child * 4u;
        const float4 parentRotation = load_quat(linkRotations, parentBase4);
        const float4 childRotation = load_quat(linkRotations, childBase4);

        for (uint axisIndex = 0u; axisIndex < joint.dofCount; ++axisIndex) {
            const uint dofBase = envDofBase + joint.dofOffset + axisIndex;
            const uint impulseBase = envDofBase + joint.dofOffset + axisIndex;
            const float targetRelativeVelocity = jointVelocities[dofBase];
            const float position = jointPositions[dofBase];
            float3 localAxis = float3(1.0f, 0.0f, 0.0f);
            if (joint.type == 4u && axisIndex == 1u) {
                localAxis = float3(0.0f, 1.0f, 0.0f);
            } else if (joint.type == 4u && axisIndex == 2u) {
                localAxis = float3(0.0f, 0.0f, 1.0f);
            }
            const float3 worldAxis = normalize(quat_rotate(parentRotation, localAxis));
            const float maxForce[3] = { joint.maxForceX, joint.maxForceY, joint.maxForceZ };
            const float mins[3] = { joint.limitMinX, joint.limitMinY, joint.limitMinZ };
            const float maxs[3] = { joint.limitMaxX, joint.limitMaxY, joint.limitMaxZ };
            const float maxMotorImpulse = max(maxForce[axisIndex] * params.dt, 1.0e-6f);
            const float maxLimitImpulse = max(maxMotorImpulse, 1.0f);

            float currentRelativeVelocity = 0.0f;
            float rowInvMass = 0.0f;
            if (joint.type == 3u) {
                float3 parentVelocity = load_float3(linkLinearVelocities, parentBase3);
                float3 childVelocity = load_float3(linkLinearVelocities, childBase3);
                const float parentInvMass = links[parent].invMass;
                const float childInvMass = links[child].invMass;
                rowInvMass = parentInvMass + childInvMass;
                if (rowInvMass <= 1.0e-7f) {
                    rowFailureCount += 1.0f;
                    continue;
                }

                const float cachedMotorImpulse = jointMotorImpulses[impulseBase];
                parentVelocity -= worldAxis * cachedMotorImpulse * parentInvMass;
                childVelocity += worldAxis * cachedMotorImpulse * childInvMass;
                currentRelativeVelocity = dot(childVelocity - parentVelocity, worldAxis);
                float motorLambda = (targetRelativeVelocity - currentRelativeVelocity) / rowInvMass;
                const float oldMotorImpulse = jointMotorImpulses[impulseBase];
                const float newMotorImpulse = clamp(oldMotorImpulse + motorLambda, -maxMotorImpulse, maxMotorImpulse);
                motorLambda = newMotorImpulse - oldMotorImpulse;
                jointMotorImpulses[impulseBase] = newMotorImpulse;
                parentVelocity -= worldAxis * motorLambda * parentInvMass;
                childVelocity += worldAxis * motorLambda * childInvMass;
                maxImpulseMagnitude = max(maxImpulseMagnitude, abs(newMotorImpulse));

                float limitSign = 0.0f;
                float limitError = 0.0f;
                if (position <= mins[axisIndex] + 1.0e-5f) {
                    limitSign = 1.0f;
                    limitError = max(0.0f, mins[axisIndex] - position);
                } else if (position >= maxs[axisIndex] - 1.0e-5f) {
                    limitSign = -1.0f;
                    limitError = max(0.0f, position - maxs[axisIndex]);
                }
                if (limitSign != 0.0f) {
                    const float cachedLimitImpulse = jointLimitImpulses[impulseBase];
                    parentVelocity -= worldAxis * (limitSign * cachedLimitImpulse) * parentInvMass;
                    childVelocity += worldAxis * (limitSign * cachedLimitImpulse) * childInvMass;
                    currentRelativeVelocity = dot(childVelocity - parentVelocity, worldAxis);
                    const float targetLimitVelocity = limitError * params.constraintBaumgarte / max(params.dt, 1.0e-6f);
                    float limitLambda = (targetLimitVelocity - limitSign * currentRelativeVelocity) / rowInvMass;
                    const float oldLimitImpulse = jointLimitImpulses[impulseBase];
                    const float newLimitImpulse = clamp(oldLimitImpulse + limitLambda, 0.0f, maxLimitImpulse);
                    limitLambda = newLimitImpulse - oldLimitImpulse;
                    jointLimitImpulses[impulseBase] = newLimitImpulse;
                    parentVelocity -= worldAxis * (limitSign * limitLambda) * parentInvMass;
                    childVelocity += worldAxis * (limitSign * limitLambda) * childInvMass;
                    maxConstraintError = max(maxConstraintError, limitError);
                    maxImpulseMagnitude = max(maxImpulseMagnitude, abs(newLimitImpulse));
                } else {
                    jointLimitImpulses[impulseBase] = 0.0f;
                }

                if (!all(isfinite(parentVelocity)) || !all(isfinite(childVelocity))) {
                    nonFiniteCount += 1.0f;
                }
                store_float3(linkLinearVelocities, parentBase3, parentVelocity);
                store_float3(linkLinearVelocities, childBase3, childVelocity);
            } else {
                float3 parentAngularVelocity = load_float3(linkAngularVelocities, parentBase3);
                float3 childAngularVelocity = load_float3(linkAngularVelocities, childBase3);
                rowInvMass = inverse_inertia_quadratic_world(links[parent], parentRotation, worldAxis)
                    + inverse_inertia_quadratic_world(links[child], childRotation, worldAxis);
                if (rowInvMass <= 1.0e-7f) {
                    rowFailureCount += 1.0f;
                    continue;
                }

                const float3 parentAngularMass = inverse_inertia_mul_world(links[parent], parentRotation, worldAxis);
                const float3 childAngularMass = inverse_inertia_mul_world(links[child], childRotation, worldAxis);
                const float cachedMotorImpulse = jointMotorImpulses[impulseBase];
                parentAngularVelocity -= parentAngularMass * cachedMotorImpulse;
                childAngularVelocity += childAngularMass * cachedMotorImpulse;
                currentRelativeVelocity = dot(childAngularVelocity - parentAngularVelocity, worldAxis);
                float motorLambda = (targetRelativeVelocity - currentRelativeVelocity) / rowInvMass;
                const float oldMotorImpulse = jointMotorImpulses[impulseBase];
                const float newMotorImpulse = clamp(oldMotorImpulse + motorLambda, -maxMotorImpulse, maxMotorImpulse);
                motorLambda = newMotorImpulse - oldMotorImpulse;
                jointMotorImpulses[impulseBase] = newMotorImpulse;
                parentAngularVelocity -= parentAngularMass * motorLambda;
                childAngularVelocity += childAngularMass * motorLambda;
                maxImpulseMagnitude = max(maxImpulseMagnitude, abs(newMotorImpulse));

                float limitSign = 0.0f;
                float limitError = 0.0f;
                if (position <= mins[axisIndex] + 1.0e-5f) {
                    limitSign = 1.0f;
                    limitError = max(0.0f, mins[axisIndex] - position);
                } else if (position >= maxs[axisIndex] - 1.0e-5f) {
                    limitSign = -1.0f;
                    limitError = max(0.0f, position - maxs[axisIndex]);
                }
                if (limitSign != 0.0f) {
                    const float cachedLimitImpulse = jointLimitImpulses[impulseBase];
                    parentAngularVelocity -= parentAngularMass * (limitSign * cachedLimitImpulse);
                    childAngularVelocity += childAngularMass * (limitSign * cachedLimitImpulse);
                    currentRelativeVelocity = dot(childAngularVelocity - parentAngularVelocity, worldAxis);
                    const float targetLimitVelocity = limitError * params.constraintBaumgarte / max(params.dt, 1.0e-6f);
                    float limitLambda = (targetLimitVelocity - limitSign * currentRelativeVelocity) / rowInvMass;
                    const float oldLimitImpulse = jointLimitImpulses[impulseBase];
                    const float newLimitImpulse = clamp(oldLimitImpulse + limitLambda, 0.0f, maxLimitImpulse);
                    limitLambda = newLimitImpulse - oldLimitImpulse;
                    jointLimitImpulses[impulseBase] = newLimitImpulse;
                    parentAngularVelocity -= parentAngularMass * (limitSign * limitLambda);
                    childAngularVelocity += childAngularMass * (limitSign * limitLambda);
                    maxConstraintError = max(maxConstraintError, limitError);
                    maxImpulseMagnitude = max(maxImpulseMagnitude, abs(newLimitImpulse));
                } else {
                    jointLimitImpulses[impulseBase] = 0.0f;
                }

                if (!all(isfinite(parentAngularVelocity)) || !all(isfinite(childAngularVelocity))) {
                    nonFiniteCount += 1.0f;
                }
                store_float3(linkAngularVelocities, parentBase3, parentAngularVelocity);
                store_float3(linkAngularVelocities, childBase3, childAngularVelocity);
            }
        }
    }
    solverDiagnostics[diagnosticsBase + 0u] = maxConstraintError;
    solverDiagnostics[diagnosticsBase + 1u] = maxImpulseMagnitude;
    solverDiagnostics[diagnosticsBase + 2u] = nonFiniteCount;
    solverDiagnostics[diagnosticsBase + 3u] = rowFailureCount;
}

kernel void humanoid_integrate_free_bodies(
    constant HumanoidEnvParams &params [[buffer(0)]],
    device float *linkPositions [[buffer(1)]],
    device float *linkRotations [[buffer(2)]],
    device float *linkLinearVelocities [[buffer(3)]],
    device float *linkAngularVelocities [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.envCount) {
        return;
    }

    const float3 gravity = float3(params.gravityX, params.gravityY, params.gravityZ);
    const uint envLinkBase3 = gid * params.linkCount * 3u;
    const uint envLinkBase4 = gid * params.linkCount * 4u;
    for (uint linkIndex = 0u; linkIndex < params.linkCount; ++linkIndex) {
        const uint base3 = envLinkBase3 + linkIndex * 3u;
        const uint base4 = envLinkBase4 + linkIndex * 4u;

        float3 position = float3(
            linkPositions[base3 + 0u],
            linkPositions[base3 + 1u],
            linkPositions[base3 + 2u]
        );
        float4 rotation = normalize(float4(
            linkRotations[base4 + 0u],
            linkRotations[base4 + 1u],
            linkRotations[base4 + 2u],
            linkRotations[base4 + 3u]
        ));
        float3 linearVelocity = float3(
            linkLinearVelocities[base3 + 0u],
            linkLinearVelocities[base3 + 1u],
            linkLinearVelocities[base3 + 2u]
        );
        const float3 angularVelocity = float3(
            linkAngularVelocities[base3 + 0u],
            linkAngularVelocities[base3 + 1u],
            linkAngularVelocities[base3 + 2u]
        );

        linearVelocity += params.dt * gravity;
        position += params.dt * linearVelocity;
        rotation = integrate_quaternion(rotation, angularVelocity, params.dt);

        linkPositions[base3 + 0u] = position.x;
        linkPositions[base3 + 1u] = position.y;
        linkPositions[base3 + 2u] = position.z;
        linkRotations[base4 + 0u] = rotation.x;
        linkRotations[base4 + 1u] = rotation.y;
        linkRotations[base4 + 2u] = rotation.z;
        linkRotations[base4 + 3u] = rotation.w;
        linkLinearVelocities[base3 + 0u] = linearVelocity.x;
        linkLinearVelocities[base3 + 1u] = linearVelocity.y;
        linkLinearVelocities[base3 + 2u] = linearVelocity.z;
    }
}

kernel void humanoid_solve_joint_anchor_constraints(
    device const HumanoidLinkGPUConstants *links [[buffer(0)]],
    device const HumanoidJointGPUConstants *joints [[buffer(1)]],
    constant HumanoidEnvParams &params [[buffer(2)]],
    device float *linkPositions [[buffer(3)]],
    device float *linkRotations [[buffer(4)]],
    device float *linkLinearVelocities [[buffer(5)]],
    device float *linkAngularVelocities [[buffer(6)]],
    device float *jointAnchorImpulses [[buffer(7)]],
    device float *jointAngularImpulses [[buffer(8)]],
    device float *solverDiagnostics [[buffer(9)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.envCount) {
        return;
    }

    const uint envLinkBase3 = gid * params.linkCount * 3u;
    const uint envLinkBase4 = gid * params.linkCount * 4u;
    const uint diagnosticsBase = gid * 4u;
    float maxConstraintError = solverDiagnostics[diagnosticsBase + 0u];
    float maxImpulseMagnitude = solverDiagnostics[diagnosticsBase + 1u];
    float nonFiniteCount = solverDiagnostics[diagnosticsBase + 2u];
    float rowFailureCount = solverDiagnostics[diagnosticsBase + 3u];
    const float velocityScale = params.constraintBaumgarte / max(params.dt, 1.0e-6f);
    for (uint iteration = 0u; iteration < params.constraintIterations; ++iteration) {
        for (uint jointIndex = 0u; jointIndex < params.jointCount; ++jointIndex) {
            const HumanoidJointGPUConstants joint = joints[jointIndex];
            if (joint.type == 0u || joint.parentLink < 0 || joint.childLink < 0) {
                continue;
            }

            const uint parent = uint(joint.parentLink);
            const uint child = uint(joint.childLink);
            const uint parentBase3 = envLinkBase3 + parent * 3u;
            const uint childBase3 = envLinkBase3 + child * 3u;
            const uint parentBase4 = envLinkBase4 + parent * 4u;
            const uint childBase4 = envLinkBase4 + child * 4u;

            float3 parentPosition = load_float3(linkPositions, parentBase3);
            float3 childPosition = load_float3(linkPositions, childBase3);
            float4 parentRotation = load_quat(linkRotations, parentBase4);
            float4 childRotation = load_quat(linkRotations, childBase4);
            float3 parentVelocity = load_float3(linkLinearVelocities, parentBase3);
            float3 childVelocity = load_float3(linkLinearVelocities, childBase3);
            float3 parentAngularVelocity = load_float3(linkAngularVelocities, parentBase3);
            float3 childAngularVelocity = load_float3(linkAngularVelocities, childBase3);

            const float3 anchorParent = joint_vec3(joint.anchorParentX, joint.anchorParentY, joint.anchorParentZ);
            const float3 anchorChild = joint_vec3(joint.anchorChildX, joint.anchorChildY, joint.anchorChildZ);
            const float4 parentFrame = joint_frame_parent(joint);
            const float4 childFrame = joint_frame_child(joint);
            const float3 sliderAxis = normalize(quat_rotate(quat_mul(parentRotation, parentFrame), float3(1.0f, 0.0f, 0.0f)));
            const float3 rowAxes[3] = {
                float3(1.0f, 0.0f, 0.0f),
                float3(0.0f, 1.0f, 0.0f),
                float3(0.0f, 0.0f, 1.0f),
            };
            const uint impulseBase = (gid * params.jointCount + jointIndex) * 3u;
            for (uint row = 0u; row < 3u; ++row) {
                float3 rowAxis = rowAxes[row];
                if (joint.type == 3u) {
                    rowAxis -= sliderAxis * dot(rowAxis, sliderAxis);
                    const float rowAxisLength = length(rowAxis);
                    if (rowAxisLength <= 1.0e-6f) {
                        continue;
                    }
                    rowAxis /= rowAxisLength;
                }
                const float cachedLambda = jointAnchorImpulses[impulseBase + row];
                if (abs(cachedLambda) <= 1.0e-8f) {
                    continue;
                }
                const float3 parentAnchorOffset = quat_rotate(parentRotation, anchorParent);
                const float3 childAnchorOffset = quat_rotate(childRotation, anchorChild);
                const float3 parentAngularAxis = cross(parentAnchorOffset, rowAxis);
                const float3 childAngularAxis = cross(childAnchorOffset, rowAxis);
                parentVelocity -= rowAxis * cachedLambda * links[parent].invMass;
                childVelocity += rowAxis * cachedLambda * links[child].invMass;
                parentAngularVelocity -= inverse_inertia_mul_world(links[parent], parentRotation, parentAngularAxis) * cachedLambda;
                childAngularVelocity += inverse_inertia_mul_world(links[child], childRotation, childAngularAxis) * cachedLambda;
            }
            for (uint row = 0u; row < 3u; ++row) {
                float3 rowAxis = rowAxes[row];
                if (joint.type == 3u) {
                    rowAxis -= sliderAxis * dot(rowAxis, sliderAxis);
                    const float rowAxisLength = length(rowAxis);
                    if (rowAxisLength <= 1.0e-6f) {
                        continue;
                    }
                    rowAxis /= rowAxisLength;
                }

                const float3 parentAnchorOffset = quat_rotate(parentRotation, anchorParent);
                const float3 childAnchorOffset = quat_rotate(childRotation, anchorChild);
                const float3 worldParentAnchor = parentPosition + parentAnchorOffset;
                const float3 worldChildAnchor = childPosition + childAnchorOffset;
                const float errorAlongAxis = dot(worldParentAnchor - worldChildAnchor, rowAxis);
                const float parentRowInvMass = anchor_row_inverse_mass(links[parent], parentRotation, parentAnchorOffset, rowAxis);
                const float childRowInvMass = anchor_row_inverse_mass(links[child], childRotation, childAnchorOffset, rowAxis);
                const float rowInvMass = parentRowInvMass + childRowInvMass;
                if (rowInvMass <= 1.0e-7f) {
                    rowFailureCount += 1.0f;
                    continue;
                }
                maxConstraintError = max(maxConstraintError, abs(errorAlongAxis));

                const float positionLambda = errorAlongAxis / rowInvMass;
                const float3 parentAngularAxis = cross(parentAnchorOffset, rowAxis);
                const float3 childAngularAxis = cross(childAnchorOffset, rowAxis);
                const float3 parentAngularDelta = -inverse_inertia_mul_world(links[parent], parentRotation, parentAngularAxis) * positionLambda;
                const float3 childAngularDelta = inverse_inertia_mul_world(links[child], childRotation, childAngularAxis) * positionLambda;
                parentPosition -= rowAxis * positionLambda * links[parent].invMass;
                childPosition += rowAxis * positionLambda * links[child].invMass;
                parentRotation = normalize(quat_mul(quat_from_rotation_vector(parentAngularDelta), parentRotation));
                childRotation = normalize(quat_mul(quat_from_rotation_vector(childAngularDelta), childRotation));

                const float3 updatedParentAnchorOffset = quat_rotate(parentRotation, anchorParent);
                const float3 updatedChildAnchorOffset = quat_rotate(childRotation, anchorChild);
                const float3 parentAnchorVelocity = parentVelocity + cross(parentAngularVelocity, updatedParentAnchorOffset);
                const float3 childAnchorVelocity = childVelocity + cross(childAngularVelocity, updatedChildAnchorOffset);
                const float relativeVelocity = dot(childAnchorVelocity - parentAnchorVelocity, rowAxis);
                const float velocityLambda = (errorAlongAxis * velocityScale - relativeVelocity) / rowInvMass;
                jointAnchorImpulses[impulseBase + row] += velocityLambda;
                maxImpulseMagnitude = max(maxImpulseMagnitude, abs(jointAnchorImpulses[impulseBase + row]));
                const float3 updatedParentAngularAxis = cross(updatedParentAnchorOffset, rowAxis);
                const float3 updatedChildAngularAxis = cross(updatedChildAnchorOffset, rowAxis);
                parentVelocity -= rowAxis * velocityLambda * links[parent].invMass;
                childVelocity += rowAxis * velocityLambda * links[child].invMass;
                parentAngularVelocity -= inverse_inertia_mul_world(links[parent], parentRotation, updatedParentAngularAxis) * velocityLambda;
                childAngularVelocity += inverse_inertia_mul_world(links[child], childRotation, updatedChildAngularAxis) * velocityLambda;
            }

            store_float3(linkPositions, parentBase3, parentPosition);
            store_float3(linkPositions, childBase3, childPosition);
            store_quat(linkRotations, parentBase4, parentRotation);
            store_quat(linkRotations, childBase4, childRotation);
            store_float3(linkLinearVelocities, parentBase3, parentVelocity);
            store_float3(linkLinearVelocities, childBase3, childVelocity);
            store_float3(linkAngularVelocities, parentBase3, parentAngularVelocity);
            store_float3(linkAngularVelocities, childBase3, childAngularVelocity);

            float3 angularError = float3(0.0f);
            float3 angularRowAxes[3];
            uint angularRowCount = 0u;
            const float4 parentJointFrame = normalize(quat_mul(parentRotation, parentFrame));
            if (joint.type == 1u || joint.type == 3u) {
                const float4 targetChildRotation = normalize(quat_mul(parentJointFrame, quat_conjugate(childFrame)));
                angularError = quat_to_rotation_vector(quat_mul(targetChildRotation, quat_conjugate(childRotation)));
                angularRowAxes[0] = normalize(quat_rotate(parentJointFrame, float3(1.0f, 0.0f, 0.0f)));
                angularRowAxes[1] = normalize(quat_rotate(parentJointFrame, float3(0.0f, 1.0f, 0.0f)));
                angularRowAxes[2] = normalize(quat_rotate(parentJointFrame, float3(0.0f, 0.0f, 1.0f)));
                angularRowCount = 3u;
            } else if (joint.type == 2u) {
                const float3 parentAxis = normalize(quat_rotate(parentJointFrame, float3(1.0f, 0.0f, 0.0f)));
                const float3 childAxis = normalize(quat_rotate(quat_mul(childRotation, childFrame), float3(1.0f, 0.0f, 0.0f)));
                angularError = cross(childAxis, parentAxis);
                angularRowAxes[0] = normalize(quat_rotate(parentJointFrame, float3(0.0f, 1.0f, 0.0f)));
                angularRowAxes[1] = normalize(quat_rotate(parentJointFrame, float3(0.0f, 0.0f, 1.0f)));
                angularRowAxes[2] = parentAxis;
                angularRowCount = 2u;
            }

            const uint angularImpulseBase = (gid * params.jointCount + jointIndex) * 3u;
            for (uint row = 0u; row < angularRowCount; ++row) {
                const float3 angularAxis = angularRowAxes[row];
                const float cachedLambda = jointAngularImpulses[angularImpulseBase + row];
                if (abs(cachedLambda) <= 1.0e-8f) {
                    continue;
                }
                parentAngularVelocity -= inverse_inertia_mul_world(links[parent], parentRotation, angularAxis) * cachedLambda;
                childAngularVelocity += inverse_inertia_mul_world(links[child], childRotation, angularAxis) * cachedLambda;
            }
            for (uint row = 0u; row < angularRowCount; ++row) {
                const float3 angularAxis = angularRowAxes[row];
                const float parentInvAngularMass = inverse_inertia_quadratic_world(links[parent], parentRotation, angularAxis);
                const float childInvAngularMass = inverse_inertia_quadratic_world(links[child], childRotation, angularAxis);
                const float rowInvMass = parentInvAngularMass + childInvAngularMass;
                if (rowInvMass <= 1.0e-7f) {
                    rowFailureCount += 1.0f;
                    continue;
                }

                const float errorAlongAxis = dot(angularError, angularAxis);
                maxConstraintError = max(maxConstraintError, abs(errorAlongAxis));
                const float positionLambda = errorAlongAxis / rowInvMass;
                const float3 parentAngularDelta = -inverse_inertia_mul_world(links[parent], parentRotation, angularAxis) * positionLambda;
                const float3 childAngularDelta = inverse_inertia_mul_world(links[child], childRotation, angularAxis) * positionLambda;
                parentRotation = normalize(quat_mul(quat_from_rotation_vector(parentAngularDelta), parentRotation));
                childRotation = normalize(quat_mul(quat_from_rotation_vector(childAngularDelta), childRotation));

                const float relativeAngularVelocity = dot(childAngularVelocity - parentAngularVelocity, angularAxis);
                const float velocityLambda = (errorAlongAxis * velocityScale - relativeAngularVelocity) / rowInvMass;
                jointAngularImpulses[angularImpulseBase + row] += velocityLambda;
                maxImpulseMagnitude = max(maxImpulseMagnitude, abs(jointAngularImpulses[angularImpulseBase + row]));
                parentAngularVelocity -= inverse_inertia_mul_world(links[parent], parentRotation, angularAxis) * velocityLambda;
                childAngularVelocity += inverse_inertia_mul_world(links[child], childRotation, angularAxis) * velocityLambda;
            }

            if (angularRowCount > 0u) {
                store_quat(linkRotations, parentBase4, parentRotation);
                store_quat(linkRotations, childBase4, childRotation);
                store_float3(linkAngularVelocities, parentBase3, parentAngularVelocity);
                store_float3(linkAngularVelocities, childBase3, childAngularVelocity);
            }
            if (!all(isfinite(parentPosition)) || !all(isfinite(childPosition)) ||
                !all(isfinite(parentRotation)) || !all(isfinite(childRotation)) ||
                !all(isfinite(parentVelocity)) || !all(isfinite(childVelocity)) ||
                !all(isfinite(parentAngularVelocity)) || !all(isfinite(childAngularVelocity))) {
                nonFiniteCount += 1.0f;
            }
        }
    }
    solverDiagnostics[diagnosticsBase + 0u] = maxConstraintError;
    solverDiagnostics[diagnosticsBase + 1u] = maxImpulseMagnitude;
    solverDiagnostics[diagnosticsBase + 2u] = nonFiniteCount;
    solverDiagnostics[diagnosticsBase + 3u] = rowFailureCount;
}

kernel void humanoid_measure_joint_anchor_errors(
    device const HumanoidJointGPUConstants *joints [[buffer(0)]],
    constant HumanoidEnvParams &params [[buffer(1)]],
    device const float *linkPositions [[buffer(2)]],
    device const float *linkRotations [[buffer(3)]],
    device float *jointAnchorErrors [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.envCount) {
        return;
    }

    const uint envLinkBase3 = gid * params.linkCount * 3u;
    const uint envLinkBase4 = gid * params.linkCount * 4u;
    const uint envJointBase = gid * params.jointCount;
    for (uint jointIndex = 0u; jointIndex < params.jointCount; ++jointIndex) {
        const HumanoidJointGPUConstants joint = joints[jointIndex];
        if (joint.type == 0u || joint.parentLink < 0 || joint.childLink < 0) {
            jointAnchorErrors[envJointBase + jointIndex] = 0.0f;
            continue;
        }

        const uint parent = uint(joint.parentLink);
        const uint child = uint(joint.childLink);
        const uint parentBase3 = envLinkBase3 + parent * 3u;
        const uint childBase3 = envLinkBase3 + child * 3u;
        const uint parentBase4 = envLinkBase4 + parent * 4u;
        const uint childBase4 = envLinkBase4 + child * 4u;

        const float3 parentPosition = load_float3(linkPositions, parentBase3);
        const float3 childPosition = load_float3(linkPositions, childBase3);
        const float4 parentRotation = load_quat(linkRotations, parentBase4);
        const float4 childRotation = load_quat(linkRotations, childBase4);
        const float3 anchorParent = joint_vec3(joint.anchorParentX, joint.anchorParentY, joint.anchorParentZ);
        const float3 anchorChild = joint_vec3(joint.anchorChildX, joint.anchorChildY, joint.anchorChildZ);
        const float3 worldParentAnchor = parentPosition + quat_rotate(parentRotation, anchorParent);
        const float3 worldChildAnchor = childPosition + quat_rotate(childRotation, anchorChild);
        jointAnchorErrors[envJointBase + jointIndex] = length(worldParentAnchor - worldChildAnchor);
    }
}

kernel void humanoid_detect_ground_contacts(
    device const HumanoidCollisionGPUConstants *collisions [[buffer(0)]],
    constant HumanoidEnvParams &params [[buffer(1)]],
    device const float *linkPositions [[buffer(2)]],
    device const float *linkRotations [[buffer(3)]],
    device float *contactPoints [[buffer(4)]],
    device float *contactNormals [[buffer(5)]],
    device float *contactPenetrations [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total = params.envCount * params.linkCount;
    if (gid >= total) {
        return;
    }

    const uint env = gid / params.linkCount;
    const uint link = gid - env * params.linkCount;
    const uint linkBase3 = (env * params.linkCount + link) * 3u;
    const uint linkBase4 = (env * params.linkCount + link) * 4u;
    const uint contactBase3 = gid * 3u;
    const HumanoidCollisionGPUConstants shape = collisions[link];

    contactNormals[contactBase3 + 0u] = 0.0f;
    contactNormals[contactBase3 + 1u] = 0.0f;
    contactNormals[contactBase3 + 2u] = 1.0f;
    contactPoints[contactBase3 + 0u] = 0.0f;
    contactPoints[contactBase3 + 1u] = 0.0f;
    contactPoints[contactBase3 + 2u] = 0.0f;
    contactPenetrations[gid] = 0.0f;

    if (shape.type == 0u) {
        return;
    }

    const float3 linkPosition = load_float3(linkPositions, linkBase3);
    const float4 linkRotation = load_quat(linkRotations, linkBase4);
    const float3 localCenter = float3(shape.translationX, shape.translationY, shape.translationZ);
    const float4 localRotation = normalize(float4(shape.rotationX, shape.rotationY, shape.rotationZ, shape.rotationW));
    const float4 worldRotation = normalize(quat_mul(linkRotation, localRotation));
    const float3 center = linkPosition + quat_rotate(linkRotation, localCenter);

    float supportDepth = 0.0f;
    float2 contactXY = center.xy;
    if (shape.type == 1u) {
        const float3 axisX = quat_rotate(worldRotation, float3(1.0f, 0.0f, 0.0f));
        const float3 axisY = quat_rotate(worldRotation, float3(0.0f, 1.0f, 0.0f));
        const float3 axisZ = quat_rotate(worldRotation, float3(0.0f, 0.0f, 1.0f));
        supportDepth = abs(axisX.z) * shape.paramX + abs(axisY.z) * shape.paramY + abs(axisZ.z) * shape.paramZ;
        contactXY = center.xy - axisX.xy * sign(axisX.z) * shape.paramX
            - axisY.xy * sign(axisY.z) * shape.paramY
            - axisZ.xy * sign(axisZ.z) * shape.paramZ;
    } else if (shape.type == 2u) {
        supportDepth = shape.paramX;
    } else if (shape.type == 3u || shape.type == 4u) {
        const float3 axisZ = quat_rotate(worldRotation, float3(0.0f, 0.0f, 1.0f));
        supportDepth = abs(axisZ.z) * shape.paramY + shape.paramX;
        contactXY = center.xy - axisZ.xy * sign(axisZ.z) * shape.paramY;
    }

    const float lowest = center.z - supportDepth;
    const float penetration = max(0.0f, -lowest);
    contactPenetrations[gid] = penetration;
    contactPoints[contactBase3 + 0u] = contactXY.x;
    contactPoints[contactBase3 + 1u] = contactXY.y;
    contactPoints[contactBase3 + 2u] = 0.0f;
}

float3 collision_center(
    const HumanoidCollisionGPUConstants shape,
    float3 linkPosition,
    float4 linkRotation
) {
    return linkPosition + quat_rotate(linkRotation, float3(shape.translationX, shape.translationY, shape.translationZ));
}

float4 collision_rotation(
    const HumanoidCollisionGPUConstants shape,
    float4 linkRotation
) {
    const float4 localRotation = normalize(float4(shape.rotationX, shape.rotationY, shape.rotationZ, shape.rotationW));
    return normalize(quat_mul(linkRotation, localRotation));
}

void capsule_segment(
    const HumanoidCollisionGPUConstants shape,
    float3 linkPosition,
    float4 linkRotation,
    thread float3 &a,
    thread float3 &b
) {
    const float3 center = collision_center(shape, linkPosition, linkRotation);
    const float4 rotation = collision_rotation(shape, linkRotation);
    const float3 axis = normalize(quat_rotate(rotation, float3(0.0f, 0.0f, 1.0f)));
    a = center - axis * shape.paramY;
    b = center + axis * shape.paramY;
}

void closest_points_segments(
    float3 p1,
    float3 q1,
    float3 p2,
    float3 q2,
    thread float3 &c1,
    thread float3 &c2
) {
    const float3 d1 = q1 - p1;
    const float3 d2 = q2 - p2;
    const float3 r = p1 - p2;
    const float a = dot(d1, d1);
    const float e = dot(d2, d2);
    const float f = dot(d2, r);
    float s = 0.0f;
    float t = 0.0f;

    if (a <= 1.0e-8f && e <= 1.0e-8f) {
        c1 = p1;
        c2 = p2;
        return;
    }
    if (a <= 1.0e-8f) {
        s = 0.0f;
        t = clamp(f / e, 0.0f, 1.0f);
    } else {
        const float c = dot(d1, r);
        if (e <= 1.0e-8f) {
            t = 0.0f;
            s = clamp(-c / a, 0.0f, 1.0f);
        } else {
            const float b = dot(d1, d2);
            const float denom = a * e - b * b;
            if (denom != 0.0f) {
                s = clamp((b * f - c * e) / denom, 0.0f, 1.0f);
            }
            t = (b * s + f) / e;
            if (t < 0.0f) {
                t = 0.0f;
                s = clamp(-c / a, 0.0f, 1.0f);
            } else if (t > 1.0f) {
                t = 1.0f;
                s = clamp((b - c) / a, 0.0f, 1.0f);
            }
        }
    }

    c1 = p1 + d1 * s;
    c2 = p2 + d2 * t;
}

float3 contact_normal_from_delta(float3 delta) {
    const float d = length(delta);
    if (d > 1.0e-7f) {
        return delta / d;
    }
    return float3(1.0f, 0.0f, 0.0f);
}

kernel void humanoid_detect_self_contacts(
    device const HumanoidCollisionGPUConstants *collisions [[buffer(0)]],
    device const HumanoidCollisionPairGPUConstants *pairs [[buffer(1)]],
    constant HumanoidEnvParams &params [[buffer(2)]],
    device const float *linkPositions [[buffer(3)]],
    device const float *linkRotations [[buffer(4)]],
    device float *contactPoints [[buffer(5)]],
    device float *contactNormals [[buffer(6)]],
    device float *contactPenetrations [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total = params.envCount * params.selfCollisionPairCount;
    if (gid >= total) {
        return;
    }

    const uint env = gid / params.selfCollisionPairCount;
    const uint pairIndex = gid - env * params.selfCollisionPairCount;
    const HumanoidCollisionPairGPUConstants pair = pairs[pairIndex];
    const uint linkA = pair.linkA;
    const uint linkB = pair.linkB;
    const uint linkABase3 = (env * params.linkCount + linkA) * 3u;
    const uint linkBBase3 = (env * params.linkCount + linkB) * 3u;
    const uint linkABase4 = (env * params.linkCount + linkA) * 4u;
    const uint linkBBase4 = (env * params.linkCount + linkB) * 4u;
    const uint contactBase3 = gid * 3u;

    contactPoints[contactBase3 + 0u] = 0.0f;
    contactPoints[contactBase3 + 1u] = 0.0f;
    contactPoints[contactBase3 + 2u] = 0.0f;
    contactNormals[contactBase3 + 0u] = 1.0f;
    contactNormals[contactBase3 + 1u] = 0.0f;
    contactNormals[contactBase3 + 2u] = 0.0f;
    contactPenetrations[gid] = 0.0f;

    const HumanoidCollisionGPUConstants shapeA = collisions[linkA];
    const HumanoidCollisionGPUConstants shapeB = collisions[linkB];
    const float3 linkPositionA = load_float3(linkPositions, linkABase3);
    const float3 linkPositionB = load_float3(linkPositions, linkBBase3);
    const float4 linkRotationA = load_quat(linkRotations, linkABase4);
    const float4 linkRotationB = load_quat(linkRotations, linkBBase4);

    float3 pointA = float3(0.0f);
    float3 pointB = float3(0.0f);
    float radiusSum = 0.0f;
    bool supported = false;
    if (shapeA.type == 2u && shapeB.type == 2u) {
        pointA = collision_center(shapeA, linkPositionA, linkRotationA);
        pointB = collision_center(shapeB, linkPositionB, linkRotationB);
        radiusSum = shapeA.paramX + shapeB.paramX;
        supported = true;
    } else if (shapeA.type == 3u && shapeB.type == 3u) {
        float3 a0;
        float3 a1;
        float3 b0;
        float3 b1;
        capsule_segment(shapeA, linkPositionA, linkRotationA, a0, a1);
        capsule_segment(shapeB, linkPositionB, linkRotationB, b0, b1);
        closest_points_segments(a0, a1, b0, b1, pointA, pointB);
        radiusSum = shapeA.paramX + shapeB.paramX;
        supported = true;
    }

    if (!supported) {
        return;
    }

    const float3 delta = pointB - pointA;
    const float distance = length(delta);
    const float penetration = max(0.0f, radiusSum - distance);
    if (penetration <= 0.0f) {
        return;
    }

    const float3 normal = contact_normal_from_delta(delta);
    const float3 contactPoint = 0.5f * (pointA + normal * shapeA.paramX + pointB - normal * shapeB.paramX);
    contactPoints[contactBase3 + 0u] = contactPoint.x;
    contactPoints[contactBase3 + 1u] = contactPoint.y;
    contactPoints[contactBase3 + 2u] = contactPoint.z;
    contactNormals[contactBase3 + 0u] = normal.x;
    contactNormals[contactBase3 + 1u] = normal.y;
    contactNormals[contactBase3 + 2u] = normal.z;
    contactPenetrations[gid] = penetration;
}

kernel void humanoid_solve_ground_contacts(
    constant HumanoidEnvParams &params [[buffer(0)]],
    device float *linkPositions [[buffer(1)]],
    device float *linkLinearVelocities [[buffer(2)]],
    device const float *contactNormals [[buffer(3)]],
    device const float *contactPenetrations [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total = params.envCount * params.linkCount;
    if (gid >= total) {
        return;
    }

    const float penetration = contactPenetrations[gid];
    const uint base3 = gid * 3u;
    if (penetration <= 0.0f) {
        return;
    }

    const float3 normal = normalize(load_float3(contactNormals, base3));
    float3 position = load_float3(linkPositions, base3);
    float3 velocity = load_float3(linkLinearVelocities, base3);

    position += normal * penetration;

    const float normalVelocity = dot(velocity, normal);
    const float normalImpulse = max(max(-normalVelocity, 0.0f), penetration * params.contactBaumgarte / max(params.dt, 1.0e-6f));
    if (normalVelocity < 0.0f) {
        velocity -= normal * normalVelocity;
    }

    const float3 normalComponent = normal * dot(velocity, normal);
    float3 tangentVelocity = velocity - normalComponent;
    const float tangentSpeed = length(tangentVelocity);
    if (tangentSpeed > 1.0e-7f) {
        const float frictionDelta = params.contactFriction * normalImpulse;
        const float remainingSpeed = max(0.0f, tangentSpeed - frictionDelta);
        tangentVelocity *= remainingSpeed / tangentSpeed;
    }
    velocity = normalComponent + tangentVelocity;

    store_float3(linkPositions, base3, position);
    store_float3(linkLinearVelocities, base3, velocity);
}

kernel void humanoid_sync_root_from_pelvis(
    constant HumanoidEnvParams &params [[buffer(0)]],
    device const float *linkPositions [[buffer(1)]],
    device const float *linkRotations [[buffer(2)]],
    device float *rootPositions [[buffer(3)]],
    device float *rootRotations [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.envCount) {
        return;
    }

    const uint linkBase3 = gid * params.linkCount * 3u;
    const uint linkBase4 = gid * params.linkCount * 4u;
    const uint rootBase3 = gid * 3u;
    const uint rootBase4 = gid * 4u;
    rootPositions[rootBase3 + 0u] = linkPositions[linkBase3 + 0u];
    rootPositions[rootBase3 + 1u] = linkPositions[linkBase3 + 1u];
    rootPositions[rootBase3 + 2u] = linkPositions[linkBase3 + 2u];
    rootRotations[rootBase4 + 0u] = linkRotations[linkBase4 + 0u];
    rootRotations[rootBase4 + 1u] = linkRotations[linkBase4 + 1u];
    rootRotations[rootBase4 + 2u] = linkRotations[linkBase4 + 2u];
    rootRotations[rootBase4 + 3u] = linkRotations[linkBase4 + 3u];
}

kernel void humanoid_forward_kinematics(
    device const HumanoidJointGPUConstants *joints [[buffer(0)]],
    constant HumanoidEnvParams &params [[buffer(1)]],
    device const float *rootPositions [[buffer(2)]],
    device const float *rootRotations [[buffer(3)]],
    device const float *jointPositions [[buffer(4)]],
    device float *linkPositions [[buffer(5)]],
    device float *linkRotations [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.envCount) {
        return;
    }

    const uint envDofBase = gid * params.dofCount;
    const uint envLinkBase3 = gid * params.linkCount * 3u;
    const uint envLinkBase4 = gid * params.linkCount * 4u;
    const uint rootBase3 = gid * 3u;
    const uint rootBase4 = gid * 4u;

    for (uint jointIndex = 0u; jointIndex < params.jointCount; ++jointIndex) {
        const HumanoidJointGPUConstants joint = joints[jointIndex];
        const uint child = uint(joint.childLink);
        const uint childBase3 = envLinkBase3 + child * 3u;
        const uint childBase4 = envLinkBase4 + child * 4u;

        if (joint.type == 0u) {
            linkPositions[childBase3 + 0u] = rootPositions[rootBase3 + 0u];
            linkPositions[childBase3 + 1u] = rootPositions[rootBase3 + 1u];
            linkPositions[childBase3 + 2u] = rootPositions[rootBase3 + 2u];
            linkRotations[childBase4 + 0u] = rootRotations[rootBase4 + 0u];
            linkRotations[childBase4 + 1u] = rootRotations[rootBase4 + 1u];
            linkRotations[childBase4 + 2u] = rootRotations[rootBase4 + 2u];
            linkRotations[childBase4 + 3u] = rootRotations[rootBase4 + 3u];
            continue;
        }

        const uint parent = uint(joint.parentLink);
        const uint parentBase3 = envLinkBase3 + parent * 3u;
        const uint parentBase4 = envLinkBase4 + parent * 4u;
        const float3 parentPosition = float3(
            linkPositions[parentBase3 + 0u],
            linkPositions[parentBase3 + 1u],
            linkPositions[parentBase3 + 2u]
        );
        const float4 parentRotation = normalize(float4(
            linkRotations[parentBase4 + 0u],
            linkRotations[parentBase4 + 1u],
            linkRotations[parentBase4 + 2u],
            linkRotations[parentBase4 + 3u]
        ));

        const float3 anchorParent = joint_vec3(joint.anchorParentX, joint.anchorParentY, joint.anchorParentZ);
        const float3 anchorChild = joint_vec3(joint.anchorChildX, joint.anchorChildY, joint.anchorChildZ);
        const float3 worldAnchor = parentPosition + quat_rotate(parentRotation, anchorParent);
        const float4 childRotation = normalize(quat_mul(parentRotation, joint_rotation(joint, jointPositions, envDofBase)));
        const float3 childPosition = worldAnchor - quat_rotate(childRotation, anchorChild);

        linkPositions[childBase3 + 0u] = childPosition.x;
        linkPositions[childBase3 + 1u] = childPosition.y;
        linkPositions[childBase3 + 2u] = childPosition.z;
        linkRotations[childBase4 + 0u] = childRotation.x;
        linkRotations[childBase4 + 1u] = childRotation.y;
        linkRotations[childBase4 + 2u] = childRotation.z;
        linkRotations[childBase4 + 3u] = childRotation.w;
    }
}

kernel void humanoid_write_outputs(
    constant HumanoidEnvParams &params [[buffer(0)]],
    device const float *rootPositions [[buffer(1)]],
    device const float *rootRotations [[buffer(2)]],
    device const float *jointPositions [[buffer(3)]],
    device const float *jointVelocities [[buffer(4)]],
    device const float *linkPositions [[buffer(5)]],
    device float *observations [[buffer(6)]],
    device float *rewards [[buffer(7)]],
    device uint *dones [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.envCount) {
        return;
    }

    const uint obsBase = gid * params.observationDim;
    const uint rootBase3 = gid * 3u;
    const uint rootBase4 = gid * 4u;
    const uint dofBase = gid * params.dofCount;
    observations[obsBase + 0u] = rootPositions[rootBase3 + 0u];
    observations[obsBase + 1u] = rootPositions[rootBase3 + 1u];
    observations[obsBase + 2u] = rootPositions[rootBase3 + 2u];
    observations[obsBase + 3u] = rootRotations[rootBase4 + 0u];
    observations[obsBase + 4u] = rootRotations[rootBase4 + 1u];
    observations[obsBase + 5u] = rootRotations[rootBase4 + 2u];
    observations[obsBase + 6u] = rootRotations[rootBase4 + 3u];
    for (uint i = 0u; i < params.dofCount; ++i) {
        observations[obsBase + 7u + i] = jointPositions[dofBase + i];
        observations[obsBase + 7u + params.dofCount + i] = jointVelocities[dofBase + i];
    }

    const uint pelvisBase = gid * params.linkCount * 3u;
    const float pelvisHeight = linkPositions[pelvisBase + 2u];
    const bool fallen = pelvisHeight < params.rootHeightMin;
    dones[gid] = fallen ? 1u : 0u;
    rewards[gid] = fallen ? 0.0f : pelvisHeight;
}
