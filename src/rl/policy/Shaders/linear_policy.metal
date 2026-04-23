#include <metal_stdlib>
using namespace metal;

struct LinearPolicyParams {
    uint envCount;
    uint observationDim;
    uint actionDim;
    float minAction;
    float maxAction;
};

kernel void linear_policy_forward(
    device const float *observations [[buffer(0)]],
    device const float *weights [[buffer(1)]],
    device const float *bias [[buffer(2)]],
    device float *actions [[buffer(3)]],
    constant LinearPolicyParams &params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint totalActions = params.envCount * params.actionDim;
    if (gid >= totalActions) {
        return;
    }

    const uint envIndex = gid / params.actionDim;
    const uint actionIndex = gid % params.actionDim;
    const uint obsBase = envIndex * params.observationDim;
    const uint weightBase = actionIndex * params.observationDim;

    float value = bias[actionIndex];
    for (uint obsIndex = 0; obsIndex < params.observationDim; ++obsIndex) {
        value += weights[weightBase + obsIndex] * observations[obsBase + obsIndex];
    }

    actions[gid] = clamp(value, params.minAction, params.maxAction);
}
