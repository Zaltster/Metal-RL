#include <metal_stdlib>
using namespace metal;

struct MLPPolicyParams {
    uint envCount;
    uint observationDim;
    uint hiddenDim;
    uint actionDim;
    float minAction;
    float maxAction;
};

kernel void mlp_policy_forward(
    device const float *observations [[buffer(0)]],
    device const float *inputWeights [[buffer(1)]],
    device const float *inputBias [[buffer(2)]],
    device const float *outputWeights [[buffer(3)]],
    device const float *outputBias [[buffer(4)]],
    device const float *valueWeights [[buffer(5)]],
    device const float *valueBias [[buffer(6)]],
    device float *actions [[buffer(7)]],
    device float *values [[buffer(8)]],
    constant MLPPolicyParams &params [[buffer(9)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.envCount) {
        return;
    }

    const uint envIndex = gid;
    const uint obsBase = envIndex * params.observationDim;
    const uint actionBase = envIndex * params.actionDim;

    constexpr uint kMaxHiddenDim = 64;
    if (params.hiddenDim > kMaxHiddenDim) {
        return;
    }

    float hidden[kMaxHiddenDim];
    for (uint hiddenIndex = 0; hiddenIndex < params.hiddenDim; ++hiddenIndex) {
        const uint inputWeightBase = hiddenIndex * params.observationDim;
        float value = inputBias[hiddenIndex];
        for (uint obsIndex = 0; obsIndex < params.observationDim; ++obsIndex) {
            value += inputWeights[inputWeightBase + obsIndex] * observations[obsBase + obsIndex];
        }
        hidden[hiddenIndex] = max(0.0f, value);
    }

    for (uint actionIndex = 0; actionIndex < params.actionDim; ++actionIndex) {
        const uint outputWeightBase = actionIndex * params.hiddenDim;
        float actionValue = outputBias[actionIndex];
        for (uint hiddenIndex = 0; hiddenIndex < params.hiddenDim; ++hiddenIndex) {
            actionValue += outputWeights[outputWeightBase + hiddenIndex] * hidden[hiddenIndex];
        }
        actions[actionBase + actionIndex] = clamp(actionValue, params.minAction, params.maxAction);
    }

    float stateValue = valueBias[0];
    for (uint hiddenIndex = 0; hiddenIndex < params.hiddenDim; ++hiddenIndex) {
        stateValue += valueWeights[hiddenIndex] * hidden[hiddenIndex];
    }
    values[envIndex] = stateValue;
}
