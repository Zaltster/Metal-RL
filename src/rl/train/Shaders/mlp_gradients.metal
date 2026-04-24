#include <metal_stdlib>
using namespace metal;

struct MLPGradientParams {
    uint sampleCount;
    uint observationDim;
    uint hiddenDim;
    uint actionDim;
    float clipEpsilon;
    float valueCoefficient;
};

struct MLPGradientReductionParams {
    uint sampleCount;
    uint parameterCount;
};

struct MLPSGDUpdateParams {
    float learningRate;
    uint parameterCount;
};

struct MLPAdamUpdateParams {
    float learningRate;
    float beta1;
    float beta2;
    float epsilon;
    float biasCorrection1;
    float biasCorrection2;
    uint parameterCount;
};

kernel void mlp_ppo_per_sample_gradients(
    device const float *observations [[buffer(0)]],
    device const float *actions [[buffer(1)]],
    device const float *oldLogProbs [[buffer(2)]],
    device const float *advantages [[buffer(3)]],
    device const float *returns [[buffer(4)]],
    device const float *inputWeights [[buffer(5)]],
    device const float *inputBias [[buffer(6)]],
    device const float *outputWeights [[buffer(7)]],
    device const float *outputBias [[buffer(8)]],
    device const float *valueWeights [[buffer(9)]],
    device const float *valueBias [[buffer(10)]],
    device const float *logStd [[buffer(11)]],
    device float *inputWeightGrads [[buffer(12)]],
    device float *inputBiasGrads [[buffer(13)]],
    device float *outputWeightGrads [[buffer(14)]],
    device float *outputBiasGrads [[buffer(15)]],
    device float *valueWeightGrads [[buffer(16)]],
    device float *valueBiasGrads [[buffer(17)]],
    constant MLPGradientParams &params [[buffer(18)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.sampleCount) {
        return;
    }

    constexpr uint kMaxObservationDim = 64;
    constexpr uint kMaxHiddenDim = 64;
    constexpr uint kMaxActionDim = 8;
    if (params.observationDim > kMaxObservationDim ||
        params.hiddenDim > kMaxHiddenDim ||
        params.actionDim > kMaxActionDim) {
        return;
    }

    const uint sampleIndex = gid;
    const uint obsBase = sampleIndex * params.observationDim;
    const uint actionBase = sampleIndex * params.actionDim;
    const float batchScale = 1.0f / float(params.sampleCount);

    float preActivation[kMaxHiddenDim];
    float hidden[kMaxHiddenDim];
    for (uint hiddenIndex = 0; hiddenIndex < params.hiddenDim; ++hiddenIndex) {
        const uint inputWeightBase = hiddenIndex * params.observationDim;
        float value = inputBias[hiddenIndex];
        for (uint obsIndex = 0; obsIndex < params.observationDim; ++obsIndex) {
            value += inputWeights[inputWeightBase + obsIndex] * observations[obsBase + obsIndex];
        }
        preActivation[hiddenIndex] = value;
        hidden[hiddenIndex] = max(0.0f, value);
    }

    float mean[kMaxActionDim];
    for (uint actionIndex = 0; actionIndex < params.actionDim; ++actionIndex) {
        const uint outputWeightBase = actionIndex * params.hiddenDim;
        float value = outputBias[actionIndex];
        for (uint hiddenIndex = 0; hiddenIndex < params.hiddenDim; ++hiddenIndex) {
            value += outputWeights[outputWeightBase + hiddenIndex] * hidden[hiddenIndex];
        }
        mean[actionIndex] = value;
    }

    float stateValue = valueBias[0];
    for (uint hiddenIndex = 0; hiddenIndex < params.hiddenDim; ++hiddenIndex) {
        stateValue += valueWeights[hiddenIndex] * hidden[hiddenIndex];
    }

    constexpr float kLogTwoPi = 1.8378770664093453f;
    float newLogProb = 0.0f;
    for (uint actionIndex = 0; actionIndex < params.actionDim; ++actionIndex) {
        const float diff = actions[actionBase + actionIndex] - mean[actionIndex];
        const float currentLogStd = logStd[actionIndex];
        const float variance = exp(2.0f * currentLogStd);
        newLogProb += -0.5f * ((diff * diff) / variance + 2.0f * currentLogStd + kLogTwoPi);
    }

    const float ratio = exp(newLogProb - oldLogProbs[sampleIndex]);
    const float clippedRatio = clamp(ratio, 1.0f - params.clipEpsilon, 1.0f + params.clipEpsilon);
    const float advantage = advantages[sampleIndex];
    const float surrogate1 = ratio * advantage;
    const float surrogate2 = clippedRatio * advantage;

    float dLoss_dLogProb = 0.0f;
    if (surrogate1 < surrogate2) {
        dLoss_dLogProb = -(ratio * advantage) * batchScale;
    }

    float dLoss_dMean[kMaxActionDim];
    for (uint actionIndex = 0; actionIndex < params.actionDim; ++actionIndex) {
        const float variance = exp(2.0f * logStd[actionIndex]);
        dLoss_dMean[actionIndex] =
            dLoss_dLogProb * ((actions[actionBase + actionIndex] - mean[actionIndex]) / variance);
    }

    const float dLoss_dValue =
        params.valueCoefficient * (stateValue - returns[sampleIndex]) * batchScale;

    const uint inputWeightGradBase = sampleIndex * params.hiddenDim * params.observationDim;
    const uint outputWeightGradBase = sampleIndex * params.actionDim * params.hiddenDim;
    const uint valueWeightGradBase = sampleIndex * params.hiddenDim;

    for (uint actionIndex = 0; actionIndex < params.actionDim; ++actionIndex) {
        const uint outputWeightBase = actionIndex * params.hiddenDim;
        for (uint hiddenIndex = 0; hiddenIndex < params.hiddenDim; ++hiddenIndex) {
            outputWeightGrads[outputWeightGradBase + outputWeightBase + hiddenIndex] =
                dLoss_dMean[actionIndex] * hidden[hiddenIndex];
        }
        outputBiasGrads[sampleIndex * params.actionDim + actionIndex] = dLoss_dMean[actionIndex];
    }

    for (uint hiddenIndex = 0; hiddenIndex < params.hiddenDim; ++hiddenIndex) {
        valueWeightGrads[valueWeightGradBase + hiddenIndex] = dLoss_dValue * hidden[hiddenIndex];
    }
    valueBiasGrads[sampleIndex] = dLoss_dValue;

    float dLoss_dHidden[kMaxHiddenDim];
    for (uint hiddenIndex = 0; hiddenIndex < params.hiddenDim; ++hiddenIndex) {
        float total = valueWeights[hiddenIndex] * dLoss_dValue;
        for (uint actionIndex = 0; actionIndex < params.actionDim; ++actionIndex) {
            total += outputWeights[actionIndex * params.hiddenDim + hiddenIndex] * dLoss_dMean[actionIndex];
        }
        dLoss_dHidden[hiddenIndex] = preActivation[hiddenIndex] > 0.0f ? total : 0.0f;
    }

    for (uint hiddenIndex = 0; hiddenIndex < params.hiddenDim; ++hiddenIndex) {
        const uint inputWeightBase = hiddenIndex * params.observationDim;
        for (uint obsIndex = 0; obsIndex < params.observationDim; ++obsIndex) {
            inputWeightGrads[inputWeightGradBase + inputWeightBase + obsIndex] =
                dLoss_dHidden[hiddenIndex] * observations[obsBase + obsIndex];
        }
        inputBiasGrads[sampleIndex * params.hiddenDim + hiddenIndex] = dLoss_dHidden[hiddenIndex];
    }
}

kernel void mlp_reduce_per_sample_gradients(
    device const float *perSampleGradients [[buffer(0)]],
    device float *reducedGradients [[buffer(1)]],
    constant MLPGradientReductionParams &params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.parameterCount) {
        return;
    }

    float total = 0.0f;
    for (uint sampleIndex = 0; sampleIndex < params.sampleCount; ++sampleIndex) {
        total += perSampleGradients[sampleIndex * params.parameterCount + gid];
    }
    reducedGradients[gid] = total;
}

kernel void mlp_sgd_update(
    device float *parameters [[buffer(0)]],
    device const float *gradients [[buffer(1)]],
    constant MLPSGDUpdateParams &params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.parameterCount) {
        return;
    }

    parameters[gid] -= params.learningRate * gradients[gid];
}

kernel void mlp_adam_update(
    device float *parameters [[buffer(0)]],
    device const float *gradients [[buffer(1)]],
    device float *momentum [[buffer(2)]],
    device float *velocity [[buffer(3)]],
    constant MLPAdamUpdateParams &params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.parameterCount) {
        return;
    }

    const float grad = gradients[gid];
    const float m = params.beta1 * momentum[gid] + (1.0f - params.beta1) * grad;
    const float v = params.beta2 * velocity[gid] + (1.0f - params.beta2) * grad * grad;
    momentum[gid] = m;
    velocity[gid] = v;

    const float mHat = m / params.biasCorrection1;
    const float vHat = v / params.biasCorrection2;
    parameters[gid] -= params.learningRate * mHat / (sqrt(vHat) + params.epsilon);
}
