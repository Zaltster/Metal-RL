#include <metal_stdlib>
using namespace metal;

struct CartPoleState {
    float x;
    float xDot;
    float theta;
    float thetaDot;
    float reward;
    uint done;
};

struct CartPoleParams {
    uint envCount;
    float dt;
    float gravity;
    float massCart;
    float massPole;
    float halfPoleLength;
    float forceMag;
    float xThreshold;
    float thetaThresholdRadians;
};

struct ResetParams {
    uint envCount;
    uint baseSeed;
};

uint mix_bits(uint value) {
    uint x = value + 0x9E3779B9u;
    x = (x ^ (x >> 16)) * 0x85EBCA6Bu;
    x = (x ^ (x >> 13)) * 0xC2B2AE35u;
    return x ^ (x >> 16);
}

float uniform01(uint value) {
    return float(value & 0x00FFFFFFu) / 16777215.0f;
}

float centered_random(uint baseSeed, uint lane, uint resetCount, uint salt, float scale) {
    uint hash = mix_bits(baseSeed ^ (lane * 0x045D9F3Bu) ^ (resetCount * 0x27D4EB2Du) ^ salt);
    return (uniform01(hash) * 2.0f - 1.0f) * scale;
}

CartPoleState make_reset_state(uint lane, uint resetCount, constant ResetParams &params) {
    CartPoleState state;
    state.x = centered_random(params.baseSeed, lane, resetCount, 0xA511E9B3u, 0.05f);
    state.xDot = centered_random(params.baseSeed, lane, resetCount, 0x63D83595u, 0.05f);
    state.theta = centered_random(params.baseSeed, lane, resetCount, 0xC2B2AE35u, 0.05f);
    state.thetaDot = centered_random(params.baseSeed, lane, resetCount, 0x27D4EB2Fu, 0.05f);
    state.reward = 0.0f;
    state.done = 0u;
    return state;
}

kernel void step_cartpole(
    device CartPoleState *states [[buffer(0)]],
    device const float *actions [[buffer(1)]],
    constant CartPoleParams &params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.envCount) {
        return;
    }

    CartPoleState state = states[gid];
    if (state.done != 0u) {
        state.reward = 0.0f;
        states[gid] = state;
        return;
    }

    const float totalMass = params.massCart + params.massPole;
    const float poleMassLength = params.massPole * params.halfPoleLength;
    const float appliedForce = clamp(actions[gid], -1.0f, 1.0f) * params.forceMag;

    const float cosTheta = cos(state.theta);
    const float sinTheta = sin(state.theta);
    const float temp = (appliedForce + poleMassLength * state.thetaDot * state.thetaDot * sinTheta) / totalMass;
    const float thetaAcc = (
        params.gravity * sinTheta - cosTheta * temp
    ) / (
        params.halfPoleLength * (4.0f / 3.0f - params.massPole * cosTheta * cosTheta / totalMass)
    );
    const float xAcc = temp - poleMassLength * thetaAcc * cosTheta / totalMass;

    state.x += params.dt * state.xDot;
    state.xDot += params.dt * xAcc;
    state.theta += params.dt * state.thetaDot;
    state.thetaDot += params.dt * thetaAcc;

    const bool outOfBounds = fabs(state.x) > params.xThreshold ||
        fabs(state.theta) > params.thetaThresholdRadians;

    state.done = outOfBounds ? 1u : 0u;
    state.reward = outOfBounds ? 0.0f : 1.0f;
    states[gid] = state;
}

kernel void reset_done_cartpoles(
    device CartPoleState *states [[buffer(0)]],
    device uint *resetCounts [[buffer(1)]],
    constant ResetParams &params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.envCount) {
        return;
    }

    CartPoleState state = states[gid];
    if (state.done == 0u) {
        return;
    }

    uint resetCount = resetCounts[gid] + 1u;
    resetCounts[gid] = resetCount;
    states[gid] = make_reset_state(gid, resetCount, params);
}

kernel void write_cartpole_outputs(
    device const CartPoleState *states [[buffer(0)]],
    device float *observations [[buffer(1)]],
    device float *rewards [[buffer(2)]],
    device uint *dones [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    const CartPoleState state = states[gid];
    const uint base = gid * 4u;
    observations[base + 0u] = state.x;
    observations[base + 1u] = state.xDot;
    observations[base + 2u] = state.theta;
    observations[base + 3u] = state.thetaDot;
    rewards[gid] = state.reward;
    dones[gid] = state.done;
}
