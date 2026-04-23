import Foundation

func cartPoleObservation(from state: CartPoleState) -> [Float] {
    [state.x, state.xDot, state.theta, state.thetaDot]
}

func makeCartPoleInitialStates(count: Int) -> [CartPoleState] {
    (0..<count).map { index in
        let lane = Float(index)
        var state = CartPoleState(
            x: (lane.truncatingRemainder(dividingBy: 7.0) - 3.0) * 0.01,
            xDot: (lane.truncatingRemainder(dividingBy: 3.0) - 1.0) * 0.015,
            theta: (lane.truncatingRemainder(dividingBy: 5.0) - 2.0) * 0.02,
            thetaDot: (lane.truncatingRemainder(dividingBy: 4.0) - 1.5) * 0.01,
            reward: -1.0,
            done: 0
        )

        if index % 17 == 0 {
            state.x = 2.395
            state.xDot = 0.35
        }
        if index % 23 == 0 {
            state.theta = 0.205
            state.thetaDot = 0.18
        }

        return state
    }
}

func makeCartPoleActions(step: Int, count: Int) -> [Float] {
    (0..<count).map { index in
        switch (step * 7 + index * 3) % 5 {
        case 0:
            return -1.0
        case 1:
            return -0.5
        case 2:
            return 0.0
        case 3:
            return 0.5
        default:
            return 1.0
        }
    }
}

func mixBits(_ value: UInt32) -> UInt32 {
    var x = value &+ 0x9E37_79B9
    x = (x ^ (x >> 16)) &* 0x85EB_CA6B
    x = (x ^ (x >> 13)) &* 0xC2B2_AE35
    return x ^ (x >> 16)
}

func uniform01(_ value: UInt32) -> Float {
    Float(value & 0x00FF_FFFF) / 16_777_215.0
}

func centeredRandom(baseSeed: UInt32, lane: UInt32, resetCount: UInt32, salt: UInt32, scale: Float) -> Float {
    let hash = mixBits(baseSeed ^ (lane &* 0x45D9_F3B) ^ (resetCount &* 0x27D4_EB2D) ^ salt)
    return (uniform01(hash) * 2.0 - 1.0) * scale
}

func makeCartPoleResetState(lane: UInt32, resetCount: UInt32, params: ResetParams) -> CartPoleState {
    CartPoleState(
        x: centeredRandom(baseSeed: params.baseSeed, lane: lane, resetCount: resetCount, salt: 0xA511_E9B3, scale: 0.05),
        xDot: centeredRandom(baseSeed: params.baseSeed, lane: lane, resetCount: resetCount, salt: 0x63D8_3595, scale: 0.05),
        theta: centeredRandom(baseSeed: params.baseSeed, lane: lane, resetCount: resetCount, salt: 0xC2B2_AE35, scale: 0.05),
        thetaDot: centeredRandom(baseSeed: params.baseSeed, lane: lane, resetCount: resetCount, salt: 0x27D4_EB2F, scale: 0.05),
        reward: 0.0,
        done: 0
    )
}

func stepCartPoleReference(
    state: CartPoleState,
    action: Float,
    params: CartPoleParams
) -> CartPoleState {
    if state.done != 0 {
        return CartPoleState(
            x: state.x,
            xDot: state.xDot,
            theta: state.theta,
            thetaDot: state.thetaDot,
            reward: 0.0,
            done: state.done
        )
    }

    let totalMass = params.massCart + params.massPole
    let poleMassLength = params.massPole * params.halfPoleLength
    let appliedForce = min(max(action, -1.0), 1.0) * params.forceMag

    let cosTheta = cos(state.theta)
    let sinTheta = sin(state.theta)
    let temp = (appliedForce + poleMassLength * state.thetaDot * state.thetaDot * sinTheta) / totalMass
    let thetaAcc = (
        params.gravity * sinTheta - cosTheta * temp
    ) / (
        params.halfPoleLength * (4.0 / 3.0 - params.massPole * cosTheta * cosTheta / totalMass)
    )
    let xAcc = temp - poleMassLength * thetaAcc * cosTheta / totalMass

    var next = state
    next.x += params.dt * state.xDot
    next.xDot += params.dt * xAcc
    next.theta += params.dt * state.thetaDot
    next.thetaDot += params.dt * thetaAcc

    let outOfBounds = abs(next.x) > params.xThreshold ||
        abs(next.theta) > params.thetaThresholdRadians

    next.done = outOfBounds ? 1 : 0
    next.reward = outOfBounds ? 0.0 : 1.0
    return next
}

func resetDoneReference(
    state: CartPoleState,
    lane: UInt32,
    resetCount: inout UInt32,
    params: ResetParams
) -> CartPoleState {
    if state.done == 0 {
        return state
    }

    resetCount &+= 1
    return makeCartPoleResetState(lane: lane, resetCount: resetCount, params: params)
}

func validateCartPoleState(
    actual: CartPoleState,
    expected: CartPoleState,
    phase: String,
    step: Int,
    index: Int,
    tolerance: Float
) throws {
    func checkFloat(_ field: String, _ actualValue: Float, _ expectedValue: Float) throws {
        if abs(actualValue - expectedValue) > tolerance {
            throw EnvProjectError.validationFailed(
                message: "Validation failed during \(phase) at step \(step), lane \(index), field \(field): expected \(expectedValue), got \(actualValue)."
            )
        }
    }

    try checkFloat("x", actual.x, expected.x)
    try checkFloat("xDot", actual.xDot, expected.xDot)
    try checkFloat("theta", actual.theta, expected.theta)
    try checkFloat("thetaDot", actual.thetaDot, expected.thetaDot)
    try checkFloat("reward", actual.reward, expected.reward)

    if actual.done != expected.done {
        throw EnvProjectError.validationFailed(
            message: "Validation failed during \(phase) at step \(step), lane \(index), field done: expected \(expected.done), got \(actual.done)."
        )
    }
}

func validateCartPoleOutputs(
    observations: [Float],
    rewards: [Float],
    dones: [UInt32],
    expectedStates: [CartPoleState],
    phase: String,
    step: Int,
    tolerance: Float
) throws {
    let elementsPerObservation = 4

    if observations.count != expectedStates.count * elementsPerObservation {
        throw EnvProjectError.validationFailed(
            message: "Observation buffer size mismatch during \(phase) at step \(step): expected \(expectedStates.count * elementsPerObservation), got \(observations.count)."
        )
    }
    if rewards.count != expectedStates.count {
        throw EnvProjectError.validationFailed(
            message: "Reward buffer size mismatch during \(phase) at step \(step): expected \(expectedStates.count), got \(rewards.count)."
        )
    }
    if dones.count != expectedStates.count {
        throw EnvProjectError.validationFailed(
            message: "Done buffer size mismatch during \(phase) at step \(step): expected \(expectedStates.count), got \(dones.count)."
        )
    }

    for index in expectedStates.indices {
        let expectedObservation = cartPoleObservation(from: expectedStates[index])
        let base = index * elementsPerObservation

        for offset in 0..<elementsPerObservation {
            let actual = observations[base + offset]
            let expected = expectedObservation[offset]
            if abs(actual - expected) > tolerance {
                throw EnvProjectError.validationFailed(
                    message: "Observation validation failed during \(phase) at step \(step), lane \(index), element \(offset): expected \(expected), got \(actual)."
                )
            }
        }

        let expectedReward = expectedStates[index].reward
        if abs(rewards[index] - expectedReward) > tolerance {
            throw EnvProjectError.validationFailed(
                message: "Reward validation failed during \(phase) at step \(step), lane \(index): expected \(expectedReward), got \(rewards[index])."
            )
        }

        let expectedDone = expectedStates[index].done
        if dones[index] != expectedDone {
            throw EnvProjectError.validationFailed(
                message: "Done validation failed during \(phase) at step \(step), lane \(index): expected \(expectedDone), got \(dones[index])."
            )
        }
    }
}
