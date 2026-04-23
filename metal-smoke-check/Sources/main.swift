import Foundation
import Metal

func compareRolloutActions(lhs: VectorRollout, rhs: VectorRollout, tolerance: Float) throws {
    if lhs.steps.count != rhs.steps.count {
        throw EnvProjectError.validationFailed(
            message: "Random-policy rollout length mismatch: expected \(lhs.steps.count), got \(rhs.steps.count)."
        )
    }

    for stepIndex in lhs.steps.indices {
        let left = lhs.steps[stepIndex]
        let right = rhs.steps[stepIndex]

        if left.actions.count != right.actions.count {
            throw EnvProjectError.validationFailed(
                message: "Random-policy action width mismatch at step \(stepIndex): expected \(left.actions.count), got \(right.actions.count)."
            )
        }

        for actionIndex in left.actions.indices {
            let expected = left.actions[actionIndex]
            let actual = right.actions[actionIndex]
            if abs(expected - actual) > tolerance {
                throw EnvProjectError.validationFailed(
                    message: "Random-policy action mismatch at step \(stepIndex), index \(actionIndex): expected \(expected), got \(actual)."
                )
            }
        }
    }
}

func compareRolloutBatches(lhs: VectorRollout, rhs: VectorRollout, tolerance: Float) throws {
    func compareFloatArrays(_ name: String, _ lhs: [Float], _ rhs: [Float], stepIndex: Int?) throws {
        if lhs.count != rhs.count {
            throw EnvProjectError.validationFailed(
                message: "\(name) length mismatch\(stepIndex.map { " at step \($0)" } ?? ""): expected \(lhs.count), got \(rhs.count)."
            )
        }
        for index in lhs.indices {
            if abs(lhs[index] - rhs[index]) > tolerance {
                throw EnvProjectError.validationFailed(
                    message: "\(name) mismatch\(stepIndex.map { " at step \($0)" } ?? "") index \(index): expected \(lhs[index]), got \(rhs[index])."
                )
            }
        }
    }

    func compareUIntArrays(_ name: String, _ lhs: [UInt32], _ rhs: [UInt32], stepIndex: Int?) throws {
        if lhs.count != rhs.count {
            throw EnvProjectError.validationFailed(
                message: "\(name) length mismatch\(stepIndex.map { " at step \($0)" } ?? ""): expected \(lhs.count), got \(rhs.count)."
            )
        }
        for index in lhs.indices {
            if lhs[index] != rhs[index] {
                throw EnvProjectError.validationFailed(
                    message: "\(name) mismatch\(stepIndex.map { " at step \($0)" } ?? "") index \(index): expected \(lhs[index]), got \(rhs[index])."
                )
            }
        }
    }

    try compareFloatArrays("initial observations", lhs.initialBatch.observations, rhs.initialBatch.observations, stepIndex: nil)
    try compareFloatArrays("initial rewards", lhs.initialBatch.rewards, rhs.initialBatch.rewards, stepIndex: nil)
    try compareUIntArrays("initial dones", lhs.initialBatch.dones, rhs.initialBatch.dones, stepIndex: nil)
    try compareUIntArrays("initial resetCounts", lhs.initialBatch.resetCounts, rhs.initialBatch.resetCounts, stepIndex: nil)

    if lhs.steps.count != rhs.steps.count {
        throw EnvProjectError.validationFailed(
            message: "Random-policy rollout step-count mismatch: expected \(lhs.steps.count), got \(rhs.steps.count)."
        )
    }

    for stepIndex in lhs.steps.indices {
        let left = lhs.steps[stepIndex]
        let right = rhs.steps[stepIndex]
        try compareFloatArrays("observationsBefore", left.observationsBefore, right.observationsBefore, stepIndex: stepIndex)
        try compareFloatArrays("actions", left.actions, right.actions, stepIndex: stepIndex)
        try compareFloatArrays("rewards", left.rewards, right.rewards, stepIndex: stepIndex)
        try compareUIntArrays("dones", left.dones, right.dones, stepIndex: stepIndex)
        try compareFloatArrays("observationsAfterReset", left.observationsAfterReset, right.observationsAfterReset, stepIndex: stepIndex)
        try compareUIntArrays("resetCounts", left.resetCounts, right.resetCounts, stepIndex: stepIndex)
    }
}

func ensureDifferentActionSeedChangesRollout(reference: VectorRollout, alternate: VectorRollout, tolerance: Float) throws {
    for stepIndex in reference.steps.indices {
        let left = reference.steps[stepIndex]
        let right = alternate.steps[stepIndex]
        for actionIndex in left.actions.indices {
            if abs(left.actions[actionIndex] - right.actions[actionIndex]) > tolerance {
                return
            }
        }
    }

    throw EnvProjectError.validationFailed(message: "Different random-policy action seed produced no observable action change.")
}

func validateActionBounds(rollout: VectorRollout, actionSpec: VectorActionSpec, tolerance: Float) throws {
    for stepIndex in rollout.steps.indices {
        for actionIndex in rollout.steps[stepIndex].actions.indices {
            let action = rollout.steps[stepIndex].actions[actionIndex]
            if action < actionSpec.minValue - tolerance || action > actionSpec.maxValue + tolerance {
                throw EnvProjectError.validationFailed(
                    message: "Action bound violation at step \(stepIndex), index \(actionIndex): value \(action) outside [\(actionSpec.minValue), \(actionSpec.maxValue)]."
                )
            }
        }
    }
}

func ensureDifferentPoliciesChangeRollout(reference: VectorRollout, alternate: VectorRollout, tolerance: Float) throws {
    for stepIndex in reference.steps.indices {
        let left = reference.steps[stepIndex]
        let right = alternate.steps[stepIndex]
        for actionIndex in left.actions.indices {
            if abs(left.actions[actionIndex] - right.actions[actionIndex]) > tolerance {
                return
            }
        }
    }

    throw EnvProjectError.validationFailed(message: "Alternate linear policy produced no observable action change.")
}

func compareActionArrays(lhs: [Float], rhs: [Float], tolerance: Float, context: String) throws {
    if lhs.count != rhs.count {
        throw EnvProjectError.validationFailed(
            message: "\(context) action length mismatch: expected \(lhs.count), got \(rhs.count)."
        )
    }

    for index in lhs.indices {
        if abs(lhs[index] - rhs[index]) > tolerance {
            throw EnvProjectError.validationFailed(
                message: "\(context) action mismatch at index \(index): expected \(lhs[index]), got \(rhs[index])."
            )
        }
    }
}

func compareStorage(lhs: VectorRolloutStorage, rhs: VectorRolloutStorage, tolerance: Float, context: String) throws {
    if lhs.horizon != rhs.horizon || lhs.envCount != rhs.envCount || lhs.observationDim != rhs.observationDim || lhs.actionDim != rhs.actionDim {
        throw EnvProjectError.validationFailed(
            message: "\(context) storage shape mismatch."
        )
    }

    try compareActionArrays(lhs: lhs.observations, rhs: rhs.observations, tolerance: tolerance, context: "\(context) observations")
    try compareActionArrays(lhs: lhs.actions, rhs: rhs.actions, tolerance: tolerance, context: "\(context) actions")
    try compareActionArrays(lhs: lhs.rewards, rhs: rhs.rewards, tolerance: tolerance, context: "\(context) rewards")
    try compareActionArrays(lhs: lhs.nextObservations, rhs: rhs.nextObservations, tolerance: tolerance, context: "\(context) nextObservations")

    if lhs.dones != rhs.dones {
        throw EnvProjectError.validationFailed(
            message: "\(context) dones mismatch."
        )
    }
    if lhs.resetCounts != rhs.resetCounts {
        throw EnvProjectError.validationFailed(
            message: "\(context) resetCounts mismatch."
        )
    }
}

func validateStorageAgainstRollout(storage: VectorRolloutStorage, rollout: VectorRollout, tolerance: Float, context: String) throws {
    if storage.horizon != rollout.steps.count {
        throw EnvProjectError.validationFailed(
            message: "\(context) storage horizon mismatch: expected \(rollout.steps.count), got \(storage.horizon)."
        )
    }

    for step in 0..<storage.horizon {
        let rolloutStep = rollout.steps[step]
        for env in 0..<storage.envCount {
            let obsBase = env * storage.observationDim
            let actionBase = env * storage.actionDim

            let expectedObservation = Array(rolloutStep.observationsBefore[obsBase..<(obsBase + storage.observationDim)])
            try compareActionArrays(
                lhs: storage.observation(step: step, env: env),
                rhs: expectedObservation,
                tolerance: tolerance,
                context: "\(context) observation step \(step) env \(env)"
            )

            let expectedAction = Array(rolloutStep.actions[actionBase..<(actionBase + storage.actionDim)])
            try compareActionArrays(
                lhs: storage.action(step: step, env: env),
                rhs: expectedAction,
                tolerance: tolerance,
                context: "\(context) action step \(step) env \(env)"
            )

            let expectedNextObservation = Array(rolloutStep.observationsAfterReset[obsBase..<(obsBase + storage.observationDim)])
            try compareActionArrays(
                lhs: storage.nextObservation(step: step, env: env),
                rhs: expectedNextObservation,
                tolerance: tolerance,
                context: "\(context) nextObservation step \(step) env \(env)"
            )

            let expectedReward = rolloutStep.rewards[env]
            if abs(storage.reward(step: step, env: env) - expectedReward) > tolerance {
                throw EnvProjectError.validationFailed(
                    message: "\(context) reward mismatch at step \(step), env \(env): expected \(expectedReward), got \(storage.reward(step: step, env: env))."
                )
            }

            let expectedDone = rolloutStep.dones[env]
            if storage.done(step: step, env: env) != expectedDone {
                throw EnvProjectError.validationFailed(
                    message: "\(context) done mismatch at step \(step), env \(env): expected \(expectedDone), got \(storage.done(step: step, env: env))."
                )
            }

            let expectedResetCount = rolloutStep.resetCounts[env]
            if storage.resetCount(step: step, env: env) != expectedResetCount {
                throw EnvProjectError.validationFailed(
                    message: "\(context) resetCount mismatch at step \(step), env \(env): expected \(expectedResetCount), got \(storage.resetCount(step: step, env: env))."
                )
            }
        }
    }
}

func runValidatedRollout(
    driver: CartPoleVectorEnvDriver,
    cartPoleParams: CartPoleParams,
    resetSeed: UInt32,
    initialStates: [CartPoleState],
    horizon: Int,
    tolerance: Float
) throws -> CartPoleRolloutResult {
    driver.setResetSeed(resetSeed)
    _ = try driver.reset()

    let count = initialStates.count
    var cpuStates = initialStates
    var cpuResetCounts = Array(repeating: UInt32(0), count: count)

    for step in 0..<horizon {
        let actions = makeCartPoleActions(step: step, count: count)
        let expectedStepped = zip(cpuStates, actions).map { state, action in
            stepCartPoleReference(state: state, action: action, params: cartPoleParams)
        }

        let steppedInterface = try driver.step(actions: actions)
        let gpuStepped = driver.debugReadStates()

        for index in 0..<count {
            try validateCartPoleState(
                actual: gpuStepped[index],
                expected: expectedStepped[index],
                phase: "step",
                step: step,
                index: index,
                tolerance: tolerance
            )
        }
        try validateCartPoleOutputs(
            observations: steppedInterface.observations,
            rewards: steppedInterface.rewards,
            dones: steppedInterface.dones,
            expectedStates: expectedStepped,
            phase: "step-outputs",
            step: step,
            tolerance: tolerance
        )

        var expectedPostReset = expectedStepped
        for index in 0..<count {
            expectedPostReset[index] = resetDoneReference(
                state: expectedStepped[index],
                lane: UInt32(index),
                resetCount: &cpuResetCounts[index],
                params: ResetParams(envCount: UInt32(count), baseSeed: resetSeed)
            )
        }

        let resetInterface = try driver.resetDone()
        let gpuPostReset = driver.debugReadStates()
        let gpuResetCounts = resetInterface.resetCounts

        for index in 0..<count {
            try validateCartPoleState(
                actual: gpuPostReset[index],
                expected: expectedPostReset[index],
                phase: "reset",
                step: step,
                index: index,
                tolerance: tolerance
            )

            if gpuResetCounts[index] != cpuResetCounts[index] {
                throw EnvProjectError.validationFailed(
                    message: "Reset-count validation failed at step \(step), lane \(index): expected \(cpuResetCounts[index]), got \(gpuResetCounts[index])."
                )
            }
        }
        try validateCartPoleOutputs(
            observations: resetInterface.observations,
            rewards: resetInterface.rewards,
            dones: resetInterface.dones,
            expectedStates: expectedPostReset,
            phase: "reset-outputs",
            step: step,
            tolerance: tolerance
        )

        cpuStates = expectedPostReset
    }

    let finalStates = driver.debugReadStates()
    let finalResetCounts = driver.debugReadResetCounts()
    let totalResets = finalResetCounts.reduce(0) { $0 + Int($1) }

    if totalResets == 0 {
        throw EnvProjectError.validationFailed(message: "The rollout never triggered a reset, so reset semantics were not actually exercised.")
    }

    let sampleLines = (0..<4).map { index in
        let state = finalStates[index]
        return String(
            format: "env %d final(x=% .5f xDot=% .5f theta=% .5f thetaDot=% .5f reward=%.1f done=%u resets=%u)",
            index,
            state.x,
            state.xDot,
            state.theta,
            state.thetaDot,
            state.reward,
            state.done,
            finalResetCounts[index]
        )
    }

    return CartPoleRolloutResult(
        finalStates: finalStates,
        resetCounts: finalResetCounts,
        totalResets: totalResets,
        sampleLines: sampleLines
    )
}

func compareReplay(lhs: CartPoleRolloutResult, rhs: CartPoleRolloutResult, tolerance: Float) throws {
    for index in lhs.finalStates.indices {
        let left = lhs.finalStates[index]
        let right = rhs.finalStates[index]

        func checkFloat(_ field: String, _ expected: Float, _ actual: Float) throws {
            if abs(expected - actual) > tolerance {
                throw EnvProjectError.validationFailed(
                    message: "Same-seed replay mismatch at lane \(index), field \(field): expected \(expected), got \(actual)."
                )
            }
        }

        try checkFloat("x", left.x, right.x)
        try checkFloat("xDot", left.xDot, right.xDot)
        try checkFloat("theta", left.theta, right.theta)
        try checkFloat("thetaDot", left.thetaDot, right.thetaDot)
        try checkFloat("reward", left.reward, right.reward)

        if left.done != right.done {
            throw EnvProjectError.validationFailed(
                message: "Same-seed replay mismatch at lane \(index), field done: expected \(left.done), got \(right.done)."
            )
        }

        if lhs.resetCounts[index] != rhs.resetCounts[index] {
            throw EnvProjectError.validationFailed(
                message: "Same-seed replay mismatch at lane \(index), field resetCount: expected \(lhs.resetCounts[index]), got \(rhs.resetCounts[index])."
            )
        }
    }
}

func ensureDifferentSeedChangesResetState(reference: CartPoleRolloutResult, alternate: CartPoleRolloutResult, tolerance: Float) throws {
    for index in reference.finalStates.indices where reference.resetCounts[index] > 0 || alternate.resetCounts[index] > 0 {
        let left = reference.finalStates[index]
        let right = alternate.finalStates[index]

        if abs(left.x - right.x) > tolerance ||
            abs(left.xDot - right.xDot) > tolerance ||
            abs(left.theta - right.theta) > tolerance ||
            abs(left.thetaDot - right.thetaDot) > tolerance ||
            reference.resetCounts[index] != alternate.resetCounts[index] {
            return
        }
    }

    throw EnvProjectError.validationFailed(message: "Different seeded rollout produced no observable change in reset states.")
}

func runValidationHarness() throws {
    guard let device = MTLCreateSystemDefaultDevice() else {
        throw EnvProjectError.noMetalDevice
    }

    try checkLayout(CartPoleState.self, name: "CartPoleState", expectedSize: 24, expectedStride: 24, expectedAlignment: 4)
    try checkLayout(CartPoleParams.self, name: "CartPoleParams", expectedSize: 36, expectedStride: 36, expectedAlignment: 4)
    try checkLayout(ResetParams.self, name: "ResetParams", expectedSize: 8, expectedStride: 8, expectedAlignment: 4)

    let env = ProcessInfo.processInfo.environment
    let rootDir = env["METAL_SMOKE_ROOT"] ?? FileManager.default.currentDirectoryPath
    let count = 256
    let horizon = 24
    let tolerance: Float = 1e-5
    let replayTolerance: Float = 0.0
    let resetSeed: UInt32 = 0x1234_5678
    let actionSeed: UInt32 = 0x89AB_CDEF
    let cartPoleParams = CartPoleParams(
        envCount: UInt32(count),
        dt: 0.02,
        gravity: 9.8,
        massCart: 1.0,
        massPole: 0.1,
        halfPoleLength: 0.5,
        forceMag: 10.0,
        xThreshold: 2.4,
        thetaThresholdRadians: 12.0 * .pi / 180.0
    )
    let initialStates = makeCartPoleInitialStates(count: count)

    let driver = try CartPoleVectorEnvDriver(
        device: device,
        rootDir: rootDir,
        envCount: count,
        cartPoleParams: cartPoleParams,
        resetSeed: resetSeed,
        initialStates: initialStates
    )
    let linearPolicy = try makeReferenceLinearPolicy(
        observationSpec: driver.observationSpec,
        actionSpec: driver.actionSpec
    )
    let alternateLinearPolicy = try makeAlternateLinearPolicy(
        observationSpec: driver.observationSpec,
        actionSpec: driver.actionSpec
    )
    let metalLinearPolicy = try makeReferenceMetalLinearPolicy(
        device: device,
        rootDir: rootDir,
        envCount: count,
        observationSpec: driver.observationSpec,
        actionSpec: driver.actionSpec
    )
    let alternateMetalLinearPolicy = try makeAlternateMetalLinearPolicy(
        device: device,
        rootDir: rootDir,
        envCount: count,
        observationSpec: driver.observationSpec,
        actionSpec: driver.actionSpec
    )

    let baseline = try runValidatedRollout(
        driver: driver,
        cartPoleParams: cartPoleParams,
        resetSeed: resetSeed,
        initialStates: initialStates,
        horizon: horizon,
        tolerance: tolerance
    )

    let replay = try runValidatedRollout(
        driver: driver,
        cartPoleParams: cartPoleParams,
        resetSeed: resetSeed,
        initialStates: initialStates,
        horizon: horizon,
        tolerance: tolerance
    )
    try compareReplay(lhs: baseline, rhs: replay, tolerance: replayTolerance)

    let alternateSeedRun = try runValidatedRollout(
        driver: driver,
        cartPoleParams: cartPoleParams,
        resetSeed: resetSeed &+ 1,
        initialStates: initialStates,
        horizon: horizon,
        tolerance: tolerance
    )
    try ensureDifferentSeedChangesResetState(reference: baseline, alternate: alternateSeedRun, tolerance: tolerance)

    driver.setResetSeed(resetSeed)
    let randomPolicyBaseline = try collectRandomPolicyRollout(
        driver: driver,
        config: RandomPolicyConfig(horizon: 8, actionSeed: actionSeed)
    )

    driver.setResetSeed(resetSeed)
    let randomPolicyReplay = try collectRandomPolicyRollout(
        driver: driver,
        config: RandomPolicyConfig(horizon: 8, actionSeed: actionSeed)
    )
    try compareRolloutActions(lhs: randomPolicyBaseline, rhs: randomPolicyReplay, tolerance: replayTolerance)
    try compareRolloutBatches(lhs: randomPolicyBaseline, rhs: randomPolicyReplay, tolerance: replayTolerance)

    driver.setResetSeed(resetSeed)
    let randomPolicyAlternate = try collectRandomPolicyRollout(
        driver: driver,
        config: RandomPolicyConfig(horizon: 8, actionSeed: actionSeed &+ 1)
    )
    try ensureDifferentActionSeedChangesRollout(reference: randomPolicyBaseline, alternate: randomPolicyAlternate, tolerance: replayTolerance)
    try validateActionBounds(rollout: randomPolicyBaseline, actionSpec: driver.actionSpec, tolerance: tolerance)

    let randomPolicyStorage = try makeRolloutStorage(from: randomPolicyBaseline, driver: driver)
    try validateStorageAgainstRollout(storage: randomPolicyStorage, rollout: randomPolicyBaseline, tolerance: tolerance, context: "random-policy storage")
    let randomPolicyStoredReplay = try collectRandomPolicyRolloutStorage(
        driver: driver,
        config: RandomPolicyConfig(horizon: 8, actionSeed: actionSeed)
    )
    try compareStorage(lhs: randomPolicyStorage, rhs: randomPolicyStoredReplay, tolerance: replayTolerance, context: "random-policy storage replay")

    let policyProbeBatch = try driver.reset()
    let cpuLinearProbeActions = try linearPolicy.actions(
        for: policyProbeBatch.observations,
        envCount: driver.envCount,
        observationSpec: driver.observationSpec,
        actionSpec: driver.actionSpec
    )
    let gpuLinearProbeActions = try metalLinearPolicy.actions(
        for: policyProbeBatch.observations,
        envCount: driver.envCount,
        observationSpec: driver.observationSpec,
        actionSpec: driver.actionSpec
    )
    try compareActionArrays(
        lhs: cpuLinearProbeActions,
        rhs: gpuLinearProbeActions,
        tolerance: tolerance,
        context: "CPU-vs-GPU linear-policy probe"
    )

    driver.setResetSeed(resetSeed)
    let linearPolicyBaseline = try collectPolicyRollout(
        driver: driver,
        policy: linearPolicy,
        config: PolicyRolloutConfig(horizon: 8)
    )

    driver.setResetSeed(resetSeed)
    let linearPolicyReplay = try collectPolicyRollout(
        driver: driver,
        policy: linearPolicy,
        config: PolicyRolloutConfig(horizon: 8)
    )
    try compareRolloutActions(lhs: linearPolicyBaseline, rhs: linearPolicyReplay, tolerance: replayTolerance)
    try compareRolloutBatches(lhs: linearPolicyBaseline, rhs: linearPolicyReplay, tolerance: replayTolerance)
    try validateActionBounds(rollout: linearPolicyBaseline, actionSpec: driver.actionSpec, tolerance: tolerance)

    driver.setResetSeed(resetSeed)
    let linearPolicyAlternate = try collectPolicyRollout(
        driver: driver,
        policy: alternateLinearPolicy,
        config: PolicyRolloutConfig(horizon: 8)
    )
    try ensureDifferentPoliciesChangeRollout(reference: linearPolicyBaseline, alternate: linearPolicyAlternate, tolerance: replayTolerance)

    let linearPolicyStorage = try makeRolloutStorage(from: linearPolicyBaseline, driver: driver)
    try validateStorageAgainstRollout(storage: linearPolicyStorage, rollout: linearPolicyBaseline, tolerance: tolerance, context: "cpu linear-policy storage")
    let linearPolicyStoredReplay = try collectPolicyRolloutStorage(
        driver: driver,
        policy: linearPolicy,
        config: PolicyRolloutConfig(horizon: 8)
    )
    try compareStorage(lhs: linearPolicyStorage, rhs: linearPolicyStoredReplay, tolerance: replayTolerance, context: "cpu linear-policy storage replay")

    driver.setResetSeed(resetSeed)
    let metalLinearPolicyBaseline = try collectPolicyRollout(
        driver: driver,
        policy: metalLinearPolicy,
        config: PolicyRolloutConfig(horizon: 8)
    )

    driver.setResetSeed(resetSeed)
    let metalLinearPolicyReplay = try collectPolicyRollout(
        driver: driver,
        policy: metalLinearPolicy,
        config: PolicyRolloutConfig(horizon: 8)
    )
    try compareRolloutActions(lhs: metalLinearPolicyBaseline, rhs: metalLinearPolicyReplay, tolerance: replayTolerance)
    try compareRolloutBatches(lhs: metalLinearPolicyBaseline, rhs: metalLinearPolicyReplay, tolerance: replayTolerance)
    try validateActionBounds(rollout: metalLinearPolicyBaseline, actionSpec: driver.actionSpec, tolerance: tolerance)
    try compareRolloutActions(lhs: linearPolicyBaseline, rhs: metalLinearPolicyBaseline, tolerance: tolerance)
    try compareRolloutBatches(lhs: linearPolicyBaseline, rhs: metalLinearPolicyBaseline, tolerance: tolerance)

    driver.setResetSeed(resetSeed)
    let metalLinearPolicyAlternate = try collectPolicyRollout(
        driver: driver,
        policy: alternateMetalLinearPolicy,
        config: PolicyRolloutConfig(horizon: 8)
    )
    try ensureDifferentPoliciesChangeRollout(reference: metalLinearPolicyBaseline, alternate: metalLinearPolicyAlternate, tolerance: replayTolerance)

    let metalLinearPolicyStorage = try makeRolloutStorage(from: metalLinearPolicyBaseline, driver: driver)
    try validateStorageAgainstRollout(storage: metalLinearPolicyStorage, rollout: metalLinearPolicyBaseline, tolerance: tolerance, context: "gpu linear-policy storage")
    let metalLinearPolicyStoredReplay = try collectPolicyRolloutStorage(
        driver: driver,
        policy: metalLinearPolicy,
        config: PolicyRolloutConfig(horizon: 8)
    )
    try compareStorage(lhs: metalLinearPolicyStorage, rhs: metalLinearPolicyStoredReplay, tolerance: replayTolerance, context: "gpu linear-policy storage replay")

    print("CartPole validation harness passed")
    print("device: \(device.name)")
    print("environmentModule: CartPoleMetalEnvironment")
    print("vectorDriver: CartPoleVectorEnvDriver")
    print("shaderPath: src/envs/cartpole/Shaders/cartpole_kernels.metal")
    print("actionSpec: dims=\(driver.actionSpec.dimensionsPerEnv) range=[\(driver.actionSpec.minValue), \(driver.actionSpec.maxValue)]")
    print("observationSpec: elementsPerObservation=\(driver.observationSpec.elementsPerEnv)")
    print("count: \(count)")
    print("horizon: \(horizon)")
    print("CartPoleState layout: size=\(MemoryLayout<CartPoleState>.size) stride=\(MemoryLayout<CartPoleState>.stride) alignment=\(MemoryLayout<CartPoleState>.alignment)")
    print("CartPoleParams layout: size=\(MemoryLayout<CartPoleParams>.size) stride=\(MemoryLayout<CartPoleParams>.stride) alignment=\(MemoryLayout<CartPoleParams>.alignment)")
    print("ResetParams layout: size=\(MemoryLayout<ResetParams>.size) stride=\(MemoryLayout<ResetParams>.stride) alignment=\(MemoryLayout<ResetParams>.alignment)")
    print("validationTolerance: \(tolerance)")
    print("sameSeedReplayTolerance: \(replayTolerance)")
    print("resetSeed: 0x\(String(resetSeed, radix: 16, uppercase: true))")
    print("randomPolicyActionSeed: 0x\(String(actionSeed, radix: 16, uppercase: true))")
    print("totalResets: \(baseline.totalResets)")
    print("same-seed replay matched exactly")
    print("different-seed replay changed reset outcomes")
    print("random-policy same-seed rollout matched exactly")
    print("random-policy different-seed actions changed")
    print("linear-policy replay matched exactly")
    print("alternate linear policy changed actions")
    print("cpu and gpu linear-policy outputs matched")
    print("gpu linear-policy replay matched exactly")
    print("alternate gpu linear policy changed actions")
    print("rollout storage replay matched exactly")
    for line in baseline.sampleLines {
        print(line)
    }
}

do {
    try runValidationHarness()
} catch {
    fputs("error: \(error)\n", stderr)
    exit(1)
}
