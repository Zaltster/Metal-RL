import Foundation
import Metal

func humanoidEnvInt(_ name: String, default defaultValue: Int) -> Int {
    if let raw = ProcessInfo.processInfo.environment[name], let value = Int(raw) {
        return value
    }
    return defaultValue
}

func humanoidEnvFloat(_ name: String, default defaultValue: Float) -> Float {
    if let raw = ProcessInfo.processInfo.environment[name], let value = Float(raw) {
        return value
    }
    return defaultValue
}

func humanoidEnvString(_ name: String, default defaultValue: String) -> String {
    ProcessInfo.processInfo.environment[name] ?? defaultValue
}

func makeDemoActions(step: Int, envCount: Int, dofCount: Int) -> [Float] {
    var actions = Array(repeating: Float.zero, count: envCount * dofCount)
    if dofCount == 0 {
        return actions
    }
    let phase = Float(step) * 0.08
    for env in 0..<envCount {
        let envPhase = phase + Float(env) * 0.07
        let base = env * dofCount
        for dof in 0..<dofCount {
            let scale: Float = dof % 3 == 0 ? 0.18 : 0.10
            actions[base + dof] = sin(envPhase + Float(dof) * 0.31) * scale
        }
    }
    return actions
}

func runHumanoidDemo() throws {
    guard let device = MTLCreateSystemDefaultDevice() else {
        throw EnvProjectError.noMetalDevice
    }

    let env = ProcessInfo.processInfo.environment
    let rootDir = env["METAL_SMOKE_ROOT"] ?? FileManager.default.currentDirectoryPath
    let envCount = humanoidEnvInt("HUMANOID_ENV_COUNT", default: 256)
    let steps = humanoidEnvInt("HUMANOID_STEPS", default: 240)
    let replayEnv = humanoidEnvInt("HUMANOID_REPLAY_ENV", default: 0)
    let dt = humanoidEnvFloat("HUMANOID_DT", default: 1.0 / 60.0)
    let specPath = humanoidEnvString(
        "HUMANOID_SPEC_PATH",
        default: URL(fileURLWithPath: rootDir).appending(path: "docs/humanoid_v1_baseline.json").path()
    )
    let replayPath = humanoidEnvString(
        "HUMANOID_REPLAY_PATH",
        default: URL(fileURLWithPath: rootDir).appending(path: "humanoid-demo/.build/humanoid_replay.html").path()
    )

    let humanoid = try HumanoidMetalEnvironment(
        device: device,
        rootDir: rootDir,
        envCount: envCount,
        specURL: URL(fileURLWithPath: specPath),
        dt: dt
    )

    var frames: [HumanoidReplayFrame] = []
    frames.reserveCapacity(steps + 1)
    _ = try humanoid.reset()
    frames.append(try humanoid.makeReplayFrame(envIndex: replayEnv))
    for step in 0..<steps {
        let actions = makeDemoActions(step: step, envCount: envCount, dofCount: humanoid.dofCount)
        _ = try humanoid.step(actions: actions)
        frames.append(try humanoid.makeReplayFrame(envIndex: replayEnv))
    }

    try writeHumanoidHTMLReplay(
        frames: frames,
        linkNames: humanoid.linkNames,
        parentLinkIndices: humanoid.parentLinkIndices,
        to: URL(fileURLWithPath: replayPath),
        title: "Humanoid Elastic Joint Replay"
    )

    print("Humanoid GPU demo completed")
    print("device: \(device.name)")
    print("envCount: \(envCount)")
    print("steps: \(steps)")
    print("dofCount: \(humanoid.dofCount)")
    print("linkCount: \(humanoid.linkCount)")
    print("jointCount: \(humanoid.jointCount)")
    print("observationDim: \(humanoid.observationSpec.elementsPerEnv)")
    print("replayPath: \(replayPath)")
    for warning in humanoid.warnings {
        print("warning: \(warning)")
    }
}

do {
    try runHumanoidDemo()
} catch {
    fputs("error: \(error)\n", stderr)
    exit(1)
}

