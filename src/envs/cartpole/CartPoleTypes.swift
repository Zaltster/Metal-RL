import Foundation

struct CartPoleState {
    var x: Float
    var xDot: Float
    var theta: Float
    var thetaDot: Float
    var reward: Float
    var done: UInt32
}

struct CartPoleParams {
    var envCount: UInt32
    var dt: Float
    var gravity: Float
    var massCart: Float
    var massPole: Float
    var halfPoleLength: Float
    var forceMag: Float
    var xThreshold: Float
    var thetaThresholdRadians: Float
}

struct ResetParams {
    var envCount: UInt32
    var baseSeed: UInt32
}

struct CartPoleRolloutResult {
    var finalStates: [CartPoleState]
    var resetCounts: [UInt32]
    var totalResets: Int
    var sampleLines: [String]
}
