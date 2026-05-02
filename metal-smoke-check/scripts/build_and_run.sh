#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../" && pwd)"
PROJECT_DIR="$ROOT_DIR/metal-smoke-check"
BUILD_DIR="$PROJECT_DIR/.build"
SDK_PATH="/Library/Developer/CommandLineTools/SDKs/MacOSX15.4.sdk"

mkdir -p "$BUILD_DIR"

swiftc \
  -sdk "$SDK_PATH" \
  -framework Metal \
  "$ROOT_DIR/src/metal_core/MetalSupport.swift" \
  "$ROOT_DIR/src/envs/common/VectorEnv.swift" \
  "$ROOT_DIR/src/envs/cartpole/CartPoleTypes.swift" \
  "$ROOT_DIR/src/envs/cartpole/CartPoleReference.swift" \
  "$ROOT_DIR/src/envs/cartpole/CartPoleHTMLReplay.swift" \
  "$ROOT_DIR/src/envs/cartpole/CartPoleMetalEnvironment.swift" \
  "$ROOT_DIR/src/envs/cartpole/CartPoleVectorEnvDriver.swift" \
  "$ROOT_DIR/src/envs/humanoid/HumanoidTypes.swift" \
  "$ROOT_DIR/src/envs/humanoid/HumanoidRobotLoader.swift" \
  "$ROOT_DIR/src/envs/humanoid/HumanoidMetalEnvironment.swift" \
  "$ROOT_DIR/src/envs/humanoid/HumanoidHTMLReplay.swift" \
  "$ROOT_DIR/src/rl/random/RandomPolicyRollout.swift" \
  "$ROOT_DIR/src/rl/policy/LinearPolicy.swift" \
  "$ROOT_DIR/src/rl/policy/MLPPolicy.swift" \
  "$ROOT_DIR/src/rl/policy/MetalLinearPolicy.swift" \
  "$ROOT_DIR/src/rl/policy/MetalMLPPolicy.swift" \
  "$ROOT_DIR/src/rl/policy/PolicyRollout.swift" \
  "$ROOT_DIR/src/rl/losses/PPOLoss.swift" \
  "$ROOT_DIR/src/rl/postprocess/GAE.swift" \
  "$ROOT_DIR/src/rl/storage/RolloutStorage.swift" \
  "$ROOT_DIR/src/rl/train/CPUActorCriticUpdate.swift" \
  "$ROOT_DIR/src/rl/train/CPUTrainingLoop.swift" \
  "$ROOT_DIR/src/rl/train/HybridTrainingLoop.swift" \
  "$ROOT_DIR/src/rl/train/MetalMLPGradients.swift" \
  "$ROOT_DIR/src/rl/train/MetalSGDTrainingStep.swift" \
  "$ROOT_DIR/src/rl/train/MetalTrainableMLPActorCritic.swift" \
  "$ROOT_DIR/src/rl/train/PersistentMetalTrainingLoop.swift" \
  "$ROOT_DIR/src/rl/train/TrainingCheckpoint.swift" \
  "$PROJECT_DIR/Sources/main.swift" \
  -o "$BUILD_DIR/cartpole-validation"

METAL_SMOKE_ROOT="$ROOT_DIR" "$BUILD_DIR/cartpole-validation"
