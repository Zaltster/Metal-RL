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
  "$ROOT_DIR/src/envs/cartpole/CartPoleMetalEnvironment.swift" \
  "$ROOT_DIR/src/envs/cartpole/CartPoleVectorEnvDriver.swift" \
  "$ROOT_DIR/src/rl/random/RandomPolicyRollout.swift" \
  "$ROOT_DIR/src/rl/policy/LinearPolicy.swift" \
  "$ROOT_DIR/src/rl/policy/MetalLinearPolicy.swift" \
  "$ROOT_DIR/src/rl/policy/PolicyRollout.swift" \
  "$ROOT_DIR/src/rl/storage/RolloutStorage.swift" \
  "$PROJECT_DIR/Sources/main.swift" \
  -o "$BUILD_DIR/cartpole-validation"

METAL_SMOKE_ROOT="$ROOT_DIR" "$BUILD_DIR/cartpole-validation"
