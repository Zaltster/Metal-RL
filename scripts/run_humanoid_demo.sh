#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT_DIR="$ROOT_DIR/humanoid-demo"
BUILD_DIR="$PROJECT_DIR/.build"
SDK_PATH="/Library/Developer/CommandLineTools/SDKs/MacOSX15.4.sdk"

mkdir -p "$BUILD_DIR"

swiftc \
  -sdk "$SDK_PATH" \
  -framework Metal \
  "$ROOT_DIR/src/metal_core/MetalSupport.swift" \
  "$ROOT_DIR/src/envs/common/VectorEnv.swift" \
  "$ROOT_DIR/src/envs/humanoid/HumanoidTypes.swift" \
  "$ROOT_DIR/src/envs/humanoid/HumanoidRobotLoader.swift" \
  "$ROOT_DIR/src/envs/humanoid/HumanoidMetalEnvironment.swift" \
  "$ROOT_DIR/src/envs/humanoid/HumanoidStandingVectorEnvDriver.swift" \
  "$ROOT_DIR/src/envs/humanoid/HumanoidHTMLReplay.swift" \
  "$PROJECT_DIR/Sources/main.swift" \
  -o "$BUILD_DIR/humanoid-demo"

METAL_SMOKE_ROOT="$ROOT_DIR" "$BUILD_DIR/humanoid-demo"
