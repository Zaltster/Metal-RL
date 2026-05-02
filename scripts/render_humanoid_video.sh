#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 <humanoid_replay.json> <output.mp4>" >&2
  exit 64
fi

REPLAY_JSON="$1"
OUTPUT_MP4="$2"
FPS="${HUMANOID_RENDER_FPS:-20}"
WIDTH="${HUMANOID_RENDER_WIDTH:-1080}"
HEIGHT="${HUMANOID_RENDER_HEIGHT:-680}"
MAX_FRAMES="${HUMANOID_RENDER_MAX_FRAMES:-0}"

if [ ! -f "$REPLAY_JSON" ]; then
  echo "error: replay JSON does not exist: $REPLAY_JSON" >&2
  exit 66
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "error: ffmpeg is required to encode humanoid replay video; install ffmpeg and retry." >&2
  exit 69
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "error: python3 is required to inspect humanoid replay JSON." >&2
  exit 69
fi

if [ -n "${HUMANOID_RENDER_BROWSER:-}" ]; then
  BROWSER="$HUMANOID_RENDER_BROWSER"
elif command -v chromium >/dev/null 2>&1; then
  BROWSER="$(command -v chromium)"
elif command -v chromium-browser >/dev/null 2>&1; then
  BROWSER="$(command -v chromium-browser)"
elif command -v google-chrome >/dev/null 2>&1; then
  BROWSER="$(command -v google-chrome)"
elif [ -x "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" ]; then
  BROWSER="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
else
  echo "error: a Chromium-compatible browser is required; install Chromium/Chrome or set HUMANOID_RENDER_BROWSER." >&2
  exit 69
fi

if [ ! -x "$BROWSER" ]; then
  echo "error: browser executable is not runnable: $BROWSER" >&2
  exit 69
fi

FRAME_COUNT="$(
  python3 - "$REPLAY_JSON" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    replay = json.load(f)
frames = replay.get("frames")
if not isinstance(frames, list) or not frames:
    raise SystemExit("0")
print(len(frames))
PY
)"
if [ "$FRAME_COUNT" = "0" ]; then
  echo "error: replay JSON contains no frames: $REPLAY_JSON" >&2
  exit 65
fi
if [ "$MAX_FRAMES" -gt 0 ] && [ "$MAX_FRAMES" -lt "$FRAME_COUNT" ]; then
  FRAME_COUNT="$MAX_FRAMES"
fi

WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/humanoid-video.XXXXXX")"
trap 'rm -rf "$WORKDIR"' EXIT
HTML="$WORKDIR/render.html"

cat > "$HTML" <<'HTML'
<!doctype html>
<meta charset="utf-8">
<canvas id="scene"></canvas>
<script>
const params = new URLSearchParams(location.search);
const frameIndex = Number(params.get("frame") || "0");
const replay = "__REPLAY_JSON__";
const canvas = document.getElementById("scene");
const ctx = canvas.getContext("2d");
canvas.width = Number(params.get("width") || "1080");
canvas.height = Number(params.get("height") || "680");

function point(frame, link) {
  const i = link * 3;
  return { x: frame.p[i], y: frame.p[i + 1], z: frame.p[i + 2] };
}

function computeViewBounds(replay) {
  const bounds = {
    front: { minA: Infinity, maxA: -Infinity, minZ: Infinity, maxZ: -Infinity },
    side: { minA: Infinity, maxA: -Infinity, minZ: Infinity, maxZ: -Infinity }
  };
  for (const frame of replay.frames) {
    for (let i = 0; i < replay.linkNames.length; i++) {
      const p = point(frame, i);
      bounds.front.minA = Math.min(bounds.front.minA, p.y);
      bounds.front.maxA = Math.max(bounds.front.maxA, p.y);
      bounds.side.minA = Math.min(bounds.side.minA, p.x);
      bounds.side.maxA = Math.max(bounds.side.maxA, p.x);
      bounds.front.minZ = Math.min(bounds.front.minZ, p.z);
      bounds.front.maxZ = Math.max(bounds.front.maxZ, p.z);
      bounds.side.minZ = Math.min(bounds.side.minZ, p.z);
      bounds.side.maxZ = Math.max(bounds.side.maxZ, p.z);
    }
  }
  for (const key of ["front", "side"]) {
    const b = bounds[key];
    const padA = Math.max(0.25, (b.maxA - b.minA) * 0.20);
    const padZ = Math.max(0.25, (b.maxZ - b.minZ) * 0.20);
    b.minA -= padA;
    b.maxA += padA;
    b.minZ = Math.min(0, b.minZ - padZ);
    b.maxZ += padZ;
    b.scale = Math.min(440 / Math.max(0.1, b.maxA - b.minA), 500 / Math.max(0.1, b.maxZ - b.minZ));
  }
  return bounds;
}

function project(p, view, bounds) {
  const b = bounds[view];
  const a = view === "front" ? p.y : p.x;
  const left = view === "front" ? 50 : 600;
  const bottom = 610;
  return { x: left + (a - b.minA) * b.scale, y: bottom - (p.z - b.minZ) * b.scale };
}

function shapeRadius(shape, view) {
  if (shape.t === 1) return view === "front" ? Math.max(shape.p[1], shape.p[2]) : Math.max(shape.p[0], shape.p[2]);
  if (shape.t === 2) return shape.p[0];
  if (shape.t === 3 || shape.t === 4) return shape.p[0] + shape.p[1] * 0.35;
  return 0;
}

function drawView(replay, frame, view, label, bounds) {
  ctx.fillStyle = "#0f172a";
  ctx.font = "15px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
  ctx.fillText(label, view === "front" ? 34 : 584, 36);
  ctx.strokeStyle = "#cbd5e1";
  ctx.lineWidth = 2;
  const y = project({x:0,y:0,z:0}, view, bounds).y;
  ctx.beginPath();
  ctx.moveTo(view === "front" ? 30 : 580, y);
  ctx.lineTo(view === "front" ? 520 : 1050, y);
  ctx.stroke();

  ctx.strokeStyle = "#64748b";
  ctx.fillStyle = "rgba(20, 184, 166, 0.10)";
  for (let i = 0; i < replay.linkNames.length; i++) {
    const shape = replay.collisionShapes[i];
    if (!shape || shape.t === 0) continue;
    const p = project(point(frame, i), view, bounds);
    const r = Math.max(3, shapeRadius(shape, view) * bounds[view].scale);
    ctx.beginPath();
    if (shape.t === 1) ctx.rect(p.x - r, p.y - r, r * 2, r * 2);
    else ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }

  for (let i = 0; i < replay.linkNames.length; i++) {
    const pi = replay.parent[i];
    if (pi >= 0) {
      const a = project(point(frame, pi), view, bounds);
      const b = project(point(frame, i), view, bounds);
      ctx.strokeStyle = frame.d ? "#ef4444" : "#334155";
      ctx.lineWidth = Math.max(3, 9 - Math.min(6, i % 7));
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
    }
  }

  for (let i = 0; i < replay.linkNames.length; i++) {
    const p = project(point(frame, i), view, bounds);
    ctx.fillStyle = i === 0 ? "#f59e0b" : "#2563eb";
    ctx.beginPath();
    ctx.arc(p.x, p.y, i === 0 ? 7 : 5, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.strokeStyle = "#dc2626";
  ctx.fillStyle = "#dc2626";
  ctx.lineWidth = 2;
  for (let i = 0; i < replay.linkNames.length; i++) {
    if (frame.pen[i] <= 0) continue;
    const base = i * 3;
    const p3 = { x: frame.cp[base], y: frame.cp[base + 1], z: frame.cp[base + 2] };
    const n3 = { x: frame.cn[base], y: frame.cn[base + 1], z: frame.cn[base + 2] };
    const a = project(p3, view, bounds);
    const b = project({ x: p3.x + n3.x * 0.12, y: p3.y + n3.y * 0.12, z: p3.z + n3.z * 0.12 }, view, bounds);
    ctx.beginPath();
    ctx.arc(a.x, a.y, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }
}

const frame = replay.frames[Math.max(0, Math.min(frameIndex, replay.frames.length - 1))];
const bounds = computeViewBounds(replay);
ctx.fillStyle = "#f8fafc";
ctx.fillRect(0, 0, canvas.width, canvas.height);
drawView(replay, frame, "front", "+Y / +Z front view", bounds);
drawView(replay, frame, "side", "+X / +Z side view", bounds);
ctx.fillStyle = "#0f172a";
ctx.font = "14px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
ctx.fillText(`frame ${frameIndex} reward ${frame.r.toFixed(3)} done ${frame.d} reset ${frame.rc}`, 34, canvas.height - 24);
document.body.dataset.ready = "1";
</script>
HTML

python3 - "$HTML" "$REPLAY_JSON" <<'PY'
from pathlib import Path
import sys
html_path = Path(sys.argv[1])
json_path = Path(sys.argv[2])
html = html_path.read_text(encoding="utf-8")
replay = json_path.read_text(encoding="utf-8")
html_path.write_text(html.replace('"__REPLAY_JSON__"', replay), encoding="utf-8")
PY

for ((i = 0; i < FRAME_COUNT; i++)); do
  printf -v FRAME_PATH "%s/frame_%05d.png" "$WORKDIR" "$i"
  "$BROWSER" \
    --headless=new \
    --disable-gpu \
    --hide-scrollbars \
    --allow-file-access-from-files \
    --no-first-run \
    --user-data-dir="$WORKDIR/chrome-profile" \
    --run-all-compositor-stages-before-draw \
    --window-size="${WIDTH},${HEIGHT}" \
    --screenshot="$FRAME_PATH" \
    "file://$HTML?frame=$i&width=$WIDTH&height=$HEIGHT" >/dev/null 2>&1 &
  BROWSER_PID="$!"
  for _ in $(seq 1 200); do
    if [ -s "$FRAME_PATH" ]; then
      kill "$BROWSER_PID" >/dev/null 2>&1 || true
      wait "$BROWSER_PID" >/dev/null 2>&1 || true
      break
    fi
    if ! kill -0 "$BROWSER_PID" >/dev/null 2>&1; then
      wait "$BROWSER_PID" >/dev/null 2>&1 || true
      break
    fi
    sleep 0.1
  done
  if kill -0 "$BROWSER_PID" >/dev/null 2>&1; then
    kill "$BROWSER_PID" >/dev/null 2>&1 || true
    wait "$BROWSER_PID" >/dev/null 2>&1 || true
  fi
  if [ ! -s "$FRAME_PATH" ]; then
    echo "error: browser did not capture frame $i." >&2
    exit 70
  fi
done

mkdir -p "$(dirname "$OUTPUT_MP4")"
ffmpeg -hide_banner -loglevel error -y \
  -framerate "$FPS" \
  -i "$WORKDIR/frame_%05d.png" \
  -pix_fmt yuv420p \
  -movflags +faststart \
  "$OUTPUT_MP4"

if [ ! -s "$OUTPUT_MP4" ]; then
  echo "error: ffmpeg did not produce output video: $OUTPUT_MP4" >&2
  exit 70
fi

echo "wrote $OUTPUT_MP4"
