import Foundation

func writeHumanoidHTMLReplay(
    frames: [HumanoidReplayFrame],
    linkNames: [String],
    parentLinkIndices: [Int],
    collisionShapes: [HumanoidCollisionGPUConstants],
    to url: URL,
    title: String
) throws {
    if frames.isEmpty {
        throw EnvProjectError.validationFailed(message: "Humanoid replay requires at least one frame.")
    }
    let expectedLinkPositionCount = linkNames.count * 3
    let expectedLinkCount = linkNames.count
    if parentLinkIndices.count != expectedLinkCount {
        throw EnvProjectError.validationFailed(message: "Humanoid replay parent index shape mismatch.")
    }
    if collisionShapes.count != expectedLinkCount {
        throw EnvProjectError.validationFailed(message: "Humanoid replay collision shape count mismatch.")
    }
    for (index, frame) in frames.enumerated() {
        if frame.linkPositions.count != expectedLinkPositionCount {
            throw EnvProjectError.validationFailed(
                message: "Humanoid replay frame \(index) link position shape mismatch."
            )
        }
        if frame.contactPoints.count != expectedLinkPositionCount ||
            frame.contactNormals.count != expectedLinkPositionCount ||
            frame.contactPenetrations.count != expectedLinkCount {
            throw EnvProjectError.validationFailed(
                message: "Humanoid replay frame \(index) contact shape mismatch."
            )
        }
        if !frame.linkPositions.allSatisfy({ $0.isFinite }) ||
            !frame.contactPoints.allSatisfy({ $0.isFinite }) ||
            !frame.contactNormals.allSatisfy({ $0.isFinite }) ||
            !frame.contactPenetrations.allSatisfy({ $0.isFinite }) {
            throw EnvProjectError.validationFailed(
                message: "Humanoid replay frame \(index) contains non-finite replay data."
            )
        }
    }

    let escapedTitle = title
        .replacingOccurrences(of: "&", with: "&amp;")
        .replacingOccurrences(of: "<", with: "&lt;")
        .replacingOccurrences(of: ">", with: "&gt;")
        .replacingOccurrences(of: "\"", with: "&quot;")

    let namesJSON = linkNames.map { "\"\($0)\"" }.joined(separator: ",")
    let parentsJSON = parentLinkIndices.map(String.init).joined(separator: ",")
    let shapeJSON = collisionShapes.map { shape in
        String(
            format: "{\"t\":%u,\"tr\":[%.8g,%.8g,%.8g],\"q\":[%.8g,%.8g,%.8g,%.8g],\"p\":[%.8g,%.8g,%.8g]}",
            shape.type,
            shape.translationX,
            shape.translationY,
            shape.translationZ,
            shape.rotationX,
            shape.rotationY,
            shape.rotationZ,
            shape.rotationW,
            shape.paramX,
            shape.paramY,
            shape.paramZ
        )
    }.joined(separator: ",")
    let frameJSON = frames.map { frame in
        let positions = frame.linkPositions.map { String(format: "%.8g", $0) }.joined(separator: ",")
        let contactPoints = frame.contactPoints.map { String(format: "%.8g", $0) }.joined(separator: ",")
        let contactNormals = frame.contactNormals.map { String(format: "%.8g", $0) }.joined(separator: ",")
        let contactPenetrations = frame.contactPenetrations.map { String(format: "%.8g", $0) }.joined(separator: ",")
        return String(
            format: "{\"p\":[%@],\"cp\":[%@],\"cn\":[%@],\"pen\":[%@],\"r\":%.8g,\"d\":%u,\"rc\":%u}",
            positions,
            contactPoints,
            contactNormals,
            contactPenetrations,
            frame.reward,
            frame.done,
            frame.resetCount
        )
    }.joined(separator: ",\n")
    let replayJSON = """
    {"title":"\(escapedTitle)","linkNames":[\(namesJSON)],"parent":[\(parentsJSON)],"collisionShapes":[\(shapeJSON)],"frames":[
    \(frameJSON)
    ]}
    """

    let html = """
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>\(escapedTitle)</title>
      <style>
        body { margin: 0; background: #101418; color: #e5e7eb; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
        main { max-width: 1120px; margin: 0 auto; padding: 20px; }
        canvas { width: 100%; height: auto; display: block; background: #f8fafc; border: 1px solid #334155; }
        .bar { display: flex; gap: 8px; align-items: center; margin: 12px 0; flex-wrap: wrap; }
        .stat { background: #1f2937; padding: 7px 9px; border-radius: 6px; font-size: 13px; }
        button { padding: 8px 12px; border: 0; border-radius: 6px; background: #2563eb; color: white; cursor: pointer; }
      </style>
    </head>
    <body>
      <main>
        <h1>\(escapedTitle)</h1>
        <canvas id="scene" width="1080" height="680"></canvas>
        <div class="bar">
          <button id="play">Pause</button>
          <button id="restart">Restart</button>
          <div class="stat">frame <span id="frame">0</span></div>
          <div class="stat">reward <span id="reward">0</span></div>
          <div class="stat">done <span id="done">0</span></div>
          <div class="stat">reset <span id="reset">0</span></div>
        </div>
      </main>
      <script>
        const replay = \(replayJSON);
        const linkNames = replay.linkNames;
        const parent = replay.parent;
        const collisionShapes = replay.collisionShapes;
        const frames = replay.frames;
        const canvas = document.getElementById("scene");
        const ctx = canvas.getContext("2d");
        const frameEl = document.getElementById("frame");
        const rewardEl = document.getElementById("reward");
        const doneEl = document.getElementById("done");
        const resetEl = document.getElementById("reset");
        const playButton = document.getElementById("play");
        const restartButton = document.getElementById("restart");
        let frameIndex = 0;
        let playing = true;
        let last = 0;
        const viewBounds = computeViewBounds();

        function point(frame, link) {
          const i = link * 3;
          return { x: frame.p[i], y: frame.p[i + 1], z: frame.p[i + 2] };
        }

        function computeViewBounds() {
          const bounds = {
            front: { minA: Infinity, maxA: -Infinity, minZ: Infinity, maxZ: -Infinity },
            side: { minA: Infinity, maxA: -Infinity, minZ: Infinity, maxZ: -Infinity }
          };
          for (const frame of frames) {
            for (let i = 0; i < linkNames.length; i++) {
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
            if (!Number.isFinite(b.minA) || !Number.isFinite(b.maxA) || !Number.isFinite(b.minZ) || !Number.isFinite(b.maxZ)) {
              b.minA = -1; b.maxA = 1; b.minZ = 0; b.maxZ = 2;
            }
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

        function project(p, view) {
          const b = viewBounds[view];
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

        function drawCollisionShapes(frame, view) {
          ctx.save();
          ctx.strokeStyle = "#64748b";
          ctx.fillStyle = "rgba(20, 184, 166, 0.10)";
          ctx.lineWidth = 1.5;
          for (let i = 0; i < linkNames.length; i++) {
            const shape = collisionShapes[i];
            if (!shape || shape.t === 0) continue;
            const center = point(frame, i);
            const p = project(center, view);
            const r = Math.max(3, shapeRadius(shape, view) * viewBounds[view].scale);
            ctx.beginPath();
            if (shape.t === 1) {
              ctx.rect(p.x - r, p.y - r, r * 2, r * 2);
            } else {
              ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
            }
            ctx.fill();
            ctx.stroke();
          }
          ctx.restore();
        }

        function drawContacts(frame, view) {
          ctx.save();
          ctx.strokeStyle = "#dc2626";
          ctx.fillStyle = "#dc2626";
          ctx.lineWidth = 2;
          for (let i = 0; i < linkNames.length; i++) {
            if (frame.pen[i] <= 0) continue;
            const base = i * 3;
            const point3 = { x: frame.cp[base], y: frame.cp[base + 1], z: frame.cp[base + 2] };
            const normal3 = { x: frame.cn[base], y: frame.cn[base + 1], z: frame.cn[base + 2] };
            const a = project(point3, view);
            const b = project({
              x: point3.x + normal3.x * 0.12,
              y: point3.y + normal3.y * 0.12,
              z: point3.z + normal3.z * 0.12
            }, view);
            ctx.beginPath();
            ctx.arc(a.x, a.y, 4, 0, Math.PI * 2);
            ctx.fill();
            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.stroke();
          }
          ctx.restore();
        }

        function drawView(frame, view, label) {
          ctx.fillStyle = "#0f172a";
          ctx.font = "15px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
          ctx.fillText(label, view === "front" ? 34 : 584, 36);
          ctx.strokeStyle = "#cbd5e1";
          ctx.lineWidth = 2;
          ctx.beginPath();
          const y = project({x:0,y:0,z:0}, view).y;
          ctx.moveTo(view === "front" ? 30 : 580, y);
          ctx.lineTo(view === "front" ? 520 : 1050, y);
          ctx.stroke();

          drawCollisionShapes(frame, view);

          for (let i = 0; i < linkNames.length; i++) {
            const pi = parent[i];
            if (pi >= 0) {
              const a = project(point(frame, pi), view);
              const b = project(point(frame, i), view);
              ctx.strokeStyle = frame.d ? "#ef4444" : "#334155";
              ctx.lineWidth = Math.max(3, 9 - Math.min(6, i % 7));
              ctx.beginPath();
              ctx.moveTo(a.x, a.y);
              ctx.lineTo(b.x, b.y);
              ctx.stroke();
            }
          }

          for (let i = 0; i < linkNames.length; i++) {
            const p = project(point(frame, i), view);
            ctx.fillStyle = i === 0 ? "#f59e0b" : "#2563eb";
            ctx.beginPath();
            ctx.arc(p.x, p.y, i === 0 ? 7 : 5, 0, Math.PI * 2);
            ctx.fill();
          }
          drawContacts(frame, view);
        }

        function draw() {
          const frame = frames[frameIndex];
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          ctx.fillStyle = "#f8fafc";
          ctx.fillRect(0, 0, canvas.width, canvas.height);
          drawView(frame, "front", "+Y / +Z front view");
          drawView(frame, "side", "+X / +Z side view");
          frameEl.textContent = frameIndex.toString();
          rewardEl.textContent = frame.r.toFixed(3);
          doneEl.textContent = frame.d.toString();
          resetEl.textContent = frame.rc.toString();
        }

        function tick(t) {
          if (playing && t - last > 50) {
            frameIndex = (frameIndex + 1) % frames.length;
            last = t;
            draw();
          }
          requestAnimationFrame(tick);
        }
        playButton.onclick = () => {
          playing = !playing;
          playButton.textContent = playing ? "Pause" : "Play";
        };
        restartButton.onclick = () => {
          frameIndex = 0;
          draw();
        };
        draw();
        requestAnimationFrame(tick);
      </script>
    </body>
    </html>
    """

    try FileManager.default.createDirectory(
        at: url.deletingLastPathComponent(),
        withIntermediateDirectories: true
    )
    try html.write(to: url, atomically: true, encoding: .utf8)
    let jsonURL = url.deletingPathExtension().appendingPathExtension("json")
    try replayJSON.write(to: jsonURL, atomically: true, encoding: .utf8)
}
