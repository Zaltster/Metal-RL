import Foundation

func writeHumanoidHTMLReplay(
    frames: [HumanoidReplayFrame],
    linkNames: [String],
    parentLinkIndices: [Int],
    to url: URL,
    title: String
) throws {
    if frames.isEmpty {
        throw EnvProjectError.validationFailed(message: "Humanoid replay requires at least one frame.")
    }
    if frames[0].linkPositions.count != linkNames.count * 3 {
        throw EnvProjectError.validationFailed(message: "Humanoid replay link position shape mismatch.")
    }

    let escapedTitle = title
        .replacingOccurrences(of: "&", with: "&amp;")
        .replacingOccurrences(of: "<", with: "&lt;")
        .replacingOccurrences(of: ">", with: "&gt;")
        .replacingOccurrences(of: "\"", with: "&quot;")

    let namesJSON = linkNames.map { "\"\($0)\"" }.joined(separator: ",")
    let parentsJSON = parentLinkIndices.map(String.init).joined(separator: ",")
    let frameJSON = frames.map { frame in
        let positions = frame.linkPositions.map { String(format: "%.8g", $0) }.joined(separator: ",")
        return String(
            format: "{\"p\":[%@],\"r\":%.8g,\"d\":%u,\"rc\":%u}",
            positions,
            frame.reward,
            frame.done,
            frame.resetCount
        )
    }.joined(separator: ",\n")

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
        const linkNames = [\(namesJSON)];
        const parent = [\(parentsJSON)];
        const frames = [
    \(frameJSON)
        ];
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

        function point(frame, link) {
          const i = link * 3;
          return { x: frame.p[i], y: frame.p[i + 1], z: frame.p[i + 2] };
        }

        function project(p, view) {
          const scale = 260;
          if (view === "front") return { x: 270 + p.y * scale, y: 560 - p.z * scale };
          return { x: 810 + p.x * scale, y: 560 - p.z * scale };
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
}

