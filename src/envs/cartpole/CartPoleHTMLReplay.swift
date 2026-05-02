import Foundation

struct CartPoleReplayFrame {
    let x: Float
    let xDot: Float
    let theta: Float
    let thetaDot: Float
    let reward: Float
    let done: UInt32
    let resetCount: UInt32
}

func makeCartPoleReplayFrames(
    from rollout: VectorRollout,
    envIndex: Int,
    observationSpec: VectorObservationSpec
) throws -> [CartPoleReplayFrame] {
    if envIndex < 0 {
        throw EnvProjectError.validationFailed(message: "CartPole replay env index must be non-negative.")
    }
    let observationDim = observationSpec.elementsPerEnv
    if observationDim != 4 {
        throw EnvProjectError.validationFailed(message: "CartPole replay renderer expects 4-element observations.")
    }

    let initialEnvCount = rollout.initialBatch.observations.count / observationDim
    if envIndex >= initialEnvCount {
        throw EnvProjectError.validationFailed(
            message: "CartPole replay env index \(envIndex) is out of bounds for envCount \(initialEnvCount)."
        )
    }

    func frame(
        observations: [Float],
        rewards: [Float],
        dones: [UInt32],
        resetCounts: [UInt32],
        envIndex: Int
    ) throws -> CartPoleReplayFrame {
        let envCount = observations.count / observationDim
        if envIndex >= envCount ||
            rewards.count <= envIndex ||
            dones.count <= envIndex ||
            resetCounts.count <= envIndex {
            throw EnvProjectError.validationFailed(message: "CartPole replay batch shape mismatch.")
        }

        let base = envIndex * observationDim
        return CartPoleReplayFrame(
            x: observations[base],
            xDot: observations[base + 1],
            theta: observations[base + 2],
            thetaDot: observations[base + 3],
            reward: rewards[envIndex],
            done: dones[envIndex],
            resetCount: resetCounts[envIndex]
        )
    }

    var frames: [CartPoleReplayFrame] = [
        try frame(
            observations: rollout.initialBatch.observations,
            rewards: rollout.initialBatch.rewards,
            dones: rollout.initialBatch.dones,
            resetCounts: rollout.initialBatch.resetCounts,
            envIndex: envIndex
        ),
    ]
    frames.reserveCapacity(rollout.steps.count + 1)

    for step in rollout.steps {
        frames.append(
            try frame(
                observations: step.observationsAfterReset,
                rewards: step.rewards,
                dones: step.dones,
                resetCounts: step.resetCounts,
                envIndex: envIndex
            )
        )
    }

    return frames
}

func writeCartPoleHTMLReplay(
    frames: [CartPoleReplayFrame],
    to url: URL,
    title: String,
    dt: Float,
    xThreshold: Float,
    halfPoleLength: Float
) throws {
    if frames.isEmpty {
        throw EnvProjectError.validationFailed(message: "CartPole replay renderer requires at least one frame.")
    }

    let frameJSON = frames.map { frame in
        String(
            format: "{\"x\":%.8g,\"xDot\":%.8g,\"theta\":%.8g,\"thetaDot\":%.8g,\"reward\":%.8g,\"done\":%u,\"resetCount\":%u}",
            frame.x,
            frame.xDot,
            frame.theta,
            frame.thetaDot,
            frame.reward,
            frame.done,
            frame.resetCount
        )
    }.joined(separator: ",\n")

    let escapedTitle = title
        .replacingOccurrences(of: "&", with: "&amp;")
        .replacingOccurrences(of: "<", with: "&lt;")
        .replacingOccurrences(of: ">", with: "&gt;")
        .replacingOccurrences(of: "\"", with: "&quot;")

    let html = """
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>\(escapedTitle)</title>
      <style>
        body {
          margin: 0;
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          background: #111;
          color: #eee;
        }
        main {
          max-width: 960px;
          margin: 0 auto;
          padding: 24px;
        }
        canvas {
          display: block;
          width: 100%;
          height: auto;
          background: #f8fafc;
          border: 1px solid #333;
        }
        .stats {
          display: grid;
          grid-template-columns: repeat(4, minmax(0, 1fr));
          gap: 8px;
          margin: 12px 0;
          font-size: 14px;
        }
        .stat {
          background: #1f2937;
          padding: 8px 10px;
          border-radius: 6px;
        }
        button {
          margin-right: 8px;
          padding: 8px 12px;
          border: 0;
          border-radius: 6px;
          background: #2563eb;
          color: white;
          cursor: pointer;
        }
      </style>
    </head>
    <body>
      <main>
        <h1>\(escapedTitle)</h1>
        <canvas id="scene" width="960" height="480"></canvas>
        <div class="stats">
          <div class="stat">frame <span id="frame">0</span></div>
          <div class="stat">x <span id="x">0</span></div>
          <div class="stat">theta <span id="theta">0</span></div>
          <div class="stat">reward <span id="reward">0</span></div>
        </div>
        <button id="play">Pause</button>
        <button id="restart">Restart</button>
      </main>
      <script>
        const frames = [
    \(frameJSON)
        ];
        const dt = \(dt);
        const xThreshold = \(xThreshold);
        const halfPoleLength = \(halfPoleLength);
        const canvas = document.getElementById("scene");
        const ctx = canvas.getContext("2d");
        const frameEl = document.getElementById("frame");
        const xEl = document.getElementById("x");
        const thetaEl = document.getElementById("theta");
        const rewardEl = document.getElementById("reward");
        const playButton = document.getElementById("play");
        const restartButton = document.getElementById("restart");
        let frameIndex = 0;
        let playing = true;
        let lastTime = 0;
        const playbackScale = 2.0;

        function draw(frame) {
          const w = canvas.width;
          const h = canvas.height;
          ctx.clearRect(0, 0, w, h);

          const trackY = h * 0.68;
          const pixelsPerMeter = w * 0.36 / xThreshold;
          const centerX = w / 2;
          const cartX = centerX + frame.x * pixelsPerMeter;
          const cartW = 86;
          const cartH = 38;
          const wheelR = 9;
          const poleLength = Math.max(120, halfPoleLength * 2 * pixelsPerMeter * 0.75);
          const hingeX = cartX;
          const hingeY = trackY - cartH;
          const poleAngle = frame.theta;
          const poleX = hingeX + Math.sin(poleAngle) * poleLength;
          const poleY = hingeY - Math.cos(poleAngle) * poleLength;

          ctx.strokeStyle = "#cbd5e1";
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(48, trackY);
          ctx.lineTo(w - 48, trackY);
          ctx.stroke();

          ctx.strokeStyle = "#94a3b8";
          ctx.lineWidth = 1;
          for (const sign of [-1, 1]) {
            const limitX = centerX + sign * xThreshold * pixelsPerMeter;
            ctx.beginPath();
            ctx.moveTo(limitX, trackY - 72);
            ctx.lineTo(limitX, trackY + 28);
            ctx.stroke();
          }

          ctx.fillStyle = frame.done ? "#dc2626" : "#2563eb";
          ctx.fillRect(cartX - cartW / 2, trackY - cartH, cartW, cartH);
          ctx.fillStyle = "#0f172a";
          ctx.beginPath();
          ctx.arc(cartX - cartW * 0.3, trackY + wheelR, wheelR, 0, Math.PI * 2);
          ctx.arc(cartX + cartW * 0.3, trackY + wheelR, wheelR, 0, Math.PI * 2);
          ctx.fill();

          ctx.strokeStyle = frame.done ? "#ef4444" : "#111827";
          ctx.lineWidth = 8;
          ctx.lineCap = "round";
          ctx.beginPath();
          ctx.moveTo(hingeX, hingeY);
          ctx.lineTo(poleX, poleY);
          ctx.stroke();

          ctx.fillStyle = "#f59e0b";
          ctx.beginPath();
          ctx.arc(hingeX, hingeY, 8, 0, Math.PI * 2);
          ctx.fill();

          ctx.fillStyle = "#111827";
          ctx.font = "16px ui-monospace, SFMono-Regular, Menlo, monospace";
          ctx.fillText(`resetCount=${frame.resetCount} done=${frame.done}`, 24, 32);
          if (frame.done) {
            ctx.fillStyle = "#dc2626";
            ctx.font = "24px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
            ctx.fillText("terminal step / reset", 24, 66);
          }

          frameEl.textContent = `${frameIndex + 1}/${frames.length}`;
          xEl.textContent = frame.x.toFixed(4);
          thetaEl.textContent = frame.theta.toFixed(4);
          rewardEl.textContent = frame.reward.toFixed(1);
        }

        function tick(timestamp) {
          if (!lastTime) lastTime = timestamp;
          const interval = Math.max(16, dt * 1000 / playbackScale);
          if (playing && timestamp - lastTime >= interval) {
            frameIndex = (frameIndex + 1) % frames.length;
            lastTime = timestamp;
          }
          draw(frames[frameIndex]);
          requestAnimationFrame(tick);
        }

        playButton.addEventListener("click", () => {
          playing = !playing;
          playButton.textContent = playing ? "Pause" : "Play";
        });
        restartButton.addEventListener("click", () => {
          frameIndex = 0;
          lastTime = 0;
          draw(frames[frameIndex]);
        });

        draw(frames[0]);
        requestAnimationFrame(tick);
      </script>
    </body>
    </html>
    """

    try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    try html.write(to: url, atomically: true, encoding: .utf8)
}
