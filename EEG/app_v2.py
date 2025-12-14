# app.py
# EEG Pong Dashboard:
# - Blink detector (BrainFlow, Cyton CH1+CH2) runs in a background thread.
# - Frontend in browser (localhost:5000) shows:
#     * Controls to start/stop detector
#     * Settings: threshold (µV) and cooldown (s)
#     * Last blink event
#     * Pong game (canvas) that toggles paddle direction on blinks
#     * Live signal plot of blink channel

import time
import threading
from collections import deque
import numpy as np

from flask import Flask, jsonify, request, render_template_string

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from scipy.signal import butter, filtfilt, iirnotch

# ------------- CONFIG -------------
SERIAL_PORT = "COM3"                # change if needed (e.g., "/dev/ttyUSB0", "/dev/tty.usbserial-XXXX")
BOARD_ID = BoardIds.CYTON_BOARD.value

CYTON_BLINK_CH1 = 1                 # forehead electrode on Cyton CH1
CYTON_BLINK_CH2 = 2                 # forehead electrode on Cyton CH2

NOTCH_HZ   = 50                     # 50Hz in CH/EU, 60 in US, or None
BANDPASS   = (1.0, 15.0)            # Hz, emphasize blink deflections

BASELINE_WIN_SECS = 2             # seconds for baseline window
CHUNK_SECS        = 0.05            # processing step (50 ms)

# Settings that we can tweak from UI:
DEFAULT_THRESHOLD_UV   = 150.0      # µV above baseline
DEFAULT_COOLDOWN_SECS  = 0.35        # min seconds between blink events

SIGNAL_BUFFER_SECS = 10.0            # seconds of signal to keep for plotting
MAX_SIGNAL_POINTS  = 1000           # max samples to send to frontend
# ----------------------------------

app = Flask(__name__)

# Shared state
state_lock = threading.Lock()
detector_thread = None
detector_running = False

threshold_uv = DEFAULT_THRESHOLD_UV
cooldown_secs = DEFAULT_COOLDOWN_SECS

last_blink_ts = 0.0
last_event_text = "No events yet."
blink_pending = False              # one-shot flag consumed by the frontend
signal_buffer = deque(maxlen=int(250 * SIGNAL_BUFFER_SECS))  # will resize when fs known
fs_global = None

def design_bandpass(fs, low, high, order=4):
    ny = 0.5 * fs
    from scipy.signal import butter
    b, a = butter(order, [low/ny, high/ny], btype="band")
    return b, a

def design_notch(fs, f0, q=30.0):
    ny = 0.5 * fs
    from scipy.signal import iirnotch
    w0 = f0 / ny
    return iirnotch(w0, q)

def apply_filters(x, fs, bp=None, nf=None):
    y = x
    if nf is not None:
        b, a = nf
        y = filtfilt(b, a, y, axis=-1, padlen=50)
    if bp is not None:
        b, a = bp
        y = filtfilt(b, a, y, axis=-1, padlen=50)
    return y

def eeg_detector_loop():
    global detector_running, fs_global, last_blink_ts, blink_pending, last_event_text, signal_buffer

    BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()
    params.serial_port = SERIAL_PORT
    board = BoardShim(BOARD_ID, params)

    try:
        board.prepare_session()
        board.start_stream(45000)

        fs = BoardShim.get_sampling_rate(BOARD_ID)
        fs_global = fs

        eeg_rows = BoardShim.get_eeg_channels(BOARD_ID)
        n_eeg = len(eeg_rows)
        idx_a = CYTON_BLINK_CH1 - 1
        idx_b = CYTON_BLINK_CH2 - 1
        if not (0 <= idx_a < n_eeg and 0 <= idx_b < n_eeg):
            raise ValueError("Blink channels must be between 1 and n_eeg")

        bp = design_bandpass(fs, *BANDPASS) if BANDPASS else None
        nf = design_notch(fs, NOTCH_HZ) if NOTCH_HZ else None

        baseline_len = int(BASELINE_WIN_SECS * fs)
        baseline_buf = deque(maxlen=baseline_len)

        # adjust signal buffer length to fs
        with state_lock:
            signal_buffer = deque(maxlen=int(SIGNAL_BUFFER_SECS * fs))

        # warm-up baseline
        warmup_end = time.time() + BASELINE_WIN_SECS * 2
        while detector_running and time.time() < warmup_end:
            chunk = board.get_current_board_data(int(fs * CHUNK_SECS))
            if not chunk.size:
                time.sleep(0.01)
                continue
            s = chunk[eeg_rows, :]
            bx = 0.5 * (s[idx_a, :] + s[idx_b, :])
            bx = apply_filters(bx, fs, bp, nf) if bx.size > 30 else bx
            abs_bx = np.abs(bx)
            for v in abs_bx:
                baseline_buf.append(float(v))
            # also store signal for plotting (raw combined)
            with state_lock:
                for v in bx:
                    signal_buffer.append(float(v))
            time.sleep(0.01)

        # main loop
        while detector_running:
            chunk = board.get_current_board_data(int(fs * CHUNK_SECS))
            if not chunk.size:
                time.sleep(0.01)
                continue

            s = chunk[eeg_rows, :]
            bx = 0.5 * (s[idx_a, :] + s[idx_b, :])
            bx = apply_filters(bx, fs, bp, nf) if bx.size > 30 else bx
            abs_bx = np.abs(bx)

            # update baseline + signal buffer
            with state_lock:
                th = threshold_uv
                cd = cooldown_secs

            for v_raw, v_abs in zip(bx, abs_bx):
                baseline_buf.append(float(v_abs))
                with state_lock:
                    signal_buffer.append(float(v_raw))

                if len(baseline_buf) < baseline_len:
                    continue

                base_mean = float(np.mean(baseline_buf))
                diff = float(v_abs - base_mean)
                now = time.time()

                # strict time-based cooldown: never more frequent than cooldown_secs
                if diff > th and v_abs > th and (now - last_blink_ts) >= cd:
                    with state_lock:
                        last_blink_ts = now
                        blink_pending = True
                        last_event_text = (
                            f"BLINK at {time.strftime('%H:%M:%S', time.localtime(now))} | "
                            f"val={v_abs:.1f} µV, base={base_mean:.1f} µV, diff={diff:.1f} µV, "
                            f"thr={th:.1f} µV, cooldown={cd:.2f} s"
                        )

            time.sleep(0.005)

    except Exception as e:
        with state_lock:
            last_event_text = f"Detector error: {e}"
    finally:
        try:
            board.stop_stream()
        except Exception:
            pass
        try:
            board.release_session()
        except Exception:
            pass
        with state_lock:
            detector_running = False
        print("Detector thread stopped.")

def start_detector():
    global detector_thread, detector_running
    with state_lock:
        if detector_running:
            return False
        detector_running = True
        detector_thread = threading.Thread(target=eeg_detector_loop, daemon=True)
        detector_thread.start()
        return True

def stop_detector():
    global detector_running
    with state_lock:
        detector_running = False
    return True

# ------------- FRONTEND HTML -------------
INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>EEG Pong Dashboard</title>
  <style>
    body {
      font-family: system-ui, sans-serif;
      background: #0d0d0d;
      color: #f5f5f5;
      margin: 0;
      padding: 1.5rem;
    }
    h1 { margin-top: 0; }
    .grid {
      display: grid;
      grid-template-columns: 1.5fr 1fr;
      grid-gap: 1rem;
    }
    .card {
      background: #1b1b1b;
      border-radius: 12px;
      padding: 1rem 1.25rem;
      box-shadow: 0 0 0 1px #333;
      margin-bottom: 1rem;
    }
    button {
      padding: 0.5rem 1.2rem;
      margin: 0.3rem 0.3rem 0.3rem 0;
      border-radius: 6px;
      border: none;
      font-size: 0.95rem;
      cursor: pointer;
    }
    button.start { background: #2e7d32; color: #fff; }
    button.stop { background: #c62828; color: #fff; }
    button:disabled { opacity: 0.4; cursor: default; }
    label { display:block; margin-top:0.4rem; font-size:0.9rem; }
    input[type="number"] {
      padding: 0.2rem 0.4rem;
      border-radius: 4px;
      border: 1px solid #555;
      background: #111;
      color: #eee;
      width: 6rem;
    }
    .status-dot {
      display:inline-block;
      width:10px; height:10px;
      border-radius:50%;
      margin-right:0.35rem;
    }
    .dot-on { background:#4caf50; }
    .dot-off { background:#f44336; }
    #last-event {
      font-family: monospace;
      font-size: 0.85rem;
      white-space: pre-wrap;
      margin-top:0.4rem;
      color:#ddd;
    }
    #signalCanvas, #pongCanvas {
      background:#000;
      border-radius:8px;
      border:1px solid #333;
      width:100%;
      max-width:100%;
    }
    #signalCanvas { height:160px; }
    #pongCanvas { height:320px; margin-top:0.5rem; }
  </style>
</head>
<body>
  <h1>🧠 EEG Pong Dashboard</h1>
  <div class="grid">
    <div>
      <div class="card">
        <h2>Detector & Settings</h2>
        <p>
          Use your OpenBCI Cyton (CH1 & CH2 on forehead). This detector runs in the background
          and sends blink events to the Pong game in this page.
        </p>
        <p>
          <span class="status-dot" id="detector-dot"></span>
          <strong>Detector:</strong> <span id="detector-status">stopped</span>
        </p>
        <button class="start" id="btn-start-detector">Start Calibration + Detector</button>
        <button class="stop" id="btn-stop-detector">Stop Detector</button>

        <h3>Settings</h3>
        <label>
          Threshold (µV above baseline):
          <input type="number" id="input-threshold" step="10">
        </label>
        <label>
          Cooldown between blinks (s):
          <input type="number" id="input-cooldown" step="0.05">
        </label>
        <button class="start" id="btn-apply-settings">Apply Settings</button>

        <h3>Last event</h3>
        <div id="last-event">No events yet.</div>
      </div>

      <div class="card">
        <h2>Signal View (forehead combined)</h2>
        <canvas id="signalCanvas"></canvas>
      </div>
    </div>

    <div>
      <div class="card">
        <h2>Pong (Blink-controlled)</h2>
        <p>Blink to toggle paddle direction (up/down). Game runs entirely in the browser.</p>
        <canvas id="pongCanvas" width="400" height="240"></canvas>
      </div>
    </div>
  </div>

<script>
async function api(path, method="GET", body=null) {
  const opts = { method, headers: {} };
  if (body !== null) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(path, opts);
  return await res.json();
}

// --- UI wiring ---
const detectorDot = document.getElementById("detector-dot");
const detectorStatus = document.getElementById("detector-status");
const lastEventDiv = document.getElementById("last-event");
const inputThreshold = document.getElementById("input-threshold");
const inputCooldown = document.getElementById("input-cooldown");

document.getElementById("btn-start-detector").onclick = async () => {
  await api("/api/start_detector", "POST");
  await refreshStatus();
};
document.getElementById("btn-stop-detector").onclick = async () => {
  await api("/api/stop_detector", "POST");
  await refreshStatus();
};
document.getElementById("btn-apply-settings").onclick = async () => {
  const thr = parseFloat(inputThreshold.value);
  const cd  = parseFloat(inputCooldown.value);
  await api("/api/settings", "POST", { threshold_uv: thr, cooldown_secs: cd });
  await refreshStatus();
};

async function refreshStatus() {
  try {
    const status = await api("/api/status", "GET");
    const detOn = status.detector_running;
    detectorDot.className = "status-dot " + (detOn ? "dot-on" : "dot-off");
    detectorStatus.textContent = detOn ? "running" : "stopped";
    lastEventDiv.textContent = status.last_event || "No events.";
  } catch (e) {
    console.error(e);
  }
}

// fetch settings once on load
async function loadSettings() {
  try {
    const s = await api("/api/settings", "GET");
    inputThreshold.value = s.threshold_uv.toFixed(1);
    inputCooldown.value  = s.cooldown_secs.toFixed(2);
  } catch (e) {
    console.error(e);
  }
}

// --- Event polling for game ---
let blinkFlag = false;
async function pollEvents() {
  try {
    const ev = await api("/api/events", "GET");
    if (ev.blink) {
      blinkFlag = true;
      if (ev.last_event) {
        lastEventDiv.textContent = ev.last_event;
      }
    }
  } catch (e) {
    console.error(e);
  }
}

// --- Signal fetch + plot ---
const sigCanvas = document.getElementById("signalCanvas");
const sigCtx = sigCanvas.getContext("2d");

function drawSignal(samples) {
  const w = sigCanvas.width;
  const h = sigCanvas.height;
  sigCtx.clearRect(0,0,w,h);
  sigCtx.fillStyle = "#000";
  sigCtx.fillRect(0,0,w,h);
  if (!samples || samples.length < 2) return;
  let min = Math.min(...samples);
  let max = Math.max(...samples);
  if (min === max) { min -= 1; max += 1; }
  sigCtx.strokeStyle = "#4caf50";
  sigCtx.lineWidth = 1.2;
  sigCtx.beginPath();
  for (let i=0; i<samples.length; i++) {
    const x = (i / (samples.length-1)) * w;
    const norm = (samples[i] - min) / (max - min);
    const y = h - norm * h;
    if (i===0) sigCtx.moveTo(x,y);
    else sigCtx.lineTo(x,y);
  }
  sigCtx.stroke();
}

async function pollSignal() {
  try {
    const s = await api("/api/signal", "GET");
    if (s.samples) {
      drawSignal(s.samples);
    }
  } catch (e) {
    console.error(e);
  }
}

// --- Pong game in browser ---
const pongCanvas = document.getElementById("pongCanvas");
const pongCtx = pongCanvas.getContext("2d");
let lastTime = null;

// Game objects
const W = pongCanvas.width;
const H = pongCanvas.height;
const PADDLE_W = 8;
const PADDLE_H = 50;
const BALL_SIZE = 8;
let paddleY = H/2 - PADDLE_H/2;
let paddleDir = 0;       // -1 up, +1 down, 0 stopped
const paddleSpeed = 80;  // px/s
let ballX = W/2, ballY = H/2;
let ballVx = 60;         // slower ball
let ballVy = 45;

function onBlink() {
  if (paddleDir === 0) paddleDir = -1;
  else paddleDir *= -1;
}

function stepGame(dt) {
  // apply blink event
  if (blinkFlag) {
    onBlink();
    blinkFlag = false;
  }

  // update paddle
  paddleY += paddleDir * paddleSpeed * dt;
  if (paddleY < 0) { paddleY = 0; if (paddleDir < 0) paddleDir = 1; }
  if (paddleY > H - PADDLE_H) { paddleY = H - PADDLE_H; if (paddleDir > 0) paddleDir = -1; }

  // update ball
  ballX += ballVx * dt;
  ballY += ballVy * dt;

  // walls
  if (ballY <= 0) { ballY = 0; ballVy = Math.abs(ballVy); }
  if (ballY >= H - BALL_SIZE) { ballY = H - BALL_SIZE; ballVy = -Math.abs(ballVy); }

  // paddle collision
  if (ballX <= 20 + PADDLE_W && ballX >= 20 &&
      ballY + BALL_SIZE >= paddleY && ballY <= paddleY + PADDLE_H) {
    ballX = 20 + PADDLE_W;
    ballVx = Math.abs(ballVx);
  }

  // right wall
  if (ballX >= W - BALL_SIZE) {
    ballX = W - BALL_SIZE;
    ballVx = -Math.abs(ballVx);
  }

  // missed ball
  if (ballX < 0) {
    ballX = W/2; ballY = H/2;
    ballVx = 60; ballVy = 45;
  }

  // draw
  pongCtx.fillStyle = "#000";
  pongCtx.fillRect(0,0,W,H);
  pongCtx.fillStyle = "#fff";
  pongCtx.fillRect(20, paddleY, PADDLE_W, PADDLE_H);
  pongCtx.fillRect(ballX, ballY, BALL_SIZE, BALL_SIZE);
}

function gameLoop(timestamp) {
  if (!lastTime) lastTime = timestamp;
  const dt = (timestamp - lastTime) / 1000.0;
  lastTime = timestamp;
  stepGame(dt);
  requestAnimationFrame(gameLoop);
}

// initial setup
function resizeCanvases() {
  sigCanvas.width = sigCanvas.clientWidth;
  sigCanvas.height = sigCanvas.clientHeight;
  pongCanvas.width = pongCanvas.clientWidth;
  pongCanvas.height = pongCanvas.clientHeight;
}
window.addEventListener("resize", resizeCanvases);
resizeCanvases();

loadSettings();
refreshStatus();
requestAnimationFrame(gameLoop);
setInterval(pollEvents, 200);  // check blink events 5x/sec
setInterval(pollSignal, 200);  // update signal 5x/sec
setInterval(refreshStatus, 2000);
</script>
</body>
</html>
"""

# ------------- API ROUTES -------------

@app.route("/")
def index():
    return render_template_string(INDEX_HTML)

@app.route("/api/status", methods=["GET"])
def api_status():
    with state_lock:
        return jsonify({
            "detector_running": detector_running,
            "last_event": last_event_text,
            "threshold_uv": threshold_uv,
            "cooldown_secs": cooldown_secs
        })

@app.route("/api/start_detector", methods=["POST"])
def api_start_detector():
    started = start_detector()
    return jsonify({"ok": True, "started": started})

@app.route("/api/stop_detector", methods=["POST"])
def api_stop_detector():
    stop_detector()
    return jsonify({"ok": True})

@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    global threshold_uv, cooldown_secs
    if request.method == "POST":
        data = request.get_json(force=True)
        with state_lock:
            if "threshold_uv" in data:
                threshold_uv = float(data["threshold_uv"])
            if "cooldown_secs" in data:
                cooldown_secs = float(data["cooldown_secs"])
        return jsonify({"ok": True})
    else:
        with state_lock:
            return jsonify({
                "threshold_uv": threshold_uv,
                "cooldown_secs": cooldown_secs
            })

@app.route("/api/events", methods=["GET"])
def api_events():
    global blink_pending
    with state_lock:
        blink = blink_pending
        blink_pending = False
        le = last_event_text
    return jsonify({"blink": blink, "last_event": le})

@app.route("/api/signal", methods=["GET"])
def api_signal():
    with state_lock:
        samples = list(signal_buffer)
    # downsample if needed
    if len(samples) > MAX_SIGNAL_POINTS:
        step = len(samples) // MAX_SIGNAL_POINTS
        samples = samples[::step]
    return jsonify({"samples": samples})

if __name__ == "__main__":
    # start Flask
    app.run(host="127.0.0.1", port=5000, debug=True)
