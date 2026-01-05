import time
import threading
from collections import deque
import numpy as np
from flask import Flask, jsonify, request, render_template_string
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from scipy.signal import butter, filtfilt, iirnotch

# ------------- CONFIG -------------
SERIAL_PORT = "COM3"                # UPDATE THIS TO YOUR CURRENT PC PORT
BOARD_ID = BoardIds.CYTON_BOARD.value
CYTON_BLINK_CH1 = 1                 
CYTON_BLINK_CH2 = 2                 
NOTCH_HZ   = 50                     # 50Hz (EU) or 60Hz (US)
SIGNAL_BUFFER_SECS = 10.0            
MAX_SIGNAL_POINTS  = 1000           

# Initial Settings
DEFAULT_THRESHOLD_UV   = 150.0      
DEFAULT_COOLDOWN_SECS  = 0.2        
BLINK_OFF_FACTOR = 0.3              

app = Flask(__name__)

# Shared state
state_lock = threading.Lock()
detector_thread = None
detector_running = False
threshold_uv = DEFAULT_THRESHOLD_UV
cooldown_secs = DEFAULT_COOLDOWN_SECS
last_blink_ts = 0.0
last_event_text = "No events yet."
blink_pending = False              
signal_buffer = deque(maxlen=2500) 
calibration_readings = {"noise": [], "blinks": []}
fs_global = 250.0

# --- DSP & DETECTOR LOGIC ---
def apply_filters(data, fs, notch_freq=50):
    if len(data) < 100: return data
    nyq = 0.5 * fs
    # Notch Filter
    if notch_freq:
        b_n, a_n = iirnotch(notch_freq / nyq, 30.0)
        data = filtfilt(b_n, a_n, data)
    # Bandpass (1-15 Hz)
    low, high = 1.0 / nyq, 15.0 / nyq
    b_b, a_b = butter(4, [low, high], btype='band')
    data = filtfilt(b_b, a_b, data)
    return data

def eeg_detector_loop():
    global detector_running, fs_global, last_blink_ts, blink_pending, last_event_text, signal_buffer
    params = BrainFlowInputParams()
    params.serial_port = SERIAL_PORT
    board = BoardShim(BOARD_ID, params)
    
    try:
        # Safety check for port
        if board.is_prepared(): 
            try: board.release_session()
            except: pass
            
        board.prepare_session()
        board.start_stream(45000)
        fs = BoardShim.get_sampling_rate(BOARD_ID)
        fs_global = fs
        eeg_rows = BoardShim.get_eeg_channels(BOARD_ID)
        idx_a, idx_b = CYTON_BLINK_CH1 - 1, CYTON_BLINK_CH2 - 1
        
        # Buffer for stable filtering window
        raw_processing_buffer = deque(maxlen=int(fs * 2)) 
        blink_active = False

        while detector_running:
            data = board.get_board_data()
            if data.size == 0:
                time.sleep(0.01)
                continue
            
            # Combine channels
            raw_chunk = 0.5 * (data[eeg_rows[idx_a], :] + data[eeg_rows[idx_b], :])
            for sample in raw_chunk: 
                raw_processing_buffer.append(sample)
            
            # Wait for buffer to fill slightly
            if len(raw_processing_buffer) < fs: 
                continue

            # Filter the whole window
            filtered_window = apply_filters(np.array(raw_processing_buffer), fs, NOTCH_HZ)
            
            # Update Shared Plot Buffer
            with state_lock:
                new_filtered = filtered_window[-len(raw_chunk):]
                for s in new_filtered: signal_buffer.append(float(s))
                curr_th, curr_cd = threshold_uv, cooldown_secs

            # Detect Blinks on new samples
            for val in filtered_window[-len(raw_chunk):]:
                abs_val = abs(val)
                now = time.time()
                
                # Schmitt Trigger Logic
                if abs_val > curr_th and not blink_active and (now - last_blink_ts) > curr_cd:
                    blink_active = True
                    with state_lock:
                        last_blink_ts, blink_pending = now, True
                        last_event_text = f"BLINK! {abs_val:.1f} uV"
                elif blink_active and abs_val < (curr_th * BLINK_OFF_FACTOR):
                    blink_active = False
            
            time.sleep(0.02)
            
    except Exception as e:
        with state_lock: last_event_text = f"Error: {e}"
    finally:
        try:
            board.stop_stream()
            board.release_session()
        except: pass

# --- API ROUTES ---
@app.route("/")
def index(): return render_template_string(INDEX_HTML)

@app.route("/api/status")
def api_status():
    with state_lock:
        return jsonify({"detector_running": detector_running, "last_event": last_event_text, "threshold_uv": threshold_uv})

@app.route("/api/start_detector", methods=["POST"])
def api_start_detector():
    global detector_thread, detector_running
    with state_lock:
        if detector_running: return jsonify({"ok": True})
        detector_running = True
        detector_thread = threading.Thread(target=eeg_detector_loop, daemon=True)
        detector_thread.start()
    return jsonify({"ok": True})

@app.route("/api/stop_detector", methods=["POST"])
def api_stop_detector():
    global detector_running
    with state_lock: detector_running = False
    return jsonify({"ok": True})

@app.route("/api/mark_calibration", methods=["POST"])
def mark_calibration():
    global calibration_readings
    label = request.get_json().get("type")
    with state_lock:
        # Grab last 1 sec of clean signal
        recent = list(signal_buffer)[-int(fs_global):]
        if recent:
            val = np.max(np.abs(recent))
            calibration_readings["blinks" if label == "blink" else "noise"].append(val)
    return jsonify({"ok": True})

@app.route("/api/finish_calibration", methods=["POST"])
def finish_calibration():
    global threshold_uv, calibration_readings
    if not calibration_readings["blinks"]: return jsonify({"success": False})
    
    avg_blink = np.mean(calibration_readings["blinks"])
    avg_noise = np.mean(calibration_readings["noise"]) if calibration_readings["noise"] else 30.0
    
    # Threshold Formula: Noise + 60% of the gap to Blink
    suggested = avg_noise + (avg_blink - avg_noise) * 0.6
    
    with state_lock:
        threshold_uv = round(float(suggested), 1)
        calibration_readings = {"noise": [], "blinks": []}
    return jsonify({"success": True, "suggested_threshold": threshold_uv})

@app.route("/api/signal")
def api_signal():
    with state_lock: samples = list(signal_buffer)
    # Downsample for UI performance
    if len(samples) > MAX_SIGNAL_POINTS:
        step = len(samples) // MAX_SIGNAL_POINTS
        samples = samples[::step]
    return jsonify({"samples": samples})

@app.route("/api/events")
def api_events():
    global blink_pending
    with state_lock:
        blink, blink_pending = blink_pending, False
        le = last_event_text
    return jsonify({"blink": blink, "last_event": le})

@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    global threshold_uv, cooldown_secs
    if request.method == "POST":
        data = request.get_json()
        with state_lock:
            # Only update provided fields
            if "threshold_uv" in data and data["threshold_uv"] != "":
                threshold_uv = float(data["threshold_uv"])
            if "cooldown_secs" in data and data["cooldown_secs"] != "":
                cooldown_secs = float(data["cooldown_secs"])
        return jsonify({"ok": True})
    return jsonify({"threshold_uv": threshold_uv, "cooldown_secs": cooldown_secs})

INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>EEG Pong Dashboard</title>
  <style>
    body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0a0a0a; color: #e0e0e0; margin: 0; display: flex; flex-direction: column; height: 100vh; }
    header { padding: 1rem 2rem; background: #111; border-bottom: 1px solid #333; display:flex; align-items:center; justify-content:space-between; }
    h1 { margin: 0; font-size: 1.5rem; color: #ff9800; }
    
    .main-container { display: flex; flex: 1; overflow: hidden; }
    .sidebar { width: 25%; min-width: 300px; background: #151515; border-right: 1px solid #333; padding: 1.5rem; overflow-y: auto; }
    .game-area { flex: 1; padding: 1.5rem; display: flex; flex-direction: column; align-items: center; justify-content: center; background: #050505; }

    .card { background: #202020; border-radius: 8px; padding: 1.25rem; margin-bottom: 1.5rem; border: 1px solid #333; }
    .card h2 { font-size: 0.9rem; margin-top: 0; color: #aaa; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px; }

    button { padding: 0.7rem; border-radius: 4px; border: none; cursor: pointer; font-weight: bold; transition: 0.2s; width: 100%; margin-bottom: 0.5rem; }
    .start { background: #388e3c; color: white; }
    .stop { background: #d32f2f; color: white; }
    .calib { background: #f57c00; color: black; }
    button:hover { opacity: 0.8; }
    button:disabled { background: #444; cursor: not-allowed; opacity: 0.6; }

    input, select { background: #000; color: #fff; border: 1px solid #444; padding: 10px; border-radius: 4px; margin-bottom: 1rem; width: 100%; box-sizing: border-box; }
    label { font-size: 0.8rem; color: #999; display: block; margin-bottom: 0.2rem; }

    #calib-light { width: 30px; height: 30px; border-radius: 50%; background: #333; border: 2px solid #555; transition: background 0.2s; }
    #signalCanvas { background: #000; border: 1px solid #333; border-radius: 4px; width: 100%; height: 120px; }
    #pongCanvas { border: 2px solid #333; box-shadow: 0 0 30px rgba(0,0,0,0.5); max-width: 100%; background: #000; }
    
    .status-group { display: flex; align-items: center; gap: 10px; margin-bottom: 1rem; }
    .status-dot { width: 10px; height: 10px; border-radius: 50%; }
    .dot-on { background: #4caf50; box-shadow: 0 0 8px #4caf50; }
    .dot-off { background: #f44336; }
  </style>
</head>
<body>
  <header>
    <h1>🧠 EEG Pong Dashboard</h1>
    <div style="font-size: 0.8rem; color: #666;">v4.0 Final</div>
  </header>
  
  <div class="main-container">
    <div class="sidebar">
      <div class="card">
        <h2>System Status</h2>
        <div class="status-group">
          <div class="status-dot" id="detector-dot"></div>
          <span id="detector-status" style="font-weight: bold;">STOPPED</span>
        </div>
        <div style="display: flex; gap: 5px;">
          <button class="start" id="btn-start-detector">Start Detector</button>
          <button class="stop" id="btn-stop-detector">Stop</button>
        </div>
      </div>

      <div class="card" style="border-color: #ff9800;">
        <h2>Calibration</h2>
        <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 1rem;">
          <div id="calib-light"></div>
          <span id="calib-text" style="font-size: 0.9rem; color: #ff9800;">1. Start Detector<br>2. Run Routine</span>
        </div>
        <button class="calib" id="btn-run-calib">Run Guided Calibration</button>
      </div>

      <div class="card">
        <h2>Game Settings</h2>
        <label>Threshold (µV) <span style="font-size:0.7em">(Set by Calibration)</span></label>
        <input type="number" id="input-threshold" placeholder="150.0">
        
        <label>Cooldown (Seconds)</label>
        <input type="number" id="input-cooldown" step="0.05" placeholder="0.20">
        
        <label>Level / Difficulty</label>
        <select id="select-level">
          <option value="1">Level 1 - Easy</option>
          <option value="2">Level 2 - Normal</option>
          <option value="3">Level 3 - Fast</option>
          <option value="4">Level 4 - Expert</option>
        </select>

        <label style="display: flex; align-items: center; cursor: pointer; color: #fff; margin-top: 10px;">
          <input type="checkbox" id="chk-two-player" style="width: auto; margin-right: 10px; margin-bottom: 0;"> 
          2-Player Mode (Arrows)
        </label>
        
        <button class="start" id="btn-apply-settings" style="margin-top: 15px; background: #444;">Apply Manual Settings</button>
      </div>

      <div class="card">
        <h2>Live Signal</h2>
        <canvas id="signalCanvas"></canvas>
      </div>
    </div>

    <div class="game-area">
      <canvas id="pongCanvas" width="700" height="450"></canvas>
      <div id="last-event" style="margin-top: 15px; font-family: monospace; color: #666;">Last Event: None</div>
    </div>
  </div>

<script>
// --- API HELPER ---
async function api(path, method="GET", body=null) {
  const opts = { method, headers: body ? {"Content-Type": "application/json"} : {} };
  if (body) opts.body = JSON.stringify(body);
  const res = await fetch(path, opts); return await res.json();
}

// --- DOM ELEMENTS ---
const inputThreshold = document.getElementById("input-threshold");
const inputCooldown = document.getElementById("input-cooldown");
const lastEventDiv = document.getElementById("last-event");
const levelSelect = document.getElementById("select-level");
const twoPlayerChk = document.getElementById("chk-two-player");

// --- CALIBRATION ROUTINE ---
document.getElementById("btn-run-calib").onclick = async () => {
  const light = document.getElementById("calib-light");
  const text = document.getElementById("calib-text");
  const btn = document.getElementById("btn-run-calib");
  
  // Safety: Check if detector is running
  const status = await api("/api/status");
  if(!status.detector_running) {
    alert("Please click 'Start Detector' first!");
    return;
  }

  btn.disabled = true;
  for (let i = 0; i < 4; i++) {
    // Noise Phase
    light.style.background = "#333"; text.innerText = "RELAX...";
    await new Promise(r => setTimeout(r, 2000));
    await api("/api/mark_calibration", "POST", { type: "noise" });
    
    // Blink Phase
    light.style.background = "#00ff00"; light.style.boxShadow = "0 0 15px #00ff00"; text.innerText = "BLINK NOW!";
    await new Promise(r => setTimeout(r, 800));
    await api("/api/mark_calibration", "POST", { type: "blink" });
    light.style.boxShadow = "none";
  }
  
  const res = await api("/api/finish_calibration", "POST");
  if (res.success) {
    text.innerText = "DONE!";
    inputThreshold.value = res.suggested_threshold;
    alert("Calibration Complete! Threshold set to " + res.suggested_threshold);
  } else {
    text.innerText = "Failed";
  }
  btn.disabled = false;
  light.style.background = "#333";
};

// --- SIGNAL PLOTTING ---
const sigCanvas = document.getElementById("signalCanvas");
const sigCtx = sigCanvas.getContext("2d");

async function pollSignal() {
  const s = await api("/api/signal");
  if (!s.samples || s.samples.length < 2) return;
  
  const w = sigCanvas.width;
  const h = sigCanvas.height;
  
  sigCtx.fillStyle = "#000"; sigCtx.fillRect(0,0,w,h);
  
  // Grid Lines
  sigCtx.strokeStyle = "#222"; sigCtx.lineWidth = 1;
  sigCtx.beginPath();
  sigCtx.moveTo(0, h/2); sigCtx.lineTo(w, h/2); // Center line
  sigCtx.stroke();
  
  let min = Math.min(...s.samples);
  let max = Math.max(...s.samples);
  let range = Math.max(max - min, 10); // avoid div/0
  
  // Draw Waveform
  sigCtx.strokeStyle = "#00e676"; sigCtx.lineWidth = 1.5; sigCtx.beginPath();
  for (let i=0; i<s.samples.length; i++) {
    const x = (i / (s.samples.length-1)) * w;
    const norm = (s.samples[i] - min) / range;
    const y = h - (norm * h);
    if(i===0) sigCtx.moveTo(x,y); else sigCtx.lineTo(x,y);
  }
  sigCtx.stroke();
}

// --- PONG GAME ENGINE ---
const pongCanvas = document.getElementById("pongCanvas");
const ctx = pongCanvas.getContext("2d");

// Game State
let p1Y=175, p2Y=175, p1Dir=0, p2Dir=0;
let ballX=350, ballY=225, ballVx=0, ballVy=0;
let blinkFlag=false;
let lastTime=0;

// Level Configs (Pixels Per Second)
function getLevelConfig(level) {
  const lvl = parseInt(level);
  // Easy
  if (lvl === 1) return { paddleH: 100, ballSize: 15, ballVx: 200, ballVy: 150, paddleSpeed: 250 };
  // Normal
  if (lvl === 2) return { paddleH: 80, ballSize: 12, ballVx: 300, ballVy: 200, paddleSpeed: 300 };
  // Fast
  if (lvl === 3) return { paddleH: 60, ballSize: 10, ballVx: 400, ballVy: 300, paddleSpeed: 350 };
  // Expert
  return { paddleH: 50, ballSize: 10, ballVx: 550, ballVy: 400, paddleSpeed: 500 };
}

// Keyboard Input
window.addEventListener("keydown", (e) => {
  if(["ArrowUp", "ArrowDown"].includes(e.key)) {
    e.preventDefault();
    p2Dir = (e.key === "ArrowUp") ? -1 : 1;
  }
});
window.addEventListener("keyup", (e) => {
  if(["ArrowUp", "ArrowDown"].includes(e.key)) p2Dir = 0;
});

// Init Ball
function resetBall(cfg) {
  ballX = pongCanvas.width / 2;
  ballY = pongCanvas.height / 2;
  ballVx = Math.random() > 0.5 ? cfg.ballVx : -cfg.ballVx;
  ballVy = Math.random() > 0.5 ? cfg.ballVy : -cfg.ballVy;
}

// Main Loop
function gameLoop(timestamp) {
  if (!lastTime) lastTime = timestamp;
  const dt = (timestamp - lastTime) / 1000; // seconds
  lastTime = timestamp;

  // Limit max dt (pause fix)
  if (dt > 0.1) { requestAnimationFrame(gameLoop); return; }

  const cfg = getLevelConfig(levelSelect.value);
  const W = pongCanvas.width;
  const H = pongCanvas.height;

  // Start ball if stationary
  if (ballVx === 0) resetBall(cfg);

  // -- UPDATE POSITIONS --
  
  // Player 1 (EEG Blink)
  if(blinkFlag) {
    p1Dir = (p1Dir === 0) ? -1 : (p1Dir * -1);
    blinkFlag = false;
  }
  
  p1Y += p1Dir * cfg.paddleSpeed * dt;

  // NEW: Bounce off walls instead of sticking
  if (p1Y <= 0) {
      p1Y = 0;       
      p1Dir = 1;     // Hit top, go down
  } else if (p1Y >= H - cfg.paddleH) {
      p1Y = H - cfg.paddleH; 
      p1Dir = -1;    // Hit bottom, go up
  }

  // Player 2 (Keyboard)
  if(twoPlayerChk.checked) {
    p2Y += p2Dir * cfg.paddleSpeed * dt;
    p2Y = Math.max(0, Math.min(H - cfg.paddleH, p2Y));
  }

  // Ball
  ballX += ballVx * dt;
  ballY += ballVy * dt;

  // -- COLLISIONS --
  
  // Floor/Ceiling
  if (ballY <= 0 || ballY >= H - cfg.ballSize) {
    ballVy *= -1;
    ballY = Math.max(0, Math.min(H - cfg.ballSize, ballY));
  }

  // P1 Paddle (Left)
  // X range: 10 to 20 (paddle width 10)
  if (ballX <= 20 && ballX >= 10) {
    if (ballY + cfg.ballSize >= p1Y && ballY <= p1Y + cfg.paddleH) {
      ballVx = Math.abs(ballVx); // Force right
      ballX = 20;
    }
  }

  // P2 Paddle (Right) or Wall
  if (twoPlayerChk.checked) {
    // P2 is at W - 20
    if (ballX + cfg.ballSize >= W - 20 && ballX + cfg.ballSize <= W - 10) {
      if (ballY + cfg.ballSize >= p2Y && ballY <= p2Y + cfg.paddleH) {
        ballVx = -Math.abs(ballVx); // Force left
        ballX = W - 20 - cfg.ballSize;
      }
    }
  } else {
    // Wall Bounce
    if (ballX >= W - cfg.ballSize) {
      ballVx = -Math.abs(ballVx);
      ballX = W - cfg.ballSize;
    }
  }

  // Scoring / Reset
  if (ballX < 0 || ballX > W) {
    resetBall(cfg);
  }

  // -- DRAW --
  ctx.fillStyle = "#000"; ctx.fillRect(0,0,W,H);
  
  // P1
  ctx.fillStyle = "#ff9800"; 
  ctx.fillRect(10, p1Y, 10, cfg.paddleH);
  
  // P2
  if (twoPlayerChk.checked) {
    ctx.fillStyle = "#2196f3";
    ctx.fillRect(W - 20, p2Y, 10, cfg.paddleH);
  }
  
  // Ball
  ctx.fillStyle = "#fff";
  ctx.beginPath();
  ctx.arc(ballX + cfg.ballSize/2, ballY + cfg.ballSize/2, cfg.ballSize/2, 0, Math.PI*2);
  ctx.fill();

  requestAnimationFrame(gameLoop);
}

// --- POLLING & INIT ---
async function refreshStatus() {
  const s = await api("/api/status");
  document.getElementById("detector-dot").className = "status-dot " + (s.detector_running ? "dot-on":"dot-off");
  document.getElementById("detector-status").innerText = s.detector_running ? "RUNNING":"STOPPED";
}
async function pollEvents() {
  const e = await api("/api/events");
  if(e.blink) blinkFlag = true;
  lastEventDiv.innerText = "Last Event: " + e.last_event;
}

// Bind Buttons
document.getElementById("btn-start-detector").onclick = () => api("/api/start_detector","POST").then(refreshStatus);
document.getElementById("btn-stop-detector").onclick = () => api("/api/stop_detector","POST").then(refreshStatus);
document.getElementById("btn-apply-settings").onclick = () => {
    api("/api/settings", "POST", {
        threshold_uv: inputThreshold.value, 
        cooldown_secs: inputCooldown.value
    });
};

// Start
api("/api/settings").then(s => { 
    inputThreshold.value = s.threshold_uv; 
    inputCooldown.value = s.cooldown_secs; 
});
setInterval(pollSignal, 200); 
setInterval(pollEvents, 50); 
setInterval(refreshStatus, 2000);
requestAnimationFrame(gameLoop);

</script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)