# app.py
# Simple web frontend on http://localhost:5000 to control:
#  - EEG blink detector (eeg_blink_to_udp_threshold100.py)
#  - Pong game (pong_udp.py)

from flask import Flask, jsonify, request, render_template_string
import subprocess
import threading
import os
import signal
import sys

app = Flask(__name__)

# Paths to your scripts (same folder as this app.py)
DETECTOR_SCRIPT = "blink_detector_v2.py"
GAME_SCRIPT = "pong.py"

detector_proc = None
game_proc = None
lock = threading.Lock()

# ---------- HTML FRONTEND (simple) ----------
INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>EEG Pong Control Panel</title>
  <style>
    body {
      font-family: system-ui, sans-serif;
      background: #111;
      color: #eee;
      margin: 0;
      padding: 2rem;
    }
    h1 { margin-top: 0; }
    button {
      padding: 0.6rem 1.4rem;
      margin: 0.4rem 0.4rem 0.4rem 0;
      border-radius: 6px;
      border: none;
      font-size: 1rem;
      cursor: pointer;
    }
    button.start { background: #2e7d32; color: #fff; }
    button.stop { background: #c62828; color: #fff; }
    button:disabled { opacity: 0.4; cursor: default; }
    .card {
      background: #1e1e1e;
      padding: 1rem 1.5rem;
      border-radius: 12px;
      margin-bottom: 1rem;
      box-shadow: 0 0 0 1px #333;
    }
    .status-dot {
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 50%;
      margin-right: 0.4rem;
    }
    .dot-on { background: #4caf50; }
    .dot-off { background: #f44336; }
    #status-text {
      white-space: pre-line;
      font-family: monospace;
      font-size: 0.9rem;
      margin-top: 0.5rem;
      color: #ccc;
    }
  </style>
</head>
<body>
  <h1>🧠 EEG Pong Control Panel</h1>

  <div class="card">
    <h2>1. Blink Detector / Calibration</h2>
    <p>
      This starts <code>{{ detector_script }}</code>, which performs baseline warm-up and then
      streams blink events over UDP to the Pong game.
    </p>
    <p>
      <span class="status-dot" id="detector-dot"></span>
      <strong>Detector status:</strong>
      <span id="detector-status">unknown</span>
    </p>
    <button class="start" id="btn-start-detector">Start Calibration + Detector</button>
    <button class="stop" id="btn-stop-detector">Stop Detector</button>
  </div>

  <div class="card">
    <h2>2. Pong Game</h2>
    <p>
      This starts <code>{{ game_script }}</code>. Make sure the detector is running so that blinks
      are sent as <code>EVENT:BLINK</code> over UDP (127.0.0.1:8765).
    </p>
    <p>
      <span class="status-dot" id="game-dot"></span>
      <strong>Game status:</strong>
      <span id="game-status">unknown</span>
    </p>
    <button class="start" id="btn-start-game">Start Game</button>
    <button class="stop" id="btn-stop-game">Stop Game</button>
  </div>

  <div class="card">
    <h2>Log / Status</h2>
    <div id="status-text">Waiting for status...</div>
  </div>

<script>
async function callApi(path, method="POST") {
  const res = await fetch(path, {method});
  return await res.json();
}

function updateUI(status) {
  const detOn = status.detector_running;
  const gameOn = status.game_running;

  const detDot = document.getElementById("detector-dot");
  const gameDot = document.getElementById("game-dot");
  const detStatus = document.getElementById("detector-status");
  const gameStatus = document.getElementById("game-status");
  const statusText = document.getElementById("status-text");

  detDot.className = "status-dot " + (detOn ? "dot-on" : "dot-off");
  gameDot.className = "status-dot " + (gameOn ? "dot-on" : "dot-off");

  detStatus.textContent = detOn ? "running" : "stopped";
  gameStatus.textContent = gameOn ? "running" : "stopped";

  statusText.textContent = status.message || "";
}

async function refreshStatus() {
  try {
    const status = await callApi("/api/status", "GET");
    updateUI(status);
  } catch (e) {
    console.error(e);
  }
}

document.getElementById("btn-start-detector").onclick = async () => {
  await callApi("/api/start_detector");
  await refreshStatus();
};
document.getElementById("btn-stop-detector").onclick = async () => {
  await callApi("/api/stop_detector");
  await refreshStatus();
};
document.getElementById("btn-start-game").onclick = async () => {
  await callApi("/api/start_game");
  await refreshStatus();
};
document.getElementById("btn-stop-game").onclick = async () => {
  await callApi("/api/stop_game");
  await refreshStatus();
};

// Poll status every 2 seconds
setInterval(refreshStatus, 2000);
refreshStatus();
</script>
</body>
</html>
"""

# ---------- Helper functions ----------

def is_proc_running(proc):
    return proc is not None and proc.poll() is None

def start_detector():
    global detector_proc
    with lock:
        if is_proc_running(detector_proc):
            return False
        print("Starting detector process...")
        detector_proc = subprocess.Popen([sys.executable, DETECTOR_SCRIPT])
        return True

def stop_detector():
    global detector_proc
    with lock:
        if not is_proc_running(detector_proc):
            return False
        print("Stopping detector process...")
        if os.name == "nt":
            detector_proc.terminate()
        else:
            os.kill(detector_proc.pid, signal.SIGTERM)
        detector_proc = None
        return True

def start_game():
    global game_proc
    with lock:
        if is_proc_running(game_proc):
            return False
        print("Starting game process...")
        game_proc = subprocess.Popen([sys.executable, GAME_SCRIPT])
        return True

def stop_game():
    global game_proc
    with lock:
        if not is_proc_running(game_proc):
            return False
        print("Stopping game process...")
        if os.name == "nt":
            game_proc.terminate()
        else:
            os.kill(game_proc.pid, signal.SIGTERM)
        game_proc = None
        return True

# ---------- Routes ----------

@app.route("/")
def index():
    return render_template_string(INDEX_HTML,
                                  detector_script=DETECTOR_SCRIPT,
                                  game_script=GAME_SCRIPT)

@app.route("/api/status", methods=["GET"])
def api_status():
    msg_lines = []
    if is_proc_running(detector_proc):
        msg_lines.append("Detector: running (see terminal for detailed logs)")
    else:
        msg_lines.append("Detector: stopped")

    if is_proc_running(game_proc):
        msg_lines.append("Game: running (Pong window should be open)")
    else:
        msg_lines.append("Game: stopped")

    return jsonify({
        "detector_running": is_proc_running(detector_proc),
        "game_running": is_proc_running(game_proc),
        "message": "\n".join(msg_lines)
    })

@app.route("/api/start_detector", methods=["POST"])
def api_start_detector():
    started = start_detector()
    return jsonify({"ok": True, "started": started})

@app.route("/api/stop_detector", methods=["POST"])
def api_stop_detector():
    stopped = stop_detector()
    return jsonify({"ok": True, "stopped": stopped})

@app.route("/api/start_game", methods=["POST"])
def api_start_game():
    started = start_game()
    return jsonify({"ok": True, "started": started})

@app.route("/api/stop_game", methods=["POST"])
def api_stop_game():
    stopped = stop_game()
    return jsonify({"ok": True, "stopped": stopped})

if __name__ == "__main__":
    # Run Flask on localhost:5000
    app.run(host="127.0.0.1", port=5000, debug=True)