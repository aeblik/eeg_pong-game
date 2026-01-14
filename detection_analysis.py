import time
import threading
from collections import deque
import numpy as np
from flask import Flask, jsonify, request, render_template_string
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from scipy.signal import butter, filtfilt, iirnotch

# ------------- CONFIG -------------
SERIAL_PORT = "COM3"                
BOARD_ID = BoardIds.CYTON_BOARD.value
CYTON_BLINK_CH1 = 1                 
CYTON_BLINK_CH2 = 2                 
NOTCH_HZ   = 50                     
BLINK_OFF_FACTOR = 0.3

app = Flask(__name__)

# --- SHARED STATE ---
state_lock = threading.Lock()
detector_running = False
threshold_uv = 150.0      
cooldown_secs = 0.2
signal_buffer = deque(maxlen=2500)
calibration_readings = {"noise": [], "blinks": []}
fs_global = 250.0

# Stats for the Thesis (12 cues in 60 seconds)
stats = {
    "test_active": False,
    "current_cue": "idle", 
    "tp": 0, "fp": 0, "fn": 0,
    "total_cues": 12, 
    "progress": 0
}

blink_detected_in_window = False

# --- EXACT FILTER LOGIC FROM APP ---
def apply_filters(data, fs, notch_freq=50):
    if len(data) < 100: return data

    data = data-np.median(data) 

    nyq = 0.5 * fs
    # Notch Filter
    if notch_freq:
        b_n, a_n = iirnotch(notch_freq / nyq, 30.0)
        data = filtfilt(b_n, a_n, data)
    # Bandpass (1-10 Hz)
    low, high = 1.0 / nyq, 10.0 / nyq
    b_b, a_b = butter(4, [low, high], btype='band')
    data = filtfilt(b_b, a_b, data)
    return data

# --- DETECTOR LOOP (Matched to App) ---
def eeg_loop():
    global detector_running, blink_detected_in_window, fs_global
    params = BrainFlowInputParams()
    params.serial_port = SERIAL_PORT
    board = BoardShim(BOARD_ID, params)
    try:
        board.prepare_session()
        board.start_stream(45000)
        fs = board.get_sampling_rate(BOARD_ID)
        fs_global = fs
        eeg_ch = board.get_eeg_channels(BOARD_ID)
        idx = [CYTON_BLINK_CH1-1, CYTON_BLINK_CH2-1]
        
        # Buffer for stable filtering window as in the app
        raw_processing_buffer = deque(maxlen=int(fs * 2)) 
        last_blink_ts = 0
        blink_active = False

        while detector_running:
            data = board.get_board_data()
            if data.size == 0:
                time.sleep(0.01)
                continue
            
            # Combine channels
            chunk = 0.5 * (data[eeg_ch[idx[0]], :] + data[eeg_ch[idx[1]], :])
            for sample in chunk: 
                raw_processing_buffer.append(sample)
            
            if len(raw_processing_buffer) < fs: 
                continue

            # Filter the whole window using the exact function
            filtered_window = apply_filters(np.array(raw_processing_buffer), fs, NOTCH_HZ)
            new_filtered = filtered_window[-len(chunk):]
            
            with state_lock:
                for s in new_filtered: signal_buffer.append(float(s))
                curr_th, curr_cd = threshold_uv, cooldown_secs

            # Detect Blinks (Schmitt Trigger approach)
            for val in new_filtered:
                abs_val = abs(val)
                now = time.time()
                
                if abs_val > curr_th and not blink_active and (now - last_blink_ts) > curr_cd:
                    blink_active = True
                    last_blink_ts = now
                    blink_detected_in_window = True 
                elif blink_active and abs_val < (curr_th * BLINK_OFF_FACTOR):
                    blink_active = False
            time.sleep(0.02)
    finally:
        try:
            board.stop_stream()
            board.release_session()
        except: pass

# --- PERFORMANCE TEST THREAD ---
def run_test_logic():
    global blink_detected_in_window
    with state_lock:
        stats.update({"test_active": True, "tp": 0, "fp": 0, "fn": 0, "progress": 0})
    
    for i in range(stats["total_cues"]):
        # RELAX (3.5s)
        stats["current_cue"] = "relax"
        blink_detected_in_window = False
        time.sleep(3.5)
        if blink_detected_in_window:
            with state_lock: stats["fp"] += 1
            blink_detected_in_window = False 

        # BLINK (1.5s)
        stats["current_cue"] = "blink"
        blink_detected_in_window = False
        time.sleep(1.5)
        
        with state_lock:
            if blink_detected_in_window: stats["tp"] += 1
            else: stats["fn"] += 1
            stats["progress"] = i + 1
            
    stats["current_cue"] = "finished"
    stats["test_active"] = False

# --- FLASK ROUTES ---
@app.route("/")
def index(): return render_template_string(HTML)

@app.route("/api/control", methods=["POST"])
def control():
    global detector_running
    action = request.json.get("action")
    if action == "start":
        if not detector_running:
            detector_running = True
            threading.Thread(target=eeg_loop, daemon=True).start()
    else:
        detector_running = False
    return jsonify({"ok": True})

@app.route("/api/start_test", methods=["POST"])
def start_test():
    threading.Thread(target=run_test_logic, daemon=True).start()
    return jsonify({"ok": True})

@app.route("/api/mark_calibration", methods=["POST"])
def mark_calibration():
    global calibration_readings
    label = request.get_json().get("type")
    with state_lock:
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
    suggested = avg_noise + (avg_blink - avg_noise) * 0.6
    with state_lock:
        threshold_uv = round(float(suggested), 1)
        calibration_readings = {"noise": [], "blinks": []}
    return jsonify({"success": True, "suggested_threshold": threshold_uv})

@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    global threshold_uv, cooldown_secs
    if request.method == "POST":
        data = request.get_json()
        with state_lock:
            if "threshold_uv" in data and data["threshold_uv"]: threshold_uv = float(data["threshold_uv"])
            if "cooldown_secs" in data and data["cooldown_secs"]: cooldown_secs = float(data["cooldown_secs"])
        return jsonify({"ok": True})
    return jsonify({"threshold_uv": threshold_uv, "cooldown_secs": cooldown_secs})

@app.route("/api/stats")
def get_stats():
    with state_lock:
        tp, fp, fn = stats['tp'], stats['fp'], stats['fn']
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return jsonify({**stats, "precision": round(precision, 2), "recall": round(recall, 2), "f1": round(f1, 2), "running": detector_running, "threshold_uv": threshold_uv, "cooldown_secs": cooldown_secs})

@app.route("/api/signal")
def get_signal():
    with state_lock: return jsonify({"samples": list(signal_buffer)[::2]})

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>EEG Thesis Measurement (12 Cues)</title>
    <style>
        body { font-family: 'Segoe UI', sans-serif; background: #0a0a0a; color: #eee; text-align: center; padding: 20px; }
        .grid { display: grid; grid-template-columns: 320px 1fr 1fr; gap: 20px; max-width: 1400px; margin: auto; }
        .card { background: #181818; padding: 20px; border-radius: 12px; border: 1px solid #333; }
        .cue-box { font-size: 3rem; font-weight: 800; height: 140px; display: flex; align-items: center; justify-content: center; border-radius: 12px; margin: 15px 0; transition: 0.3s; }
        .relax { background: #222; color: #444; border: 2px solid #333; }
        .blink { background: #1b5e20; color: #fff; box-shadow: 0 0 30px rgba(76, 175, 80, 0.3); border: 2px solid #4caf50; }
        .metric { font-size: 2.2rem; color: #ff9800; font-weight: bold; }
        button { padding: 12px; width: 100%; font-weight: bold; cursor: pointer; border-radius: 6px; border: none; margin: 5px 0; font-size: 0.9rem; }
        input { background: #000; border: 1px solid #444; color: #fff; padding: 8px; width: 80%; border-radius: 4px; margin-bottom: 10px; }
        label { display: block; font-size: 0.8rem; color: #888; margin-top: 10px; }
        #sig { width: 100%; height: 120px; background: #000; border-radius: 8px; }
        #calib-light { width: 20px; height: 20px; border-radius: 50%; background: #333; margin: auto; border: 1px solid #555; }
    </style>
</head>
<body>
    <h1 style="color: #ff9800; margin-bottom: 25px;">Quantitative Measurement Phase</h1>
    <div class="grid">
        <div class="card">
            <h3 style="margin-top:0">Step 1: Setup</h3>
            <button style="background: #333; color:white;" onclick="control('start')">Initialize Cyton</button>
            <hr style="border-color:#333">
            <h3>Step 2: Calibrate</h3>
            <div id="calib-light"></div>
            <p id="calib-text" style="font-size:0.8rem">Waiting for Cyton...</p>
            <button style="background: #f57c00; color: black;" id="btn-calib">Run Guided Calibration</button>
            <hr style="border-color:#333">
            <label>Threshold (µV)</label>
            <input type="number" id="in-th">
            <label>Cooldown (s)</label>
            <input type="number" id="in-cd" step="0.05">
            <button style="background: #444; color:white;" onclick="applySettings()">Update Params</button>
        </div>

        <div class="card">
            <h3>Step 3: Experiment</h3>
            <div id="cue" class="cue-box relax">IDLE</div>
            <p style="font-size: 1.1rem;">Progress: <span id="prog" style="color: #ff9800;">0</span> / 12</p>
            <button style="background:#ff9800; color: black; font-size:1.1rem; padding: 15px;" onclick="startTest()">START 60s RECORDING</button>
        </div>

        <div class="card">
            <h3>Step 4: Metrics</h3>
            <div style="display: flex; justify-content: space-around;">
                <div><p style="margin:2px">TP</p><span id="tp" class="metric">0</span></div>
                <div><p style="margin:2px">FP</p><span id="fp" class="metric">0</span></div>
                <div><p style="margin:2px">FN</p><span id="fn" class="metric">0</span></div>
            </div>
            <hr style="border-color:#333; margin: 15px 0;">
            <p>Precision: <span id="pre" style="font-weight:bold">0</span></p>
            <p>Recall: <span id="rec" style="font-weight:bold">0</span></p>
            <h2 style="color:#4caf50; margin-top:10px;">F1-Score: <span id="f1">0</span></h2>
        </div>
    </div>
    <div class="card" style="margin-top:20px; max-width:1400px; margin: 20px auto;">
        <canvas id="sig"></canvas>
    </div>

    <script>
    async function api(path, method="GET", body=null) {
        const opts = { method, headers: body ? {"Content-Type":"application/json"} : {} };
        if(body) opts.body = JSON.stringify(body);
        return fetch(path, opts).then(r => r.json());
    }

    function control(action) { api('/api/control', 'POST', {action}); }
    function startTest() { api('/api/start_test', 'POST'); }
    function applySettings() {
        api('/api/settings', 'POST', {
            threshold_uv: document.getElementById('in-th').value,
            cooldown_secs: document.getElementById('in-cd').value
        });
    }

    document.getElementById('btn-calib').onclick = async () => {
        const light = document.getElementById('calib-light');
        const txt = document.getElementById('calib-text');
        for(let i=0; i<4; i++) {
            light.style.background = "#333"; txt.innerText = "RELAX...";
            await new Promise(r => setTimeout(r, 2000));
            await api('/api/mark_calibration', 'POST', {type:'noise'});
            light.style.background = "#00ff00"; txt.innerText = "BLINK!";
            await new Promise(r => setTimeout(r, 800));
            await api('/api/mark_calibration', 'POST', {type:'blink'});
        }
        const res = await api('/api/finish_calibration', 'POST');
        if(res.success) {
            document.getElementById('in-th').value = res.suggested_threshold;
            txt.innerText = "New Threshold: " + res.suggested_threshold + "uV";
            applySettings();
        }
    };

    function update() {
        api('/api/stats').then(data => {
            const cue = document.getElementById('cue');
            cue.innerText = data.current_cue.toUpperCase();
            cue.className = 'cue-box ' + data.current_cue;
            document.getElementById('prog').innerText = data.progress;
            document.getElementById('tp').innerText = data.tp;
            document.getElementById('fp').innerText = data.fp;
            document.getElementById('fn').innerText = data.fn;
            document.getElementById('pre').innerText = data.precision;
            document.getElementById('rec').innerText = data.recall;
            document.getElementById('f1').innerText = data.f1;
            if(document.activeElement.id !== 'in-th') document.getElementById('in-th').value = data.threshold_uv;
            if(document.activeElement.id !== 'in-cd') document.getElementById('in-cd').value = data.cooldown_secs;
        });
        api('/api/signal').then(data => {
            const c = document.getElementById('sig');
            const ctx = c.getContext('2d');
            ctx.clearRect(0,0,c.width,c.height);
            ctx.strokeStyle = '#00ff00';
            ctx.beginPath();
            data.samples.forEach((s, i) => {
                const x = (i/data.samples.length)*c.width;
                const y = (c.height/2) - (s/4);
                if(i==0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
            });
            ctx.stroke();
        });
    }
    setInterval(update, 200);
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(port=5001)