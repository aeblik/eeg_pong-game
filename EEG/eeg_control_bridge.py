# eeg_control_bridge_autodetect.py
import time, socket, sys
import numpy as np
from typing import List, Tuple, Optional
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from scipy.signal import butter, filtfilt, iirnotch, welch

# ===================== USER SETTINGS =====================
SERIAL_PORT = "COM3"         # "COM3" | "/dev/ttyUSB0" | "/dev/tty.usbserial-XXXX"
BOARD_ID = BoardIds.CYTON_BOARD.value

ENABLE_BLINK = True
ENABLE_ALPHA = True

# If/when you know the Cyton inputs (1..8) for your labels, put them here:
LABEL_TO_CYTON = {
    "N1P": 1,
    "N2P": 2,
    "N8P": 7,
    "N7P": 8,
}

NOTCH_HZ   = 50              # 50 in CH/EU, 60 in US; None to disable
BANDPASS   = (1.0, 40.0)     # Hz; None to disable

BLINK_DEBOUNCE_S = 0.35
BLINK_STD_FACTOR = 5.0

ALPHA_BAND   = (8.0, 13.0)
ALPHA_UP_HYST   = 1.10
ALPHA_DOWN_HYST = 0.90

ENABLE_UDP = True
UDP_HOST   = "127.0.0.1"
UDP_PORT   = 8765
# ========================================================

# --------------------- Helpers ---------------------
def design_bandpass(fs: float, low: float, high: float, order: int = 4):
    ny = 0.5 * fs
    b, a = butter(order, [low / ny, high / ny], btype="band")
    return b, a

def design_notch(fs: float, f0: float, q: float = 30.0):
    ny = 0.5 * fs
    w0 = f0 / ny
    return iirnotch(w0, q)

def apply_filters(x: np.ndarray, fs: float, bp=None, nf=None) -> np.ndarray:
    y = x
    if nf is not None:
        b, a = nf
        y = filtfilt(b, a, y, axis=-1, padlen=50)
    if bp is not None:
        b, a = bp
        y = filtfilt(b, a, y, axis=-1, padlen=50)
    return y

def bandpower(sig: np.ndarray, fs: float, fmin: float, fmax: float) -> float:
    f, pxx = welch(sig, fs=fs, nperseg=min(len(sig), 512))
    idx = np.where((f >= fmin) & (f <= fmax))[0]
    return float(np.trapz(pxx[idx], f[idx])) if idx.size else 0.0

def cyton_reduced_index_from_label(label: str, n_eeg: int) -> Optional[int]:
    """
    Map 'N1P' -> Cyton channel number (1..8) -> REDUCED INDEX (0..n_eeg-1).
    Reduced index matches rows of capture_rows() matrices and s = chunk[eeg_rows,:].
    """
    if label not in LABEL_TO_CYTON:
        return None
    cyton_num = LABEL_TO_CYTON[label]
    if not (1 <= cyton_num <= n_eeg):
        raise ValueError(f"Cyton channel number for {label} is out of range: {cyton_num}")
    return cyton_num - 1  # reduced-space index

def autodetect_blink_pair(all_eeg_reduced: np.ndarray, fs: float, bp, nf) -> Tuple[int, int]:
    # all_eeg_reduced: shape (n_eeg x samples), rows 0..n_eeg-1 = Cyton CH1..CH8
    scores = []
    for r in range(all_eeg_reduced.shape[0]):
        x = all_eeg_reduced[r]
        xf = apply_filters(x, fs, bp, nf) if x.size > 100 else x
        score = np.median(np.abs(xf)) + 3.0 * np.std(np.abs(xf))
        scores.append((score, r))
    scores.sort(reverse=True)
    return scores[0][1], scores[1][1]

def autodetect_alpha_pair(open_eeg_reduced: np.ndarray, closed_eeg_reduced: np.ndarray, fs: float, bp, nf) -> Tuple[int, int]:
    deltas = []
    for r in range(open_eeg_reduced.shape[0]):
        xo = open_eeg_reduced[r]
        xc = closed_eeg_reduced[r]
        if xo.size > 100:
            xo = apply_filters(xo, fs, bp, nf)
        if xc.size > 100:
            xc = apply_filters(xc, fs, bp, nf)
        po = bandpower(xo, fs, *ALPHA_BAND)
        pc = bandpower(xc, fs, *ALPHA_BAND)
        deltas.append((pc - po, r))
    deltas.sort(reverse=True)
    return deltas[0][1], deltas[1][1]

def emit(sock, msg: str):
    print(msg)
    if sock:
        try:
            sock.sendto(msg.encode("utf-8"), (UDP_HOST, UDP_PORT))
        except Exception as e:
            print("UDP send error:", e)

# --------------------- Main ---------------------
def main():
    print("Starting EEG Control Bridge (auto-detect N1P/N2P/N8P/N7P)")
    BoardShim.enable_dev_board_logger()

    params = BrainFlowInputParams()
    params.serial_port = SERIAL_PORT
    board = BoardShim(BOARD_ID, params)
    board.prepare_session()
    board.start_stream(45000)

    fs = BoardShim.get_sampling_rate(BOARD_ID)
    eeg_rows = BoardShim.get_eeg_channels(BOARD_ID)
    n_eeg = len(eeg_rows)
    print(f"Sampling rate: {fs} Hz | EEG rows (BrainFlow IDs): {eeg_rows} | n_eeg={n_eeg}")

    bp = design_bandpass(fs, *BANDPASS) if BANDPASS else None
    nf = design_notch(fs, NOTCH_HZ) if NOTCH_HZ else None

    sock = None
    if ENABLE_UDP:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"UDP -> {UDP_HOST}:{UDP_PORT}")

    # ----- Resolve blink reduced indices -----
    blink_pair: Optional[Tuple[int, int]] = None
    if ENABLE_BLINK:
        rA = cyton_reduced_index_from_label("N1P", n_eeg)
        rB = cyton_reduced_index_from_label("N2P", n_eeg)
        if rA is not None and rB is not None:
            blink_pair = (rA, rB)
            print(f"Blink reduced rows from mapping: N1P->{rA}, N2P->{rB}")

    # ----- Resolve alpha reduced indices -----
    alpha_pair: Optional[Tuple[int, int]] = None
    if ENABLE_ALPHA:
        rC = cyton_reduced_index_from_label("N8P", n_eeg)
        rD = cyton_reduced_index_from_label("N7P", n_eeg)
        if rC is not None and rD is not None:
            alpha_pair = (rC, rD)
            print(f"Alpha reduced rows from mapping: N8P->{rC}, N7P->{rD}")

    # -------- calibration capture (reduced space) --------
    def capture_reduced(duration_s: float) -> np.ndarray:
        nsamp = int(duration_s * fs)
        acc = np.zeros((n_eeg, 0))
        t_end = time.time() + duration_s
        while time.time() < t_end:
            chunk = board.get_current_board_data(int(fs * 0.2))
            if chunk.size:
                reduced = chunk[eeg_rows, :]   # shape (n_eeg x samples)
                acc = np.concatenate([acc, reduced], axis=1)
            else:
                time.sleep(0.01)
        if acc.shape[1] > nsamp:
            acc = acc[:, -nsamp:]
        return acc

    print("\nCALIBRATION — Phase 1 (10s): Eyes OPEN and blink naturally.")
    open_with_blinks = capture_reduced(10.0)

    if ENABLE_BLINK and blink_pair is None:
        a, b = autodetect_blink_pair(open_with_blinks, fs, bp, nf)
        blink_pair = (a, b)
        print(f"Auto-detected blink reduced rows (likely N1P/N2P): {blink_pair}")

    closed_alpha = None
    if ENABLE_ALPHA:
        print("CALIBRATION — Phase 2 (10s): Eyes CLOSED (relax).")
        closed_alpha = capture_reduced(10.0)

        print("CALIBRATION — Phase 3 (5s): Eyes OPEN.")
        open_alpha = capture_reduced(5.0)

        if alpha_pair is None:
            c, d = autodetect_alpha_pair(open_alpha, closed_alpha, fs, bp, nf)
            alpha_pair = (c, d)
            print(f"Auto-detected alpha reduced rows (likely N8P/N7P): {alpha_pair}")

    # thresholds
    blink_thr = None
    alpha_thr = None

    if ENABLE_BLINK and blink_pair is not None:
        a, b = blink_pair
        sig = open_with_blinks[a, :] - open_with_blinks[b, :]
        sigf = apply_filters(sig, fs, bp, nf) if sig.size > 100 else sig
        med = float(np.median(np.abs(sigf)))
        std = float(np.std(np.abs(sigf)))
        blink_thr = med + BLINK_STD_FACTOR * std
        print(f"Blink threshold set: {blink_thr:.3f} (median={med:.3f}, std={std:.3f}, factor={BLINK_STD_FACTOR})")

    if ENABLE_ALPHA and alpha_pair is not None and closed_alpha is not None:
        c, d = alpha_pair
        oc_closed = 0.5 * (closed_alpha[c, :] + closed_alpha[d, :])
        oc_open   = 0.5 * (open_alpha[c,   :] + open_alpha[d,   :])
        oc_closed_f = apply_filters(oc_closed, fs, bp, nf) if oc_closed.size > 100 else oc_closed
        oc_open_f   = apply_filters(oc_open,   fs, bp, nf) if oc_open.size   > 100 else oc_open
        p_closed = bandpower(oc_closed_f, fs, *ALPHA_BAND)
        p_open   = bandpower(oc_open_f,   fs, *ALPHA_BAND)
        alpha_thr = 0.7 * p_open + 0.3 * p_closed
        print(f"Alpha threshold set: {alpha_thr:.3f} (open={p_open:.3f}, closed={p_closed:.3f})")

    print("\nCalibration done. Streaming control…  Ctrl+C to stop.\n")

    last_blink_ts = 0.0
    eyes_closed_state = False

    try:
        while True:
            chunk = board.get_current_board_data(int(fs * 0.1))
            if not chunk.size:
                time.sleep(0.01)
                continue

            s = chunk[eeg_rows, :]  # reduced view

            if ENABLE_BLINK and blink_pair is not None:
                a, b = blink_pair
                bx = s[a, :] - s[b, :]
                bx = apply_filters(bx, fs, bp, nf) if bx.size > 30 else bx
                peak = float(np.max(np.abs(bx)))
                thr = blink_thr or (np.median(np.abs(bx)) + BLINK_STD_FACTOR * np.std(np.abs(bx)))
                now = time.time()
                if peak > thr and (now - last_blink_ts) > BLINK_DEBOUNCE_S:
                    last_blink_ts = now
                    emit(sock, "EVENT:BLINK")

            if ENABLE_ALPHA and alpha_pair is not None:
                c, d = alpha_pair
                oc = 0.5 * (s[c, :] + s[d, :])
                oc = apply_filters(oc, fs, bp, nf) if oc.size > 100 else oc
                p = bandpower(oc, fs, *ALPHA_BAND)
                thr = alpha_thr or p
                if (not eyes_closed_state) and p > thr * ALPHA_UP_HYST:
                    eyes_closed_state = True
                    emit(sock, "EVENT:EYES_CLOSED")
                elif eyes_closed_state and p < thr * ALPHA_DOWN_HYST:
                    eyes_closed_state = False
                    emit(sock, "EVENT:EYES_OPEN")

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("Stopping…")
    finally:
        try:
            board.stop_stream()
        except Exception:
            pass
        board.release_session()
        if sock:
            sock.close()
        print("Session closed.")

if __name__ == "__main__":
    main()
