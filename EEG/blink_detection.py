# eeg_blink_to_udp.py
# Blink detector (dynamic baseline, CH1 & CH2) that sends UDP "EVENT:BLINK" to Pong.

import time
import numpy as np
from collections import deque
import socket

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from scipy.signal import butter, filtfilt, iirnotch

# ---------- USER SETTINGS ----------
SERIAL_PORT = "COM3"          # e.g. "COM3", "/dev/ttyUSB0", "/dev/tty.usbserial-XXXX"
BOARD_ID = BoardIds.CYTON_BOARD.value

CYTON_BLINK_CH1 = 1           # forehead channel 1 (Cyton input 1)
CYTON_BLINK_CH2 = 2           # forehead channel 2 (Cyton input 2)

# Filtering
NOTCH_HZ   = 50               # 50 in CH/EU, 60 in US, or None to disable
BANDPASS   = (1.0, 15.0)      # Hz; emphasise blink deflections

# Dynamic baseline config
BASELINE_SECS   = 10.0        # seconds of history for baseline
CHUNK_SECS      = 0.1         # chunk duration (~100 ms)
BLINK_STD_FACTOR = 1.75       # your tuned value (worked well for you), increase to reduce sensitivity
BLINK_DEBOUNCE_S = 0.5        # reduce double-triggers (can tweak)

PRINT_DEBUG = True            # show peak / thr each loop

# UDP to Pong
ENABLE_UDP = True
UDP_HOST   = "127.0.0.1"
UDP_PORT   = 8765
UDP_MESSAGE = "EVENT:BLINK"
# -----------------------------------

def design_bandpass(fs, low, high, order=4):
    ny = 0.5 * fs
    b, a = butter(order, [low/ny, high/ny], btype="band")
    return b, a

def design_notch(fs, f0, q=30.0):
    ny = 0.5 * fs
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

def main():
    print("Blink detector → UDP (dynamic baseline, CH1 & CH2)")

    BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()
    params.serial_port = SERIAL_PORT
    board = BoardShim(BOARD_ID, params)
    board.prepare_session()
    board.start_stream(45000)

    fs = BoardShim.get_sampling_rate(BOARD_ID)
    eeg_rows = BoardShim.get_eeg_channels(BOARD_ID)  # e.g. [1,2,3,4,5,6,7,8]
    n_eeg = len(eeg_rows)
    print(f"fs={fs} Hz, EEG rows={eeg_rows} (n={n_eeg})")

    idx_a = CYTON_BLINK_CH1 - 1
    idx_b = CYTON_BLINK_CH2 - 1
    if not (0 <= idx_a < n_eeg and 0 <= idx_b < n_eeg):
        raise ValueError("Blink channels must be between 1 and n_eeg")

    print(f"Using blink pair (reduced indices): {idx_a}, {idx_b}")

    bp = design_bandpass(fs, *BANDPASS) if BANDPASS else None
    nf = design_notch(fs, NOTCH_HZ) if NOTCH_HZ else None

    baseline_len = int(BASELINE_SECS * fs)
    baseline_buf = deque(maxlen=baseline_len)

    last_blink_ts = 0.0
    prev_over_thr = False  # for edge detection

    # UDP setup
    sock = None
    if ENABLE_UDP:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"UDP enabled -> {UDP_HOST}:{UDP_PORT}")

    # --- Warm-up baseline ---
    warmup_secs = BASELINE_SECS
    t_end = time.time() + warmup_secs
    print(f"\nWarm-up baseline for ~{warmup_secs:.0f}s (eyes open, relaxed; blinks okay)...")

    while time.time() < t_end:
        chunk = board.get_current_board_data(int(fs * CHUNK_SECS))
        if not chunk.size:
            time.sleep(0.01)
            continue
        s = chunk[eeg_rows, :]
        bx = s[idx_a, :] - s[idx_b, :]
        bx = apply_filters(bx, fs, bp, nf) if bx.size > 30 else bx
        for v in np.abs(bx):
            baseline_buf.append(v)
        time.sleep(0.01)

    if len(baseline_buf) < 10:
        print("Baseline buffer too small; check EEG signal.")
        board.stop_stream(); board.release_session()
        if sock: sock.close()
        return

    print("Warm-up done. Starting dynamic detection… Ctrl+C to stop.\n")

    try:
        while True:
            chunk = board.get_current_board_data(int(fs * CHUNK_SECS))
            if not chunk.size:
                time.sleep(0.01)
                continue

            s = chunk[eeg_rows, :]
            bx = s[idx_a, :] - s[idx_b, :]
            bx = apply_filters(bx, fs, bp, nf) if bx.size > 30 else bx

            abs_bx = np.abs(bx)
            for v in abs_bx:
                baseline_buf.append(v)

            baseline_arr = np.array(baseline_buf)
            base_med = float(np.median(baseline_arr))
            base_std = float(np.std(baseline_arr))
            thr = base_med + BLINK_STD_FACTOR * base_std
            peak = float(np.max(abs_bx))
            now = time.time()

            if PRINT_DEBUG:
                print(
                    f"peak={peak:.1f}, base_med={base_med:.1f}, base_std={base_std:.1f}, thr={thr:.1f}",
                    end="\r",
                    flush=True
                )

            over_thr = peak > thr

            # RISING EDGE + debounce
            if over_thr and (not prev_over_thr) and (now - last_blink_ts > BLINK_DEBOUNCE_S):
                last_blink_ts = now
                print(f"\nEVENT:BLINK  peak={peak:.1f}, thr={thr:.1f}")
                if sock:
                    try:
                        sock.sendto(UDP_MESSAGE.encode("utf-8"), (UDP_HOST, UDP_PORT))
                    except Exception as e:
                        print("UDP send error:", e)

            prev_over_thr = over_thr
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nStopping…")
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
