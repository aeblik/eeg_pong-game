# eeg_blink_to_udp_threshold100.py
# Blink detector using Cyton CH1 & CH2 (forehead), based on:
#   - baseline of recent |signal|
#   - blink when |sample| - baseline_mean > THRESH_UV and |sample| > MIN_ABS_UV
# Sends UDP "EVENT:BLINK" to Pong.

import time
import numpy as np
from collections import deque
import socket

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from scipy.signal import butter, filtfilt, iirnotch

# ---------- USER SETTINGS ----------
SERIAL_PORT = "COM3"          # e.g. "COM3", "/dev/ttyUSB0", "/dev/tty.usbserial-XXXX"
BOARD_ID = BoardIds.CYTON_BOARD.value

CYTON_BLINK_CH1 = 1           # forehead electrode on Cyton CH1
CYTON_BLINK_CH2 = 2           # forehead electrode on Cyton CH2

# Filtering (light)
NOTCH_HZ   = 50               # 50 in CH/EU, 60 in US, or None
BANDPASS   = (1.0, 15.0)      # Hz; low-ish to emphasize blinks

# Baseline + threshold (OpenBCI-style)
BASELINE_WIN_SECS = 1       # how much history for baseline (seconds)
CHUNK_SECS        = 0.05      # processing every 50 ms
THRESH_UV         = 100.0     # must exceed baseline by this much (µV)
MIN_ABS_UV        = 100.0     # absolute value also must exceed this (µV)
BLINK_DEBOUNCE_S  = 0.35      # min time between blinks

PRINT_DEBUG = True            # show val / base_mean / diff / thr

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
    print("Blink detector (100 µV above baseline, CH1 & CH2) → UDP")

    BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()
    params.serial_port = SERIAL_PORT
    board = BoardShim(BOARD_ID, params)
    board.prepare_session()
    board.start_stream(45000)

    fs = BoardShim.get_sampling_rate(BOARD_ID)
    eeg_rows = BoardShim.get_eeg_channels(BOARD_ID)  # eg [1,2,3,4,5,6,7,8]
    n_eeg = len(eeg_rows)
    print(f"fs={fs} Hz, EEG rows={eeg_rows} (n={n_eeg})")

    idx_a = CYTON_BLINK_CH1 - 1
    idx_b = CYTON_BLINK_CH2 - 1
    if not (0 <= idx_a < n_eeg and 0 <= idx_b < n_eeg):
        raise ValueError("Blink channels must be between 1 and n_eeg")

    print(f"Using forehead pair (reduced indices): {idx_a}, {idx_b}")

    bp = design_bandpass(fs, *BANDPASS) if BANDPASS else None
    nf = design_notch(fs, NOTCH_HZ) if NOTCH_HZ else None

    # Baseline buffer: absolute value of combined forehead signal
    baseline_len = int(BASELINE_WIN_SECS * fs)
    baseline_buf = deque(maxlen=baseline_len)

    # UDP
    sock = None
    if ENABLE_UDP:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"UDP enabled -> {UDP_HOST}:{UDP_PORT}")

    # Warmup baseline
    print(f"\nBaseline warm-up for ~{BASELINE_WIN_SECS*2:.0f} s (eyes open, relaxed, normal blinks ok)...")
    warmup_end = time.time() + BASELINE_WIN_SECS*2
    while time.time() < warmup_end:
        chunk = board.get_current_board_data(int(fs * CHUNK_SECS))
        if not chunk.size:
            time.sleep(0.01)
            continue
        s = chunk[eeg_rows, :]
        # combine CH1 & CH2 like SARA (sum/front average)
        bx = 0.5 * (s[idx_a, :] + s[idx_b, :])
        bx = apply_filters(bx, fs, bp, nf) if bx.size > 30 else bx
        for v in np.abs(bx):
            baseline_buf.append(float(v))
        time.sleep(0.01)

    if len(baseline_buf) < 10:
        print("Not enough data in baseline buffer; check signal.")
        board.stop_stream(); board.release_session()
        if sock: sock.close()
        return

    print("Baseline warm-up done. Starting detection… Ctrl+C to stop.\n")

    last_blink_ts = 0.0

    try:
        while True:
            chunk = board.get_current_board_data(int(fs * CHUNK_SECS))
            if not chunk.size:
                time.sleep(0.01)
                continue

            s = chunk[eeg_rows, :]
            bx = 0.5 * (s[idx_a, :] + s[idx_b, :])
            bx = apply_filters(bx, fs, bp, nf) if bx.size > 30 else bx

            now = time.time()

            for v in bx:
                val = abs(float(v))
                baseline_buf.append(val)
                if len(baseline_buf) < baseline_len:
                    continue  # still filling baseline

                base_arr = np.array(baseline_buf)
                base_mean = float(np.mean(base_arr))
                diff = val - base_mean

                # OpenBCI-style condition:
                #  - diff above threshold
                #  - absolute value also above min
                over = (diff > THRESH_UV) and (val > MIN_ABS_UV)

                if PRINT_DEBUG:
                    print(
                        f"val={val:.1f}µV, base={base_mean:.1f}, diff={diff:.1f}, "
                        f"thr={THRESH_UV:.1f}, dt={now - last_blink_ts:.3f}s",
                        end="\r",
                        flush=True
                    )

                # HARD COOLDOWN: never trigger more often than BLINK_DEBOUNCE_S
                if over and (now - last_blink_ts) >= BLINK_DEBOUNCE_S:
                    last_blink_ts = now
                    print(f"\nEVENT:BLINK  val={val:.1f}, base={base_mean:.1f}, diff={diff:.1f}")
                    if sock:
                        try:
                            sock.sendto(UDP_MESSAGE.encode("utf-8"), (UDP_HOST, UDP_PORT))
                        except Exception as e:
                            print("UDP send error:", e)

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
