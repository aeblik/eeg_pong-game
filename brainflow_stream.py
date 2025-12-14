# live_plot_brainflow.py
# Real-time multi-channel viewer for OpenBCI Cyton using BrainFlow + PyQtGraph

import sys, time
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from scipy.signal import butter, filtfilt, iirnotch
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg

# ====== CONNECTION ======
SERIAL_PORT = "COM3"
BOARD_ID = BoardIds.CYTON_BOARD.value
WINDOW_SECS = 10                # seconds of data shown in the scrolling window
REFRESH_MS = 33                 # UI refresh rate (~30 FPS)
BANDPASS = (1.0, 40.0)          # Hz; set to None to disable
NOTCH_FREQ = 50                 # 50 for EU/CH; set 60 in US; set None to disable
STACK_GAP_U = 100.0             # vertical gap between stacked channels (in raw units)
# ===========================

def design_bandpass(fs, low, high, order=4):
    ny = 0.5 * fs
    b, a = butter(order, [low/ny, high/ny], btype='band')
    return b, a

def design_notch(fs, f0, q=30.0):
    ny = 0.5 * fs
    w0 = f0 / ny
    b, a = iirnotch(w0, q)
    return b, a

class LivePlot(QtWidgets.QMainWindow):
    def __init__(self, board, board_id, fs, eeg_chs):
        super().__init__()
        self.board = board
        self.board_id = board_id
        self.fs = fs
        self.eeg_chs = eeg_chs
        self.nch = len(eeg_chs)
        self.nsamp = int(WINDOW_SECS * fs)

        # ring buffer (channels x samples)
        self.buf = np.zeros((self.nch, self.nsamp), dtype=np.float64)

        # filters
        self.bp = None
        self.nf = None
        if BANDPASS:
            self.bp = design_bandpass(fs, BANDPASS[0], BANDPASS[1])
        if NOTCH_FREQ:
            self.nf = design_notch(fs, NOTCH_FREQ)

        # ---- UI ----
        self.setWindowTitle("OpenBCI Cyton — Live Plot (BrainFlow + PyQtGraph)")
        cw = QtWidgets.QWidget()
        self.setCentralWidget(cw)
        layout = QtWidgets.QVBoxLayout(cw)

        self.plot = pg.PlotWidget()
        self.plot.setBackground("k")
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.plot.setMenuEnabled(False)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.setYRange(-STACK_GAP_U, self.nch * STACK_GAP_U)
        self.plot.setLabel("bottom", "Time", units="samples (scrolling)")

        self.curves = []
        for ch in range(self.nch):
            c = self.plot.plot(pen=pg.mkPen(width=1))
            self.curves.append(c)

        layout.addWidget(self.plot)

        # status
        self.status = QtWidgets.QLabel("Streaming…  Press Q to quit.")
        layout.addWidget(self.status)

        # timer to refresh view
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(REFRESH_MS)

        # key handler
        cw.setFocusPolicy(QtCore.Qt.StrongFocus)
        cw.keyPressEvent = self.keyPressEvent

    def keyPressEvent(self, event):
        k = event.key()
        if k in (QtCore.Qt.Key_Q, QtCore.Qt.Key_Escape):
            self.close()

    def closeEvent(self, ev):
        # stop stream + release
        try:
            self.board.stop_stream()
        except Exception:
            pass
        try:
            self.board.release_session()
        except Exception:
            pass
        ev.accept()

    def _apply_filters(self, x):
        # x: (nch, n)
        if self.nf:
            b, a = self.nf
            x = filtfilt(b, a, x, axis=1, padlen=50)
        if self.bp:
            b, a = self.bp
            x = filtfilt(b, a, x, axis=1, padlen=50)
        return x

    def update_plot(self):
        # pull recent data; BrainFlow returns (rows=channels, cols=samples)
        data = self.board.get_current_board_data(num_samples=int(self.fs * 0.1))  # ~100 ms chunk
        if data.size == 0:
            return

        # select only EEG rows in the order of eeg_chs
        eeg_rows = data[self.eeg_chs, :]
        if eeg_rows.ndim == 1:  # single sample edge-case
            eeg_rows = eeg_rows[:, None]

        # append into ring buffer
        n = eeg_rows.shape[1]
        if n >= self.nsamp:
            self.buf = eeg_rows[:, -self.nsamp:]
        else:
            self.buf = np.roll(self.buf, -n, axis=1)
            self.buf[:, -n:] = eeg_rows

        # filtering (copy to avoid phase issues accumulating in buffer)
        show = self.buf.copy()
        try:
            show = self._apply_filters(show)
        except Exception:
            # If filtering fails (e.g. tiny buffer), just show raw
            pass

        # vertical stacking and draw
        for i in range(self.nch):
            y = show[i, :] + i * STACK_GAP_U
            x = np.arange(y.size)
            self.curves[i].setData(x, y)

        self.status.setText(f"Streaming | fs={self.fs} Hz | chans={self.nch} | notch={NOTCH_FREQ or 'off'} | bp={BANDPASS or 'off'}")

def main():
    # BrainFlow setup
    BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()
    params.serial_port = SERIAL_PORT
    board = BoardShim(BOARD_ID, params)
    board.prepare_session()
    # generous buffer; BrainFlow default is fine; set custom if needed:
    board.start_stream(45000)  # ~45k samples circular buffer on native side

    fs = BoardShim.get_sampling_rate(BOARD_ID)
    eeg_chs = BoardShim.get_eeg_channels(BOARD_ID)

    # GUI
    app = QtWidgets.QApplication(sys.argv)
    w = LivePlot(board, BOARD_ID, fs, eeg_chs)
    w.resize(1200, 700)
    w.show()
    rc = app.exec_()

    # safety
    try:
        board.stop_stream()
    except Exception:
        pass
    try:
        board.release_session()
    except Exception:
        pass
    sys.exit(rc)

if __name__ == "__main__":
    main()
