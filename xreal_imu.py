"""XREAL One Pro IMU reader over the glasses' USB network link.

The glasses enumerate a USB-NCM network adapter (host 169.254.2.10) and stream
IMU packets on TCP 169.254.2.1:52998. Layout below was worked out from the live
stream; it is not an official XREAL API and may change with firmware.

  134-byte packet, little-endian
    0   6B   header 28 36 00 00 00 80
    14  u64  timestamp, ns (glasses clock)
    30  u32  type: 11 = gyro+accel (1000 Hz), 4 = magnetometer (400 Hz)
    34  3xf32 gyro  [rad/s]   (type 11)
    46  3xf32 accel [m/s^2]   (type 11)
    58  3xf32 mag   [uT]      (type 4; type 11 fills -3200)
    70  f32  temperature [C]
"""
import socket
import struct
import threading
import time

import numpy as np

HOST, PORT = "169.254.2.1", 52998
HEADER = bytes([0x28, 0x36, 0x00, 0x00, 0x00, 0x80])
PKT_LEN = 134
TYPE_IMU, TYPE_MAG = 11, 4

_CLOCK_CREEP = 2e-5  # lets the host<-device offset follow ~20 ppm clock drift
_WARMUP_S = 1.0      # drop the backlog the glasses flush right after connect


class XrealImu:
    def __init__(self, host=HOST, port=PORT, capacity_s=180):
        self.host, self.port = host, port
        cap = int(capacity_s * 1000)
        self._cap = cap
        self._T = np.zeros(2 * cap)
        self._G = np.zeros((2 * cap, 3))
        self._A = np.zeros((2 * cap, 3))
        self._pos = 0
        self._lock = threading.Lock()
        self._offset = None
        self._last_recv = None
        self._running = False
        self._thread = None
        self.error = None
        self.n_packets = 0
        self.n_rejected = 0
        self.temperature = float("nan")

    # ---- lifecycle -------------------------------------------------------
    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def wait_ready(self, timeout=5.0):
        t_end = time.perf_counter() + timeout
        while time.perf_counter() < t_end:
            if self.error:
                raise RuntimeError(f"IMU 연결 실패: {self.error}")
            with self._lock:
                if self._pos > 50:
                    return
            time.sleep(0.02)
        raise RuntimeError("IMU 데이터가 들어오지 않아요 (글래스 USB 연결 확인)")

    # ---- queries -----------------------------------------------------------
    def window(self, t0, t1):
        """Samples with host time in [t0, t1]: (t, gyro, accel) copies."""
        with self._lock:
            T = self._T[: self._pos]
            i0 = np.searchsorted(T, t0, "left")
            i1 = np.searchsorted(T, t1, "right")
            return T[i0:i1].copy(), self._G[i0:i1].copy(), self._A[i0:i1].copy()

    def latest_time(self):
        with self._lock:
            return self._T[self._pos - 1] if self._pos else None

    # ---- internals ---------------------------------------------------------
    def _append(self, rows):
        with self._lock:
            for t, g, a in rows:
                if self._pos and t <= self._T[self._pos - 1]:
                    t = self._T[self._pos - 1] + 1e-6
                if self._pos == 2 * self._cap:
                    c = self._cap
                    self._T[:c], self._G[:c], self._A[:c] = self._T[c:], self._G[c:], self._A[c:]
                    self._pos = c
                p = self._pos
                self._T[p], self._G[p], self._A[p] = t, g, a
                self._pos += 1

    def _run(self):
        try:
            sock = socket.create_connection((self.host, self.port), timeout=3)
        except OSError as e:
            self.error = str(e)
            return
        sock.settimeout(1.0)
        t_connect = time.perf_counter()
        buf = b""
        try:
            while self._running:
                try:
                    chunk = sock.recv(65536)
                except socket.timeout:
                    continue
                if not chunk:
                    self.error = "연결이 끊겼어요"
                    break
                t_recv = time.perf_counter()
                buf, pkts = self._parse(buf + chunk)
                rows = []
                for ts_ns, g, a in pkts:
                    t_dev = ts_ns * 1e-9
                    cand = t_recv - t_dev
                    if self._offset is None:
                        self._offset = cand
                    else:
                        creep = _CLOCK_CREEP * (t_recv - self._last_recv)
                        self._offset = min(self._offset + creep, cand)
                    self._last_recv = t_recv
                    if t_recv - t_connect < _WARMUP_S:
                        continue
                    rows.append((t_dev + self._offset, g, a))
                if rows:
                    self._append(rows)
        finally:
            sock.close()

    def _parse(self, buf):
        out, i, n = [], 0, len(buf)
        while True:
            j = buf.find(HEADER, i)
            if j < 0:
                return buf[max(i, n - len(HEADER) + 1):], out
            if j + PKT_LEN > n:
                return buf[j:], out
            typ = struct.unpack_from("<I", buf, j + 30)[0]
            if typ == TYPE_IMU:
                ts = struct.unpack_from("<Q", buf, j + 14)[0]
                g = struct.unpack_from("<3f", buf, j + 34)
                a = struct.unpack_from("<3f", buf, j + 46)
                self.n_packets += 1
                amag = (a[0] ** 2 + a[1] ** 2 + a[2] ** 2) ** 0.5
                gmag = (g[0] ** 2 + g[1] ** 2 + g[2] ** 2) ** 0.5
                if np.isfinite(amag) and np.isfinite(gmag) and 4.0 < amag < 16.0 and gmag < 35.0:
                    out.append((ts, g, a))
                else:
                    self.n_rejected += 1
                self.temperature = struct.unpack_from("<f", buf, j + 70)[0]
                i = j + PKT_LEN
            elif typ == TYPE_MAG:
                i = j + PKT_LEN
            else:
                i = j + 1
