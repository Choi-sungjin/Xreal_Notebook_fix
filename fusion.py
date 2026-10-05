"""Camera <-> glasses-IMU calibration and complementary orientation fusion.

Frames
  C  webcam (fixed to the laptop / vehicle)
  H  head, as seen by the face landmarker
  I  glasses IMU body frame
R_CH comes from the camera, gyro measures body rate in I, and R_HI is the fixed
mounting rotation, so  R_CI = R_CH @ R_HI  and  w_H = R_HI @ w_I.

The gyro also sees the vehicle turning while the camera does not; that slow
difference is absorbed by the gyro-bias integrator, which is what we want for
head-relative-to-seat tracking. The accelerometer is not used as a tilt
reference because vehicle acceleration and braking contaminate it.
"""
import json
from dataclasses import asdict, dataclass

import numpy as np

import so3


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------
@dataclass
class Calibration:
    R_HI: list
    tau: float            # add to camera timestamps to land on the IMU timeline
    gyro_scale: float     # fitted |w_cam| / |w_imu| (≈1 -> gyro already in rad/s)
    corr: float           # speed-magnitude correlation at best tau
    residual: float       # relative RMS of w_H - s*R*w_I (camera differentiation noise dominates)
    diversity: float      # smallest/largest singular value of rotation-rate samples
    stability_deg: float  # 95th pct rotation change of R_HI over block-bootstrap resamples
    n_samples: int
    bias0: list           # gyro mean during the still phase
    passed: bool
    R_home: list = None   # camera head pose while looking straight ahead (still phase)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def load(path):
        with open(path, encoding="utf-8") as f:
            return Calibration(**json.load(f))

    @property
    def R(self):
        return np.array(self.R_HI)


def _camera_rates(cam_t, cam_R, half=5, max_gap=0.05):
    """Head body rate (in H) from poses `half` frames apart on each side.

    Differentiating ~30 Hz landmark poses over adjacent frames is noise-dominated;
    a ~330 ms span keeps head motion while cutting the noise well down."""
    ts, ws, spans = [], [], []
    for k in range(half, len(cam_t) - half):
        ta, tb = cam_t[k - half], cam_t[k + half]
        if tb - ta > 2 * half * max_gap:
            continue
        w = so3.log(cam_R[k - half].T @ cam_R[k + half]) / (tb - ta)
        ts.append(cam_t[k])
        ws.append(w)
        spans.append((ta, tb))
    return np.array(ts), np.array(ws), np.array(spans)


def _window_means(imu_t, cs, spans, tau):
    i0 = np.searchsorted(imu_t, spans[:, 0] + tau)
    i1 = np.searchsorted(imu_t, spans[:, 1] + tau)
    n = i1 - i0
    ok = n > 5
    out = np.full((len(spans), 3), np.nan)
    out[ok] = (cs[i1[ok]] - cs[i0[ok]]) / n[ok, None]
    return out


def _kabsch(A, B):
    """Rotation R minimizing sum |B - R A|^2 (rows are vectors)."""
    U, _, Vt = np.linalg.svd(A.T @ B)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


def solve_calibration(cam_t, cam_R, imu_t, imu_g, still_until=None,
                      tau_range=(-0.40, 0.15), tau_step=0.005, min_rate=0.15, seed=0):
    cam_t = np.asarray(cam_t)
    bias0 = np.zeros(3)
    if still_until is not None:
        m = imu_t < still_until
        if m.sum() > 200:
            bias0 = np.median(imu_g[m], axis=0)
    g = imu_g - bias0
    cs = np.vstack([np.zeros(3), np.cumsum(g, axis=0)])

    t_mid, w_cam, spans = _camera_rates(cam_t, cam_R)
    if len(t_mid) < 50:
        raise RuntimeError(f"얼굴 자세 샘플이 부족해요 ({len(t_mid)}개)")

    best = (-2.0, 0.0)
    cam_speed = np.linalg.norm(w_cam, axis=1)
    for tau in np.arange(tau_range[0], tau_range[1] + 1e-9, tau_step):
        w_imu = _window_means(imu_t, cs, spans, tau)
        ok = np.isfinite(w_imu[:, 0])
        if ok.sum() < 50:
            continue
        c = np.corrcoef(cam_speed[ok], np.linalg.norm(w_imu[ok], axis=1))[0, 1]
        if c > best[0]:
            best = (c, tau)
    corr, tau = best

    w_imu = _window_means(imu_t, cs, spans, tau)
    ok = np.isfinite(w_imu[:, 0]) & (np.linalg.norm(w_imu, axis=1) > min_rate)
    A, B = w_imu[ok], w_cam[ok]            # want  B ≈ s * R @ A
    if len(A) < 30:
        raise RuntimeError(f"움직임 샘플이 부족해요 ({len(A)}개) — 고개를 더 크게 움직여 주세요")

    R = _kabsch(A, B)
    RA = A @ R.T
    s = float(np.sum(B * RA) / np.sum(RA * RA))
    residual = float(np.sqrt(np.mean(np.sum((B - s * RA) ** 2, axis=1)))
                     / np.sqrt(np.mean(np.sum(B ** 2, axis=1))))
    sv = np.linalg.svd(B, compute_uv=False)
    diversity = float(sv[-1] / sv[0])

    # block bootstrap (neighbouring samples are correlated) -> how much R_HI could move
    rng = np.random.default_rng(seed)
    block = 10
    nb = int(np.ceil(len(A) / block))
    devs = []
    for _ in range(200):
        pick = rng.integers(0, nb, nb)
        idx = np.concatenate([np.arange(b * block, min((b + 1) * block, len(A))) for b in pick])
        devs.append(so3.angle_deg(_kabsch(A[idx], B[idx]), R))
    stability = float(np.percentile(devs, 95))

    R_home = None
    if still_until is not None:
        home = [Rk for tk, Rk in zip(cam_t, cam_R) if tk < still_until]
        if len(home) >= 10:
            R_home = so3.orthonormalize(np.mean(home, axis=0)).tolist()

    passed = (corr > 0.7 and residual < 0.45 and diversity > 0.15
              and 0.6 < s < 1.6 and stability < 5.0)
    return Calibration(R_HI=R.tolist(), tau=float(tau), gyro_scale=s, corr=float(corr),
                       residual=residual, diversity=diversity, stability_deg=stability,
                       n_samples=int(len(A)), bias0=bias0.tolist(), passed=bool(passed),
                       R_home=R_home)


# ---------------------------------------------------------------------------
# fusion
# ---------------------------------------------------------------------------
def propagate(R, t0, t1, imu, bias, scale=1.0):
    """Integrate gyro from t0 to t1 (zero-order hold) starting at R."""
    if R is None or t1 <= t0:
        return R
    t, g, _ = imu.window(t0, t1)
    if len(t) == 0:
        return R
    dts = np.diff(np.concatenate([[t0], t, [t1]]))
    G = np.vstack([g, g[-1:]]) * scale - bias
    return so3.integrate(R, G * dts[:, None])


class ArrayImu:
    """Recorded IMU arrays with the same window() interface as XrealImu (for replay)."""

    def __init__(self, t, g):
        self.t, self.g = np.asarray(t), np.asarray(g)

    def window(self, t0, t1):
        i0 = np.searchsorted(self.t, t0, "left")
        i1 = np.searchsorted(self.t, t1, "right")
        return self.t[i0:i1], self.g[i0:i1], None

    def latest_time(self):
        return self.t[-1]


class ComplementaryFusion:
    """Gyro-propagated orientation corrected by delayed camera head pose (Mahony-style PI).

    Webcam head pose degrades while the head turns fast (motion blur, landmark
    smoothing) and any residual camera<->IMU misalignment shows up in proportion to
    the rotation, so the camera correction is softened with head speed and the
    gyro-bias integrator only learns while the head is nearly still.

    The webcam is most accurate when the user looks straight at it and holds still,
    so in that "home" pose the correction is made strong (and large drift is snapped
    away at once). Whatever drift built up during big or fast motion is then cleared
    the moment the user faces forward again."""

    def __init__(self, calib, kp=0.06, ki=0.3, gate_deg=20.0, rate_ref=0.6, bias_gate=0.25,
                 home_deg=15.0, home_kp=0.25, home_rate=0.17, home_frames=5):
        self.R_HI = calib.R
        self.tau = calib.tau
        # Only undo a unit mismatch (deg/s); small scale deviations are camera smoothing.
        self.scale = 1.0 if 0.6 < calib.gyro_scale < 1.6 else calib.gyro_scale
        self.kp, self.ki = kp, ki
        self.gate = np.radians(gate_deg)
        self.rate_ref = rate_ref      # rad/s at which the camera gain is halved
        self.bias_gate = bias_gate    # rad/s; learn bias only below this head speed
        self.R_home = np.array(calib.R_home) if calib.R_home is not None else None
        self.home_deg = home_deg      # within this angle of R_home counts as facing forward
        self.home_kp = home_kp        # camera gain while facing forward and still
        self.home_rate = home_rate    # rad/s; "still" threshold for the home pose
        self.home_frames = home_frames
        self.bias = np.array(calib.bias0, dtype=float)
        self.R = None     # R_CI at self.t (IMU timeline)
        self.t = None
        self.rate = 0.0
        self.outliers = 0
        self.resets = 0
        self.home = False             # currently in the strong forward-facing correction
        self.recenters = 0            # large drifts snapped away while facing forward
        self._home_count = 0

    def set_home(self, R_CH):
        self.R_home = np.array(R_CH)

    @property
    def ready(self):
        return self.R is not None

    def predict(self, t, imu):
        """R_CI at time t without changing state."""
        return propagate(self.R, self.t, t, imu, self.bias, self.scale)

    def head(self, R_CI):
        return None if R_CI is None else R_CI @ self.R_HI.T

    def advance(self, t, imu, keep=0.25):
        """Without camera updates, move the state forward so outputs stay cheap."""
        if self.ready and t - self.t > 2 * keep:
            self.R = self.predict(t - keep, imu)
            self.t = t - keep

    def _head_speed(self, t0, t1, imu):
        _, g, _ = imu.window(t0, t1)
        if len(g) == 0:
            return self.rate
        return float(np.mean(np.linalg.norm(g * self.scale - self.bias, axis=1)))

    def update(self, t_cam, R_CH_meas, imu, apply=True):
        """Fuse a camera pose captured at host time t_cam.

        Returns (prediction error in deg before the update, accepted flag)."""
        t_m = t_cam + self.tau
        R_meas = R_CH_meas @ self.R_HI
        if not self.ready:
            self.R, self.t = R_meas, t_m
            return 0.0, True
        if t_m <= self.t:
            return None, False
        R_pred = self.predict(t_m, imu)
        self.rate = self._head_speed(self.t, t_m, imu)
        e = so3.log(R_pred.T @ R_meas)
        err = float(np.degrees(np.linalg.norm(e)))
        if not apply:
            self.R, self.t = R_pred, t_m
            self.home, self._home_count = False, 0
            return err, False
        facing = (self.R_home is not None and self.rate < self.home_rate
                  and so3.angle_deg(self.R_home, R_CH_meas) < self.home_deg)
        self._home_count = self._home_count + 1 if facing else 0
        self.home = self._home_count >= self.home_frames
        if self.home and np.linalg.norm(e) > self.gate:
            self.R, self.t = R_meas, t_m    # facing forward and still: trust the camera now
            self.outliers = 0
            self.recenters += 1
            return err, True
        if not self.home and np.linalg.norm(e) > self.gate:
            self.outliers += 1
            if self.outliers >= 8:      # camera consistently disagrees -> trust it
                self.R, self.t = R_meas, t_m
                self.outliers = 0
                self.resets += 1
            else:
                self.R, self.t = R_pred, t_m
            return err, False
        self.outliers = 0
        dt = t_m - self.t
        kp = self.home_kp if self.home else self.kp / (1.0 + (self.rate / self.rate_ref) ** 2)
        self.R = R_pred @ so3.exp(kp * e)
        if self.rate < self.bias_gate:
            self.bias = np.clip(self.bias - self.ki * e * dt, -0.2, 0.2)
        self.t = t_m
        return err, True
