"""Webcam face tracking + XREAL One Pro IMU sensor fusion.

  python tracker.py imu-test   --duration 5     # glasses IMU stream check
  python tracker.py face-test  --duration 15    # webcam head-pose check
  python tracker.py calibrate                   # camera<->IMU calibration (move your head)
  python tracker.py run                         # live fusion (q: quit, z: set forward, c: recalibrate)
  python tracker.py replay rec.npz              # offline re-run of a `run --record` capture
  python tracker.py                             # (or double-click the exe) calibrate if needed, then run
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import so3
from fusion import ArrayImu, Calibration, ComplementaryFusion, propagate, solve_calibration
from head_pose import Camera, HeadPoseEstimator, privacy_blur
from xreal_imu import XrealImu

FROZEN = getattr(sys, "frozen", False)
# bundled read-only files (model) vs. writable per-user files (calibration)
RES_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
DATA_DIR = (Path(os.environ.get("APPDATA", Path.home())) / "XrealHeadFusion" if FROZEN
            else Path(__file__).resolve().parent)
RECALIBRATE = 3  # cmd_run return code: user pressed c
WIN = "XREAL Head Fusion"
DISP_W, DISP_H = 960, 540

GREEN, YELLOW, RED, WHITE, GRAY = (80, 220, 80), (0, 215, 255), (60, 60, 240), (255, 255, 255), (170, 170, 170)


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------
class Text:
    def __init__(self):
        try:
            self.f = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 18)
            self.fb = ImageFont.truetype("C:/Windows/Fonts/malgunbd.ttf", 30)
        except OSError:
            self.f = self.fb = ImageFont.load_default()

    def draw(self, img, items):
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)
        for text, xy, color, big in items:
            d.text(xy, text, font=self.fb if big else self.f, fill=tuple(int(c) for c in color[::-1]))
        return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def draw_axes(img, origin, R, length, thickness):
    ox, oy = int(origin[0]), int(origin[1])
    for i, col in enumerate([(60, 60, 255), (60, 255, 60), (255, 120, 40)]):  # x red, y green, z blue
        ex = int(ox + length * R[0, i])
        ey = int(oy - length * R[1, i])
        cv2.line(img, (ox, oy), (ex, ey), col, thickness, cv2.LINE_AA)


def panel(img, x0, y0, x1, y1, alpha=0.55):
    sub = img[y0:y1, x0:x1]
    img[y0:y1, x0:x1] = (sub * (1 - alpha)).astype(np.uint8)


def strip_chart(img, series, t_now, span=10.0, rng=60.0, box=(10, 420, 950, 530)):
    x0, y0, x1, y1 = box
    panel(img, x0, y0, x1, y1)
    ymid = (y0 + y1) // 2
    cv2.line(img, (x0, ymid), (x1, ymid), (90, 90, 90), 1)
    for (ts, vs), col in series:
        if len(ts) < 2:
            continue
        ts, vs = np.asarray(ts), np.asarray(vs)
        m = (ts > t_now - span) & np.isfinite(vs)
        if m.sum() < 2:
            continue
        xs = x0 + (ts[m] - (t_now - span)) / span * (x1 - x0)
        ys = ymid - np.clip(vs[m], -rng, rng) / rng * (y1 - y0) / 2
        cv2.polylines(img, [np.stack([xs, ys], 1).astype(np.int32)], False, col, 2, cv2.LINE_AA)


def fmt(ypr):
    if ypr is None:
        return "   --       --       --"
    return f"{ypr[0]:+7.1f}° {ypr[1]:+7.1f}° {ypr[2]:+7.1f}°"


def open_window():
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, DISP_W, DISP_H)
    try:
        cv2.setWindowProperty(WIN, cv2.WND_PROP_TOPMOST, 1)
    except cv2.error:
        pass


def window_closed():
    try:
        return cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def to_display(frame):
    return cv2.resize(cv2.flip(frame, 1), (DISP_W, DISP_H))


def mirror_pt(pt, w, h):
    return ((w - 1 - pt[0]) * DISP_W / w, pt[1] * DISP_H / h)


def mirror_R(R):
    """Axis vectors as they appear in the mirrored display (flip x of each column)."""
    return np.diag([-1.0, 1.0, 1.0]) @ R


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def cmd_imu_test(args):
    imu = XrealImu().start()
    imu.wait_ready()
    t0 = time.perf_counter()
    time.sleep(args.duration)
    t, g, a = imu.window(t0, t0 + args.duration)
    imu.stop()
    rate = (len(t) - 1) / (t[-1] - t[0])
    amag = np.linalg.norm(a, axis=1)
    dt = np.diff(t) * 1e3
    print("=== IMU 테스트 ===")
    print(f"샘플 {len(t)}개, 속도 {rate:.0f} Hz (dt 중앙값 {np.median(dt):.3f} ms, 최대 {dt.max():.1f} ms)")
    print(f"가속도 크기 {amag.mean():.3f} ± {amag.std():.3f} m/s²")
    print(f"자이로 평균 {np.round(g.mean(0), 4)} rad/s, 표준편차 {np.round(g.std(0), 4)}")
    print(f"버린 패킷 {imu.n_rejected}/{imu.n_packets}, 온도 {imu.temperature:.1f}°C")
    ok = 900 < rate < 1100 and 8.8 < amag.mean() < 10.8
    print("결과:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def cmd_face_test(args):
    cam = Camera(args.camera)
    est = HeadPoseEstimator(args.model)
    text = Text()
    if not args.headless:
        open_window()
    seq, n, hits, infer_ms, angles, nfaces = 0, 0, 0, [], [], []
    t_start = time.perf_counter()
    snap_done = args.snapshot is None
    try:
        while time.perf_counter() - t_start < args.duration:
            frame, t, seq = cam.wait_frame(seq)
            if frame is None:
                continue
            n += 1
            t0 = time.perf_counter()
            pose = est.process(frame, t)
            infer_ms.append((time.perf_counter() - t0) * 1e3)
            h, w = frame.shape[:2]
            base = frame
            if pose:
                hits += 1
                angles.append(so3.euler_ypr(pose.R))
                nfaces.append(pose.n_faces)
                if not snap_done and time.perf_counter() - t_start > args.duration / 2:
                    base = privacy_blur(frame, pose.bbox)
            if args.headless and snap_done:
                continue
            img = base.copy()
            if pose:
                x0, y0, x1, y1 = pose.bbox
                cv2.rectangle(img, (x0, y0), (x1, y1), GREEN, 2)
                for p in pose.landmarks[::12]:
                    cv2.circle(img, (int(p[0]), int(p[1])), 1, GREEN, -1)
            disp = to_display(img)
            if pose:
                draw_axes(disp, mirror_pt(pose.nose, w, h), mirror_R(pose.R), 90, 3)
            panel(disp, 0, 0, 420, 70)
            disp = text.draw(disp, [
                ("얼굴 추적 테스트" + ("" if pose else "  — 얼굴 없음"), (10, 6), GREEN if pose else RED, False),
                ("yaw / pitch / roll  " + fmt(angles[-1] if pose else None), (10, 36), WHITE, False),
            ])
            if not snap_done and pose and base is not frame:
                cv2.imwrite(str(args.snapshot), disp)
                snap_done = True
            if not args.headless:
                cv2.imshow(WIN, disp)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        cam.close()
        est.close()
        cv2.destroyAllWindows()
    el = time.perf_counter() - t_start
    print("=== 얼굴 추적 테스트 ===")
    print(f"처리 프레임 {n}개 ({n / el:.1f} fps), 얼굴 검출 {hits}개 ({100 * hits / max(n, 1):.0f}%)")
    print(f"추론 시간 중앙값 {np.median(infer_ms):.1f} ms, 화면 속 얼굴 수 최대 {max(nfaces) if nfaces else 0}")
    if len(angles) > 5:
        A = np.array(angles)
        jit = np.median(np.abs(np.diff(A, axis=0)), axis=0)
        print(f"평균 yaw/pitch/roll {np.round(A.mean(0), 1)}°, 범위(표준편차) {np.round(A.std(0), 1)}°")
        print(f"프레임 간 흔들림(중앙값) {np.round(jit, 2)}°")
    ok = hits / max(n, 1) > 0.8
    print("결과:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


CAL_PHASES = [
    ("가만히 정면(카메라)을 보세요", 3.0),
    ("고개를 좌우로 천천히 돌리세요 (도리도리)", 6.0),
    ("고개를 위아래로 천천히 (끄덕끄덕)", 6.0),
    ("고개를 좌우로 갸웃갸웃 (귀를 어깨 쪽으로)", 6.0),
]


def cmd_calibrate(args):
    raw_path = Path(args.calib).with_name("calib_raw.npz")
    if args.from_raw:
        d = np.load(raw_path)
        cal = solve_calibration(list(d["cam_t"]), list(d["cam_R"]), d["imu_t"], d["imu_g"],
                                still_until=float(d["still_until"]))
        return report_calibration(cal, len(d["cam_t"]), args)
    imu = XrealImu().start()
    cam = Camera(args.camera)
    est = HeadPoseEstimator(args.model)
    text = Text()
    open_window()
    imu.wait_ready()
    countdown = 3.0
    total = countdown + sum(d for _, d in CAL_PHASES)
    cam_t, cam_R, cam_det = [], [], []
    seq = 0
    t_start = time.perf_counter()
    t_rec0 = t_start + countdown
    print("캘리브레이션: 화면 안내에 따라 고개를 움직여 주세요")
    try:
        while True:
            now = time.perf_counter()
            el = now - t_start
            if el > total:
                break
            frame, t, seq = cam.wait_frame(seq)
            if frame is None:
                continue
            pose = est.process(frame, t)
            if pose and t >= t_rec0:
                cam_t.append(t)
                cam_R.append(pose.R)
                cam_det.append(pose.det)
            if el < countdown:
                msg, left = f"준비하세요… {countdown - el:.0f}", countdown - el
            else:
                acc = countdown
                for msg, d in CAL_PHASES:
                    if el < acc + d:
                        left = acc + d - el
                        break
                    acc += d
            h, w = frame.shape[:2]
            img = frame.copy()
            if pose:
                x0, y0, x1, y1 = pose.bbox
                cv2.rectangle(img, (x0, y0), (x1, y1), GREEN, 2)
            disp = to_display(img)
            if pose:
                draw_axes(disp, mirror_pt(pose.nose, w, h), mirror_R(pose.R), 90, 3)
            panel(disp, 0, 0, DISP_W, 95)
            prog = int(DISP_W * min(el / total, 1.0))
            cv2.rectangle(disp, (0, 90), (prog, 95), YELLOW, -1)
            disp = text.draw(disp, [
                (msg, (14, 8), YELLOW, True),
                (f"남은 시간 {left:.1f}s   |   얼굴 {'인식' if pose else '없음'}   |   샘플 {len(cam_t)}",
                 (16, 56), GREEN if pose else RED, False),
            ])
            cv2.imshow(WIN, disp)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27) or window_closed():
                print("중단했어요")
                return 1
        t_end = time.perf_counter()
        imu_t, imu_g, _ = imu.window(t_rec0 - 0.5, t_end + 0.5)
    finally:
        cam.close()
        est.close()
        imu.stop()
        cv2.destroyAllWindows()

    still_until = t_rec0 + CAL_PHASES[0][1]
    np.savez(raw_path, cam_t=np.array(cam_t), cam_R=np.array(cam_R),
             cam_det=np.array(cam_det), imu_t=imu_t, imu_g=imu_g, t_rec0=t_rec0, still_until=still_until)
    try:
        cal = solve_calibration(cam_t, cam_R, imu_t, imu_g, still_until=still_until)
    except RuntimeError as e:
        print(f"캘리브레이션 실패: {e}\n얼굴이 화면 안에 있는지 확인하고 다시 실행해 주세요.")
        return 1
    return report_calibration(cal, len(cam_t), args)


def report_calibration(cal, n_cam, args):
    ang = np.degrees(np.linalg.norm(so3.log(cal.R)))
    print("=== 캘리브레이션 결과 ===")
    print(f"얼굴 자세 샘플 {n_cam}개, 회전 샘플 {cal.n_samples}개")
    print(f"카메라 지연(tau) {cal.tau * 1e3:+.0f} ms, 속도 상관계수 {cal.corr:.3f}")
    print(f"자이로 스케일 {cal.gyro_scale:.3f} (1이면 rad/s, 57 근처면 deg/s)")
    print(f"잔차 {cal.residual:.3f}, 축 다양성 {cal.diversity:.2f}, 축 정합 안정성(95%) {cal.stability_deg:.2f}°")
    print(f"IMU→머리 회전각 {ang:.1f}°, 정지 바이어스 {np.round(cal.bias0, 4)} rad/s")
    print("R_HI =\n", np.round(cal.R, 3))
    print("결과:", "PASS" if cal.passed else "FAIL")
    if cal.passed or args.force:
        cal.save(args.calib)
        print(f"저장: {args.calib}")
    return 0 if cal.passed else 1


def _parse_dropout(s):
    if not s:
        return None
    a, b = s.split("-")
    return float(a), float(b)


def cmd_run(args):
    if not Path(args.calib).exists():
        print("캘리브레이션 파일이 없어요. 먼저: python tracker.py calibrate")
        return 1
    cal = Calibration.load(args.calib)
    fus = make_fusion(cal, args)
    imu = XrealImu().start()
    cam = Camera(args.camera)
    est = HeadPoseEstimator(args.model)
    text = Text()
    if not args.headless:
        open_window()
    imu.wait_ready()
    dropout = _parse_dropout(args.dropout)
    manual_off = False
    R_zero = np.eye(3)
    sess = Session(fus, imu)
    rec_t, rec_R = [], []
    hist = {k: ([], []) for k in ("cam", "fus", "gyro")}
    last_nose = None
    seq, n = 0, 0
    snap_done = args.snapshot is None
    result = 0
    t_start = time.perf_counter()
    try:
        while True:
            el = time.perf_counter() - t_start
            if args.duration and el > args.duration:
                break
            frame, t_cam, seq = cam.wait_frame(seq)
            if frame is None:
                continue
            n += 1
            pose = est.process(frame, t_cam)
            rel = t_cam - t_start
            off = manual_off or (dropout is not None and dropout[0] <= rel < dropout[1])
            cam_ypr, fus_ypr, gyro_ypr = sess.step(rel, t_cam, pose.R if pose else None, off, R_zero)
            if pose:
                last_nose = pose.nose
                rec_t.append(t_cam)
                rec_R.append(pose.R)
            for k, v in (("cam", cam_ypr), ("fus", fus_ypr), ("gyro", gyro_ypr)):
                hist[k][0].append(rel)
                hist[k][1].append(v[0] if v is not None else np.nan)

            want_snap = not snap_done and pose is not None and el > (args.duration or 20) / 2
            if args.headless and not want_snap:
                continue

            # live output: newest IMU sample, so latency is IMU latency not camera latency
            t_imu = imu.latest_time()
            R_now = fus.head(fus.predict(t_imu, imu)) if fus.ready and t_imu else None
            h, w = frame.shape[:2]
            img = privacy_blur(frame, pose.bbox) if want_snap else frame.copy()
            if pose:
                x0, y0, x1, y1 = pose.bbox
                cv2.rectangle(img, (x0, y0), (x1, y1), GREEN, 1)
            disp = to_display(img)
            if last_nose is not None:
                o = mirror_pt(last_nose, w, h)
                if pose:
                    draw_axes(disp, o, mirror_R(pose.R), 70, 1)
                if R_now is not None:
                    draw_axes(disp, o, mirror_R(R_now), 110, 4)
            if not pose:
                state = "얼굴 놓침 — IMU만"
            elif off:
                state = "카메라 보정 OFF — IMU만"
            elif fus.home:
                state = "정면 — 강하게 재보정 중"
            else:
                state = "융합 중 (정면을 보면 재보정)"
            panel(disp, 0, 0, 470, 150)
            panel(disp, 10, 416, 950, 446)
            items = [
                (state, (12, 6), GREEN if (pose and not off) else YELLOW, False),
                ("           yaw      pitch      roll", (12, 32), GRAY, False),
                ("카메라   " + fmt(cam_ypr), (12, 56), GREEN, False),
                ("융합      " + fmt(so3.euler_ypr(R_zero.T @ R_now) if R_now is not None else None), (12, 80), YELLOW, False),
                ("자이로만 " + fmt(gyro_ypr), (12, 104), RED, False),
                (f"bias {np.degrees(fus.bias).round(2)}°/s  지연 {fus.tau * 1e3:+.0f}ms", (12, 126), GRAY, False),
            ]
            if dropout is not None:
                guide = "카메라 보정 OFF — 계속 고개를 움직이세요" if off else "고개를 크게 움직여 보세요 (좌우 · 위아래 · 갸웃)"
                panel(disp, 480, 0, DISP_W, 50)
                items.append((guide, (492, 12), YELLOW if off else GREEN, False))
                items.append((f"{rel:4.0f}s / {args.duration:.0f}s", (DISP_W - 110, 12), WHITE, False))
            strip_chart(disp, [(hist["gyro"], RED), (hist["fus"], YELLOW), (hist["cam"], GREEN)], rel, rng=45.0,
                        box=(10, 446, 950, 530))
            items.append(("yaw 10초  초록=카메라 노랑=융합 빨강=자이로만  [q 종료 · z 지금을 정면으로 · c 재캘리브레이션 · d 보정 끄기]",
                          (16, 420), WHITE, False))
            disp = text.draw(disp, items)
            if want_snap:
                cv2.imwrite(str(args.snapshot), disp)
                snap_done = True
            if not args.headless:
                cv2.imshow(WIN, disp)
                k = cv2.waitKey(1) & 0xFF
                if k in (ord("q"), 27) or window_closed():
                    break
                if k == ord("z") and pose:
                    R_zero = pose.R.copy()
                    fus.set_home(pose.R)
                if k == ord("d"):
                    manual_off = not manual_off
                if k == ord("c"):
                    result = RECALIBRATE
                    break
        if args.record and rec_t:
            imu_t, imu_g, _ = imu.window(t_start - 1.0, time.perf_counter())
            np.savez(args.record, cam_t=np.array(rec_t), cam_R=np.array(rec_R), imu_t=imu_t, imu_g=imu_g)
            print(f"녹화 저장: {args.record}")
    finally:
        cam.close()
        est.close()
        imu.stop()
        cv2.destroyAllWindows()

    if args.log:
        write_log(args.log, sess.rows)
    if sess.rows:
        summarize(np.array(sess.rows, dtype=float), COLS, fus, n, time.perf_counter() - t_start)
    return result


COLS = ["t", "face", "applied", "cam_off", "cam_yaw", "cam_pitch", "cam_roll",
        "fus_yaw", "fus_pitch", "fus_roll", "gyro_yaw", "gyro_pitch", "gyro_roll",
        "err_fus_deg", "err_gyro_deg", "bias_x", "bias_y", "bias_z", "speed_dps"]
STILL_DPS = 10.0  # head speed below which the webcam pose is trustworthy (~0.5° noise)


def write_log(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(COLS)
        wr.writerows(rows)


def make_fusion(cal, args):
    return ComplementaryFusion(cal, kp=args.kp, ki=args.ki, rate_ref=args.rate_ref, bias_gate=args.bias_gate,
                               home_deg=args.home_deg, home_kp=args.home_kp)


class Session:
    """Per-camera-frame fusion bookkeeping shared by the live run and replay."""

    def __init__(self, fus, imu):
        self.fus, self.imu = fus, imu
        self.R_gyro, self.t_gyro = None, None   # raw-gyro baseline, never corrected
        self.rows = []

    def step(self, rel, t_cam, R_cam, off, R_zero):
        fus, imu = self.fus, self.imu
        t_m = t_cam + fus.tau
        if self.R_gyro is None and R_cam is not None:
            self.R_gyro, self.t_gyro = R_cam @ fus.R_HI, t_m
        elif self.R_gyro is not None and t_m > self.t_gyro:
            self.R_gyro = propagate(self.R_gyro, self.t_gyro, t_m, imu, np.zeros(3), fus.scale)
            self.t_gyro = t_m
        fus.advance(t_m, imu)
        R_pred = fus.head(fus.predict(t_m, imu)) if fus.ready else None
        err, applied = fus.update(t_cam, R_cam, imu, apply=not off) if R_cam is not None else (None, False)
        speed = np.degrees(fus._head_speed(t_m - 0.1, t_m, imu)) if fus.ready else np.nan
        R_gyro = fus.head(self.R_gyro) if self.R_gyro is not None else None

        def ypr(R):
            return so3.euler_ypr(R_zero.T @ R) if R is not None else None

        cam_ypr, fus_ypr, gyro_ypr = ypr(R_cam), ypr(R_pred), ypr(R_gyro)
        err_gyro = so3.angle_deg(R_gyro, R_cam) if (R_cam is not None and R_gyro is not None) else None
        nan3 = [np.nan] * 3
        self.rows.append([rel, int(R_cam is not None), int(applied), int(off),
                          *(cam_ypr if cam_ypr is not None else nan3),
                          *(fus_ypr if fus_ypr is not None else nan3),
                          *(gyro_ypr if gyro_ypr is not None else nan3),
                          err if err is not None else np.nan,
                          err_gyro if err_gyro is not None else np.nan,
                          *fus.bias, speed])
        return cam_ypr, fus_ypr, gyro_ypr


def cmd_replay(args):
    """Re-run fusion offline on a recording (no camera or glasses needed)."""
    d = np.load(args.record)
    cam_t, cam_R, imu_t, imu_g = d["cam_t"], d["cam_R"], d["imu_t"], d["imu_g"]
    t0 = cam_t[0]
    cal = Calibration.load(args.calib)
    if args.recalib:
        a, b = _parse_dropout(args.recalib)
        m = (cam_t - t0 >= a) & (cam_t - t0 < b)
        mi = (imu_t - t0 >= a - 0.5) & (imu_t - t0 < b + 0.5)
        new = solve_calibration(list(cam_t[m]), list(cam_R[m]), imu_t[mi], imu_g[mi])
        new.bias0 = cal.bias0
        print(f"[재캘리브레이션 {a:.0f}–{b:.0f}s] tau {new.tau * 1e3:+.0f}ms, 상관 {new.corr:.3f}, 잔차 {new.residual:.3f}, "
              f"안정성 {new.stability_deg:.2f}°, 기존 대비 회전 차이 {so3.angle_deg(new.R, cal.R):.2f}° "
              f"→ {'PASS' if new.passed else 'FAIL'}")
        cal = new
        if args.save_calib and new.passed:
            new.save(args.calib)
            print(f"저장: {args.calib}")
    ev = _parse_dropout(args.eval)
    dropout = _parse_dropout(args.dropout)
    fus = make_fusion(cal, args)
    sess = Session(fus, ArrayImu(imu_t, imu_g))
    for tc, R in zip(cam_t, cam_R):
        rel = tc - t0
        if ev and not (ev[0] <= rel < ev[1]):
            continue
        off = dropout is not None and dropout[0] <= rel < dropout[1]
        sess.step(rel, tc, R, off, np.eye(3))
    if args.log:
        write_log(args.log, sess.rows)
    D = np.array(sess.rows, dtype=float)
    summarize(D, COLS, fus, len(D), max(D[-1, 0] - D[0, 0], 1e-6))
    return 0


def summarize(D, cols, fus, n, elapsed):
    c = {k: i for i, k in enumerate(cols)}
    t, face, off = D[:, c["t"]], D[:, c["face"]] > 0, D[:, c["cam_off"]] > 0
    ef, eg = D[:, c["err_fus_deg"]], D[:, c["err_gyro_deg"]]
    print("=== 융합 실행 결과 ===")
    print(f"프레임 {n}개 ({n / elapsed:.1f} fps), 얼굴 검출 {100 * face.mean():.0f}%, "
          f"카메라 보정 반영 {100 * (D[:, c['applied']] > 0).mean():.0f}%, 바이어스 리셋 {fus.resets}회")

    def motion(sel):
        a = D[sel][:, [c["cam_yaw"], c["cam_pitch"], c["cam_roll"]]]
        return np.round(np.nanpercentile(a, 95, axis=0) - np.nanpercentile(a, 5, axis=0), 1)
    print(f"고개 움직임 범위(5–95%) yaw/pitch/roll {motion(face)}°")
    still = D[:, c["speed_dps"]] < STILL_DPS
    m = face & ~off & np.isfinite(ef)
    m[: min(len(m), 30)] = False  # skip filter warm-up (~1 s)
    if m.sum():
        e = ef[m]
        print(f"[카메라 보정 중] 융합 예측 vs 카메라: 중앙값 {np.median(e):.2f}°, 95% {np.percentile(e, 95):.2f}°, 최대 {e.max():.2f}°")
        if (m & still).sum() > 10:
            e = ef[m & still]
            print(f"   └ 고개가 멈춘 순간만 (카메라가 정확할 때, n={len(e)}): 중앙값 {np.median(e):.2f}°, 95% {np.percentile(e, 95):.2f}°")
    # frame-to-frame jitter while still: fused output vs raw camera
    cy = D[:, [c["cam_yaw"], c["cam_pitch"], c["cam_roll"]]]
    fy = D[:, [c["fus_yaw"], c["fus_pitch"], c["fus_roll"]]]
    pair = still[1:] & still[:-1] & face[1:] & face[:-1]
    if pair.sum() > 10:
        jc = np.nanmedian(np.linalg.norm(np.diff(cy, axis=0)[pair], axis=1))
        jf = np.nanmedian(np.linalg.norm(np.diff(fy, axis=0)[pair], axis=1))
        print(f"[떨림] 멈춰 있을 때 프레임 간 변화: 카메라 {jc:.2f}° → 융합 {jf:.2f}°")
    seg = face & off
    if seg.any():
        idx = np.flatnonzero(seg)
        starts = [idx[0]] + [b for a, b in zip(idx, idx[1:]) if b != a + 1]
        ends = [a for a, b in zip(idx, idx[1:]) if b != a + 1] + [idx[-1]]
        for s, e in zip(starts, ends):
            dur = t[e] - t[s]
            if dur < 1.0:
                continue
            gy = eg[s:e + 1]
            sel = np.zeros(len(t), bool)
            sel[s:e + 1] = True
            print(f"[카메라 보정 OFF {t[s]:.1f}–{t[e]:.1f}s, {dur:.1f}s] 융합 오차: 시작 {ef[s]:.2f}° → 끝 {ef[e]:.2f}° "
                  f"(최대 {np.nanmax(ef[s:e + 1]):.2f}°) | 자이로만 오차 변화 {gy[-1] - gy[0]:+.2f}° "
                  f"| 이 구간 움직임 {motion(sel & face)}°")
            late = sel & still & (t >= t[e] - 3.0)
            if late.sum() >= 3:
                print(f"   └ 마지막 3초 중 멈춘 순간의 융합 오차 중앙값 {np.nanmedian(ef[late]):.2f}° (IMU만으로 {dur:.0f}초 버틴 뒤)")
            after = np.flatnonzero((np.arange(len(t)) > e) & face & ~off & (ef < 2.0))
            if len(after):
                print(f"   └ 보정 재개 후 2° 이내 복귀까지 {t[after[0]] - t[e]:.2f}s")
    print(f"정면 재보정(큰 드리프트 즉시 제거) {fus.recenters}회")
    g = np.isfinite(eg) & face
    if g.sum() > 10:
        slope = np.polyfit(t[g], eg[g], 1)[0] * 60
        print(f"[자이로만] 최종 오차 {eg[g][-1]:.1f}°, 대략 {slope:+.1f}°/분 드리프트")
    print(f"최종 자이로 바이어스 추정 {np.degrees(fus.bias).round(3)} °/s")


def cmd_app(args):
    """Double-click entry point: calibrate when needed, then run until q (c recalibrates)."""
    print("XREAL Head Fusion — 웹캠 얼굴 추적 + XREAL One Pro IMU 융합")
    print(f"캘리브레이션 파일: {args.calib}\n")
    need_cal = not Path(args.calib).exists()
    try:
        while True:
            if need_cal:
                print("캘리브레이션을 시작해요. 창의 안내에 따라 고개를 움직여 주세요.")
                if cmd_calibrate(args) != 0:
                    if _ask("\n다시 시도할까요? [Enter = 다시 / q = 종료] ").strip().lower() == "q":
                        return 1
                    continue
                need_cal = False
            if cmd_run(args) == RECALIBRATE:
                need_cal = True
                continue
            return 0
    except RuntimeError as e:
        print(f"\n문제가 생겼어요: {e}")
        print("글래스 USB 연결과 웹캠을 확인한 뒤 다시 실행해 주세요.")
        return 1
    finally:
        if FROZEN:
            _ask("\nEnter를 누르면 창이 닫혀요.")


def _ask(prompt):
    try:
        return input(prompt)
    except EOFError:  # no console attached (e.g. launched from a script)
        return "q"


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:  # redirected output on Korean Windows is cp949; never crash on a stray symbol
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    if argv is None:
        argv = sys.argv[1:]
    if not any(a in SUBCOMMANDS for a in argv):
        argv = [*argv, "app"]
    if FROZEN:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--model", default=str(RES_DIR / "models" / "face_landmarker.task"))
    p.add_argument("--calib", default=str(DATA_DIR / "calib.json"))
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("app")
    s.set_defaults(duration=0, dropout=None, log=None, headless=False, snapshot=None, record=None,
                   force=False, from_raw=False)
    add_gains(s)
    s = sub.add_parser("imu-test")
    s.add_argument("--duration", type=float, default=5)
    s = sub.add_parser("face-test")
    s.add_argument("--duration", type=float, default=15)
    s.add_argument("--headless", action="store_true")
    s.add_argument("--snapshot", type=Path)
    s = sub.add_parser("calibrate")
    s.add_argument("--force", action="store_true", help="save even if checks fail")
    s.add_argument("--from-raw", action="store_true", help="re-solve from the last saved calib_raw.npz")
    s = sub.add_parser("run")
    s.add_argument("--duration", type=float, default=0, help="seconds, 0 = until q")
    s.add_argument("--dropout", help="disable camera correction for a window, e.g. 15-25 (s)")
    s.add_argument("--log", help="CSV output path")
    s.add_argument("--headless", action="store_true")
    s.add_argument("--snapshot", type=Path)
    s.add_argument("--record", help="save raw camera poses + IMU to .npz for replay")
    add_gains(s)
    s = sub.add_parser("replay")
    s.add_argument("record", help=".npz saved by run --record")
    s.add_argument("--dropout", help="simulate camera correction off, e.g. 20-30 (s)")
    s.add_argument("--eval", help="only evaluate this time range, e.g. 30-60 (s)")
    s.add_argument("--recalib", help="re-solve calibration from this range of the recording, e.g. 0-30 (s)")
    s.add_argument("--save-calib", action="store_true", help="write the re-solved calibration if it passes")
    s.add_argument("--log", help="CSV output path")
    add_gains(s)
    args = p.parse_args(argv)
    return {"app": cmd_app, "imu-test": cmd_imu_test, "face-test": cmd_face_test, "calibrate": cmd_calibrate,
            "run": cmd_run, "replay": cmd_replay}[args.cmd](args)


SUBCOMMANDS = {"app", "imu-test", "face-test", "calibrate", "run", "replay", "-h", "--help"}


def add_gains(s):
    s.add_argument("--kp", type=float, default=0.06, help="camera correction gain per frame")
    s.add_argument("--ki", type=float, default=0.3, help="gyro bias learning gain")
    s.add_argument("--rate-ref", type=float, default=0.6, help="head speed (rad/s) that halves the camera gain")
    s.add_argument("--bias-gate", type=float, default=0.25, help="learn bias only below this head speed (rad/s)")
    s.add_argument("--home-deg", type=float, default=15.0,
                   help="facing-forward cone (deg) for strong re-correction; 0 disables")
    s.add_argument("--home-kp", type=float, default=0.25, help="camera gain while facing forward and still")


if __name__ == "__main__":
    sys.exit(main())
