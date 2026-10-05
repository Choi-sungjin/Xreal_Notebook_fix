"""Webcam capture + MediaPipe face landmarker head pose.

Head pose comes from the landmarker's facial transformation matrix
(canonical face -> camera, OpenGL axes: x right, y up, z toward the viewer),
so a face looking straight at the camera is roughly the identity rotation.
Only the largest face in view is used, so people behind the user are ignored.
"""
import threading
import time
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

import so3


class Camera:
    def __init__(self, index=0, width=1280, height=720):
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError("웹캠을 열 수 없어요")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._lock = threading.Lock()
        self._frame, self._t, self._seq = None, 0.0, 0
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while self._running:
            ok, frame = self.cap.read()
            t = time.perf_counter()
            if ok:
                with self._lock:
                    self._frame, self._t, self._seq = frame, t, self._seq + 1

    def latest(self):
        with self._lock:
            return self._frame, self._t, self._seq

    def wait_frame(self, last_seq, timeout=1.0):
        t_end = time.perf_counter() + timeout
        while time.perf_counter() < t_end:
            frame, t, seq = self.latest()
            if seq != last_seq and frame is not None:
                return frame, t, seq
            time.sleep(0.002)
        return None, 0.0, last_seq

    def close(self):
        self._running = False
        self._thread.join(timeout=2)
        self.cap.release()


@dataclass
class HeadPose:
    R: np.ndarray          # face -> camera rotation
    bbox: tuple            # x0, y0, x1, y1 in pixels
    nose: tuple            # nose tip pixel
    landmarks: np.ndarray  # (N, 2) pixels
    n_faces: int
    det: float = 1.0        # det of the raw 3x3 block (sanity: should be > 0)


class HeadPoseEstimator:
    def __init__(self, model_path, num_faces=2):
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=num_faces,
            output_facial_transformation_matrixes=True,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._lm = vision.FaceLandmarker.create_from_options(opts)
        self._last_ms = -1

    def process(self, frame_bgr, t):
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ms = max(int(t * 1000), self._last_ms + 1)
        self._last_ms = ms
        res = self._lm.detect_for_video(image, ms)
        if not res.face_landmarks:
            return None
        best, best_area, best_pts = None, -1.0, None
        for i, lms in enumerate(res.face_landmarks):
            pts = np.array([[p.x * w, p.y * h] for p in lms])
            x0, y0 = pts.min(0)
            x1, y1 = pts.max(0)
            area = (x1 - x0) * (y1 - y0)
            if area > best_area:
                best, best_area, best_pts = i, area, pts
        M = np.array(res.facial_transformation_matrixes[best])
        x0, y0 = best_pts.min(0)
        x1, y1 = best_pts.max(0)
        return HeadPose(
            R=so3.orthonormalize(M[:3, :3]),
            bbox=(int(x0), int(y0), int(x1), int(y1)),
            nose=(float(best_pts[1, 0]), float(best_pts[1, 1])),
            landmarks=best_pts,
            n_faces=len(res.face_landmarks),
            det=float(np.linalg.det(M[:3, :3])),
        )

    def close(self):
        self._lm.close()


def privacy_blur(frame, bbox, margin=0.4):
    """Blur everything outside the (expanded) user face box."""
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = bbox
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    x0, y0 = max(0, int(x0 - mx)), max(0, int(y0 - my))
    x1, y1 = min(w, int(x1 + mx)), min(h, int(y1 + my))
    out = cv2.GaussianBlur(cv2.resize(frame, (w // 8, h // 8)), (0, 0), 6)
    out = cv2.resize(out, (w, h))
    out[y0:y1, x0:x1] = frame[y0:y1, x0:x1]
    return out
