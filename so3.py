"""Small SO(3) helpers (rotation matrices, axis-angle vectors in rad)."""
import numpy as np


def hat(w):
    x, y, z = w
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def exp(w):
    th = np.linalg.norm(w)
    if th < 1e-9:
        return np.eye(3) + hat(w)
    K = hat(np.asarray(w) / th)
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def exp_batch(W):
    """(N,3) rotation vectors -> (N,3,3) matrices."""
    W = np.asarray(W, dtype=float)
    th = np.linalg.norm(W, axis=1)
    safe = np.where(th < 1e-12, 1.0, th)
    k = W / safe[:, None]
    K = np.zeros((len(W), 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]
    K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
    s = np.sin(th)[:, None, None]
    c = (1.0 - np.cos(th))[:, None, None]
    return np.eye(3) + s * K + c * (K @ K)


def log(R):
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    th = np.arccos(c)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if th < 1e-6:
        return 0.5 * v
    if np.pi - th < 1e-4:
        A = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(A)))
        axis = A[:, i] / np.sqrt(max(A[i, i], 1e-12))
        return axis * th
    return th / (2.0 * np.sin(th)) * v


def angle_deg(Ra, Rb):
    """Geodesic angle between two rotations, degrees."""
    c = np.clip((np.trace(Ra.T @ Rb) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def orthonormalize(M):
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def integrate(R, W):
    """Right-multiply R by exp of each rotation increment in W (N,3)."""
    for M in exp_batch(W):
        R = R @ M
    return R


def euler_ypr(R):
    """R = Ry(yaw) @ Rx(pitch) @ Rz(roll) -> (yaw, pitch, roll) degrees."""
    pitch = np.arcsin(np.clip(-R[1, 2], -1.0, 1.0))
    yaw = np.arctan2(R[0, 2], R[2, 2])
    roll = np.arctan2(R[1, 0], R[1, 1])
    return np.degrees([yaw, pitch, roll])
