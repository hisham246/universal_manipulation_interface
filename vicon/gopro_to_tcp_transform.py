#!/usr/bin/env python3
import pathlib
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

CSV_PATH = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon/gopro_to_tcp_transform.csv"
POS_SCALE = 1e-3  # mm -> m

CAM_SUFFIX = ""     # RX..TZ
TCP_SUFFIX = ".1"   # RX.1..TZ.1

MAD_K = 3.5
MAD_SCALE = 1.4826

def read_vicon_two_body_csv(csv_path: str) -> pd.DataFrame:
    p = pathlib.Path(csv_path)
    lines = p.read_text(errors="ignore").splitlines()

    header_idx = None
    for i, ln in enumerate(lines):
        low = ln.strip().lower()
        if low.startswith("frame,") and ("tx" in low) and ("rw" in low):
            header_idx = i
            break
    if header_idx is None:
        raise RuntimeError(f"Could not find header row in: {csv_path}")

    df = pd.read_csv(p, skiprows=header_idx)
    df.columns = [c.strip() for c in df.columns]
    df["Frame"] = pd.to_numeric(df["Frame"], errors="coerce")
    df = df.dropna(subset=["Frame"]).copy()
    return df

def get_pos_quat(df: pd.DataFrame, suffix: str, pos_scale: float):
    rx, ry, rz, rw = f"RX{suffix}", f"RY{suffix}", f"RZ{suffix}", f"RW{suffix}"
    tx, ty, tz = f"TX{suffix}", f"TY{suffix}", f"TZ{suffix}"

    for c in [rx, ry, rz, rw, tx, ty, tz]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[rx, ry, rz, rw, tx, ty, tz]).copy()

    pos = df[[tx, ty, tz]].to_numpy(np.float64) * pos_scale
    quat = df[[rx, ry, rz, rw]].to_numpy(np.float64)
    quat /= (np.linalg.norm(quat, axis=1, keepdims=True) + 1e-12)
    return pos, quat

def robust_inliers_on_norm(x: np.ndarray):
    # x: (N,D). Compute inliers based on norm around median using MAD.
    med = np.median(x, axis=0)
    r = np.linalg.norm(x - med, axis=1)
    med_r = np.median(r)
    mad_r = np.median(np.abs(r - med_r)) + 1e-12
    thr = med_r + MAD_K * MAD_SCALE * mad_r
    return r < thr

def quat_mean_markley(quat_xyzw: np.ndarray) -> np.ndarray:
    # Markley quaternion average. Returns xyzw.
    # Ensure consistent sign to avoid cancellation
    q = quat_xyzw.copy()
    # Reference sign by first quaternion
    ref = q[0]
    dots = (q * ref).sum(axis=1)
    q[dots < 0] *= -1.0

    A = np.zeros((4, 4), dtype=np.float64)
    for qi in q:
        A += np.outer(qi, qi)
    A /= len(q)

    eigvals, eigvecs = np.linalg.eigh(A)
    q_mean = eigvecs[:, np.argmax(eigvals)]
    # normalize and fix sign
    q_mean /= (np.linalg.norm(q_mean) + 1e-12)
    if q_mean[3] < 0:
        q_mean *= -1.0
    return q_mean

def main():
    df = read_vicon_two_body_csv(CSV_PATH)

    pos_cam, quat_cam = get_pos_quat(df, CAM_SUFFIX, POS_SCALE)
    pos_tcp, quat_tcp = get_pos_quat(df, TCP_SUFFIX, POS_SCALE)
    N = min(len(pos_cam), len(pos_tcp))
    pos_cam, quat_cam = pos_cam[:N], quat_cam[:N]
    pos_tcp, quat_tcp = pos_tcp[:N], quat_tcp[:N]

    Rw_cam = Rotation.from_quat(quat_cam)
    Rw_tcp = Rotation.from_quat(quat_tcp)

    # relative rotation cam->tcp: R_cam_tcp = R_cam^T R_tcp
    R_cam_tcp = Rw_cam.inv() * Rw_tcp
    q_rel = R_cam_tcp.as_quat()  # xyzw
    t_rel = Rw_cam.inv().apply(pos_tcp - pos_cam)  # expressed in camera frame

    # robust inliers using translation norm + rotation angle
    in_t = robust_inliers_on_norm(t_rel)
    ang = R_cam_tcp.magnitude()  # angle in radians, (N,)
    # robust inliers on angle (treat as 1D vector)
    in_r = robust_inliers_on_norm(ang.reshape(-1, 1))
    inliers = in_t & in_r

    t_in = t_rel[inliers]
    q_in = q_rel[inliers]

    t_mean = t_in.mean(axis=0)
    t_std = t_in.std(axis=0)

    q_bias = quat_mean_markley(q_in)  # mean relative rotation cam->tcp
    R_bias = Rotation.from_quat(q_bias)
    rotvec_bias = R_bias.as_rotvec()

    print("\n=== Estimated cam->tcp from relative transforms ===")
    print(f"Samples: {N} | Inliers: {inliers.sum()} ({inliers.mean()*100:.1f}%)")
    print(f"t_cam_tcp mean [m]: {t_mean}")
    print(f"t_cam_tcp std  [m]: {t_std}")
    print(f"R_cam_tcp mean rotvec [rad]: {rotvec_bias}")
    print(f"R_cam_tcp mean angle [deg]: {np.linalg.norm(rotvec_bias) * 180/np.pi:.4f}")

    print("\nIf you want to nullify tcp rotation (align tcp axes to camera axes):")
    print("  Use rotation = [0,0,0] in pose_cam_tcp, and keep this translation.")
    print("\nPaste into your main script as:")
    print(f"pose_cam_tcp = np.array([{t_mean[0]: .8f}, {t_mean[1]: .8f}, {t_mean[2]: .8f}, 0.0, 0.0, 0.0], dtype=np.float64)")
    print("T_cam_tcp = pose_to_mat(pose_cam_tcp)")

    print("\n(For reference) The rotation you are nullifying is approximately:")
    print(f"rotvec_bias = np.array([{rotvec_bias[0]: .8f}, {rotvec_bias[1]: .8f}, {rotvec_bias[2]: .8f}])")

if __name__ == "__main__":
    main()
