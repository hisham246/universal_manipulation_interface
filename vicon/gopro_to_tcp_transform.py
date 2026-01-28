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
    # x: (N,D). Compute inliers based on distance to median using MAD.
    med = np.median(x, axis=0)
    r = np.linalg.norm(x - med, axis=1)
    med_r = np.median(r)
    mad_r = np.median(np.abs(r - med_r)) + 1e-12
    thr = med_r + MAD_K * MAD_SCALE * mad_r
    return r < thr

def main():
    df = read_vicon_two_body_csv(CSV_PATH)

    pos_cam, quat_cam = get_pos_quat(df, CAM_SUFFIX, POS_SCALE)
    pos_tcp, _quat_tcp = get_pos_quat(df, TCP_SUFFIX, POS_SCALE)

    N = min(len(pos_cam), len(pos_tcp))
    pos_cam, quat_cam = pos_cam[:N], quat_cam[:N]
    pos_tcp = pos_tcp[:N]

    # Use only camera orientation to express the rigid offset in the camera frame:
    # t_cam_tcp_i = R_w_cam_i^T (p_w_tcp_i - p_w_cam_i)
    Rw_cam = Rotation.from_quat(quat_cam)
    t_cam_tcp = Rw_cam.inv().apply(pos_tcp - pos_cam)  # (N,3) in camera frame

    # Robustly keep samples consistent with a single rigid translation
    inliers = robust_inliers_on_norm(t_cam_tcp)
    t_in = t_cam_tcp[inliers]

    t_mean = t_in.mean(axis=0)
    t_std = t_in.std(axis=0)

    print("\n=== Estimated cam->tcp translation (assuming zero relative rotation) ===")
    print(f"Samples: {N} | Inliers: {inliers.sum()} ({inliers.mean()*100:.1f}%)")
    print(f"t_cam_tcp mean [m]: {t_mean}")
    print(f"t_cam_tcp std  [m]: {t_std}")

    print("\nPaste into your main script as:")
    print(
        "pose_cam_tcp = np.array(["
        f"{t_mean[0]: .8f}, {t_mean[1]: .8f}, {t_mean[2]: .8f}, "
        "0.0, 0.0, 0.0], dtype=np.float64)"
    )
    print("T_cam_tcp = pose_to_mat(pose_cam_tcp)")

    # Optional: quick sanity check on residuals (how rigid it really is)
    resid = np.linalg.norm(t_cam_tcp - t_mean[None, :], axis=1)
    print(f"\nResidual norm [m]: median={np.median(resid):.6f}, p95={np.percentile(resid,95):.6f}, max={resid.max():.6f}")

if __name__ == "__main__":
    main()
