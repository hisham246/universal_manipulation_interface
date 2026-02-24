#!/usr/bin/env python3
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


# =========================================================
# CONFIG
# =========================================================
VICON_DIR   = Path("/home/hisham246/uwaterloo/cable_route_umi/vicon_logs_to_csv/")     # vicon_1.csv ...
EPISODE_DIR = Path("/home/hisham246/uwaterloo/cable_route_umi/vicon_final/")          # episode_1.csv ...
OUT_DIR     = Path("/home/hisham246/uwaterloo/cable_route_umi/task_frames_cable_station/")

OBJECT = "cable_station"
TIME_MARGIN_SEC = 0.05

# Vicon world -> SLAM world (180 deg around Z)
R_vicon_to_slam = np.array([[-1.0,  0.0, 0.0],
                            [ 0.0, -1.0, 0.0],
                            [ 0.0,  0.0, 1.0]], dtype=np.float64)
rot_v2s = R.from_matrix(R_vicon_to_slam)

# If your files are 1-based (episode_1.csv, vicon_1.csv), set True
ONE_BASED = True

# =========================================================
# HELPERS (mapping episode -> vicon by overlap, same as your old script)
# =========================================================
def natural_idx(p: Path, prefix: str) -> int:
    m = re.search(rf"{re.escape(prefix)}_(\d+)\.csv$", p.name)
    return int(m.group(1)) if m else -1


def align_object_local_axes_to_slam(df: pd.DataFrame, rot_v2s: R) -> pd.DataFrame:
    """
    Redefines the object's LOCAL axes to match the SLAM convention, 
    without erasing the true physical orientation of the object.
    
    Applies a constant right-multiplication: R_new = R_old * R_v2s
    """
    q = df[["Rot_X","Rot_Y","Rot_Z","Rot_W"]].to_numpy(dtype=np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    r_old = R.from_quat(q)

    # Right-multiply by the 180-deg Z rotation to swap the local axes convention
    r_new = r_old * rot_v2s         

    q_new = r_new.as_quat()
    out = df.copy()
    out[["Rot_X","Rot_Y","Rot_Z","Rot_W"]] = q_new
    return out

def read_episode_timestamps(ep_path: Path) -> np.ndarray:
    df = pd.read_csv(ep_path)
    if "timestamp" not in df.columns:
        raise ValueError(f"{ep_path} missing 'timestamp'. Found: {df.columns.tolist()}")
    t = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(dtype=np.float64)
    t = t[np.isfinite(t)]
    t = np.sort(t)
    return t


def read_vicon_df(vicon_path: Path) -> pd.DataFrame:
    df = pd.read_csv(vicon_path)
    if "Timestamp" not in df.columns:
        raise ValueError(f"{vicon_path} missing 'Timestamp'. Found: {df.columns.tolist()}")

    df = df.copy()
    df["Timestamp"] = pd.to_numeric(df["Timestamp"], errors="coerce")
    df = df[np.isfinite(df["Timestamp"])]
    df = df.sort_values("Timestamp")
    df = df.drop_duplicates(subset=["Timestamp"], keep="last").reset_index(drop=True)
    return df


def file_time_range_vicon(vicon_path: Path) -> tuple[float, float]:
    df = pd.read_csv(vicon_path, usecols=["Timestamp"])
    t = pd.to_numeric(df["Timestamp"], errors="coerce").to_numpy(dtype=np.float64)
    t = t[np.isfinite(t)]
    if len(t) == 0:
        return (np.inf, -np.inf)
    return (float(np.min(t)), float(np.max(t)))


def file_time_range_episode(ep_path: Path) -> tuple[float, float]:
    t = read_episode_timestamps(ep_path)
    if len(t) == 0:
        return (np.inf, -np.inf)
    return (float(t[0]), float(t[-1]))


def overlap_seconds(a0, a1, b0, b1) -> float:
    lo = max(a0, b0)
    hi = min(a1, b1)
    return max(0.0, hi - lo)


def match_vicon_to_episodes(vicon_files: list[Path], episode_files: list[Path]) -> dict[Path, Path]:
    v_meta = [(vf, *file_time_range_vicon(vf)) for vf in vicon_files]
    ep_meta = [(ef, *file_time_range_episode(ef)) for ef in episode_files]

    unused = set(vicon_files)
    mapping: dict[Path, Path] = {}

    for ef, es, ee in sorted(ep_meta, key=lambda x: x[1]):
        best_vf = None
        best_key = None
        for vf, vs, ve in v_meta:
            if vf not in unused:
                continue
            ov = overlap_seconds(es, ee, vs, ve)
            start_diff = abs(vs - es)
            key = (ov, -start_diff)
            if best_key is None or key > best_key:
                best_key = key
                best_vf = vf
        if best_vf is None:
            raise RuntimeError(f"No vicon file available for {ef.name}")
        mapping[ef] = best_vf
        unused.remove(best_vf)

    return mapping


# =========================================================
# RESAMPLE + TRANSFORM (your single-episode logic, generalized)
# =========================================================
def resample_object_to_episode_timestamps(vicon_df: pd.DataFrame, ep_t: np.ndarray, obj: str) -> pd.DataFrame:
    vt = vicon_df["Timestamp"].to_numpy(dtype=np.float64)
    if len(vt) < 2:
        raise ValueError("Vicon segment too small to resample (need >=2 timestamps).")

    t_query = np.clip(ep_t.copy(), vt[0], vt[-1])
    clipped = (ep_t < vt[0]) | (ep_t > vt[-1])
    clipped_frac = float(np.mean(clipped))

    # position
    pos_cols = [f"{obj}_Pos_X", f"{obj}_Pos_Y", f"{obj}_Pos_Z"]
    for c in pos_cols:
        if c not in vicon_df.columns:
            raise ValueError(f"Missing {c} in {obj} stream. Columns: {vicon_df.columns.tolist()}")

    px = pd.to_numeric(vicon_df[pos_cols[0]], errors="coerce").to_numpy(dtype=np.float64)
    py = pd.to_numeric(vicon_df[pos_cols[1]], errors="coerce").to_numpy(dtype=np.float64)
    pz = pd.to_numeric(vicon_df[pos_cols[2]], errors="coerce").to_numpy(dtype=np.float64)

    pos_x = np.interp(t_query, vt, px)
    pos_y = np.interp(t_query, vt, py)
    pos_z = np.interp(t_query, vt, pz)

    # quaternion
    quat_cols = [f"{obj}_Rot_X", f"{obj}_Rot_Y", f"{obj}_Rot_Z", f"{obj}_Rot_W"]
    for c in quat_cols:
        if c not in vicon_df.columns:
            raise ValueError(f"Missing {c} in {obj} stream. Columns: {vicon_df.columns.tolist()}")

    qdf = vicon_df[["Timestamp"] + quat_cols].copy()
    for c in quat_cols:
        qdf[c] = pd.to_numeric(qdf[c], errors="coerce")

    q = qdf[quat_cols].to_numpy(dtype=np.float64)
    finite = np.isfinite(q).all(axis=1)
    norms = np.linalg.norm(q, axis=1)
    valid = finite & (norms > 1e-8)

    if not np.all(valid):
        qdf.loc[~valid, quat_cols] = np.nan
        qdf[quat_cols] = qdf[quat_cols].ffill().bfill()
        q = qdf[quat_cols].to_numpy(dtype=np.float64)
        norms = np.linalg.norm(q, axis=1)
        valid2 = np.isfinite(q).all(axis=1) & (norms > 1e-8)
        if valid2.sum() < 2:
            raise ValueError(f"{obj} quaternion stream has <2 valid samples after repair.")
        qdf = qdf.loc[valid2].reset_index(drop=True)

    vt_q = qdf["Timestamp"].to_numpy(dtype=np.float64)
    q = qdf[quat_cols].to_numpy(dtype=np.float64)

    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    for i in range(1, len(q)):
        if np.dot(q[i - 1], q[i]) < 0:
            q[i] *= -1

    rot = R.from_quat(q)  # [x,y,z,w]
    slerp = Slerp(vt_q, rot)
    t_query_q = np.clip(t_query, vt_q[0], vt_q[-1])
    quat_ep = slerp(t_query_q).as_quat()

    # nearest-vicon diagnostics
    idx = np.searchsorted(vt, t_query, side="left")
    idx = np.clip(idx, 1, len(vt) - 1)
    left = vt[idx - 1]
    right = vt[idx]
    nearest = np.where(np.abs(t_query - left) <= np.abs(t_query - right), left, right)
    time_error = ep_t - nearest

    return pd.DataFrame({
        "timestamp": ep_t,
        "vicon_timestamp_used": t_query,
        "nearest_vicon_ts": nearest,
        "time_error_sec": time_error,
        "clipped": clipped,
        "clipped_frac": clipped_frac,
        "Pos_X": pos_x, "Pos_Y": pos_y, "Pos_Z": pos_z,
        "Rot_X": quat_ep[:, 0], "Rot_Y": quat_ep[:, 1], "Rot_Z": quat_ep[:, 2], "Rot_W": quat_ep[:, 3],
    })


def apply_global_transform(df: pd.DataFrame, rot_v2s: R) -> pd.DataFrame:
    p = df[["Pos_X", "Pos_Y", "Pos_Z"]].to_numpy(dtype=np.float64)
    p2 = (rot_v2s.as_matrix() @ p.T).T

    q = df[["Rot_X", "Rot_Y", "Rot_Z", "Rot_W"]].to_numpy(dtype=np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    r_in = R.from_quat(q)
    r_out = rot_v2s * r_in  # pre-multiply
    q2 = r_out.as_quat()

    out = df.copy()
    out[["Pos_X", "Pos_Y", "Pos_Z"]] = p2
    out[["Rot_X", "Rot_Y", "Rot_Z", "Rot_W"]] = q2
    return out


# =========================================================
# MAIN LOOP
# =========================================================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    vicon_files = sorted(VICON_DIR.glob("vicon_*.csv"), key=lambda p: natural_idx(p, "vicon"))
    episode_files = sorted(EPISODE_DIR.glob("episode_*.csv"), key=lambda p: natural_idx(p, "episode"))

    if not vicon_files or not episode_files:
        raise RuntimeError("No input files found. Check VICON_DIR / EPISODE_DIR.")

    # If you KNOW vicon_i matches episode_i 1-to-1, you can skip mapping and pair by index.
    # But safest is to reuse overlap mapping (like your old script).
    mapping = match_vicon_to_episodes(vicon_files, episode_files)

    # optional: log mapping
    map_rows = []
    for ep, vf in mapping.items():
        es, ee = file_time_range_episode(ep)
        vs, ve = file_time_range_vicon(vf)
        map_rows.append({
            "episode_file": ep.name,
            "vicon_file": vf.name,
            "overlap_sec": overlap_seconds(es, ee, vs, ve),
            "start_diff_sec": abs(vs - es),
        })
    pd.DataFrame(map_rows).to_csv(OUT_DIR / "episode_to_vicon_mapping.csv", index=False)

    # process all episodes
    BAD = []
    for ep_path in episode_files:
        vf_path = mapping[ep_path]
        ep_id = natural_idx(ep_path, "episode")  # 1-based if your naming is episode_001.csv etc.

        ep_t = read_episode_timestamps(ep_path)
        if len(ep_t) < 2:
            print(f"Skipping {ep_path.name}: too few timestamps ({len(ep_t)})")
            continue

        vdf = read_vicon_df(vf_path)

        t0 = float(ep_t[0]) - TIME_MARGIN_SEC
        t1 = float(ep_t[-1]) + TIME_MARGIN_SEC
        seg = vdf[(vdf["Timestamp"] >= t0) & (vdf["Timestamp"] <= t1)].copy()
        if len(seg) < 2:
            seg = vdf

        aligned = resample_object_to_episode_timestamps(seg, ep_t, OBJECT)
        aligned = apply_global_transform(aligned, rot_v2s)
        aligned = align_object_local_axes_to_slam(aligned, rot_v2s)

        aligned.insert(0, "episode_id", ep_id)
        aligned.insert(1, "vicon_id", natural_idx(vf_path, "vicon"))

        out_name = f"{OBJECT}_aligned_episode_{ep_id:03d}.csv"
        out_path = OUT_DIR / out_name
        aligned.to_csv(out_path, index=False)

        e = aligned["time_error_sec"].to_numpy(dtype=np.float64)
        e = e[np.isfinite(e)]
        abs_e = np.abs(e) if len(e) else np.array([np.nan])

        mean_ms = 1e3 * np.mean(e) if len(e) else np.nan
        p95_ms = 1e3 * np.quantile(abs_e, 0.95) if len(e) else np.nan
        max_ms = 1e3 * np.max(abs_e) if len(e) else np.nan
        clip_frac = float(aligned["clipped_frac"].iloc[0])

        print(f"[ep {ep_id:03d}] wrote {out_path.name} | mean={mean_ms:.3f}ms p95={p95_ms:.3f}ms max={max_ms:.3f}ms clip={100*clip_frac:.2f}%")

        # optional flagging
        if clip_frac > 0.01 or p95_ms > 10.0 or max_ms > 50.0:
            BAD.append({
                "episode_id": ep_id,
                "vicon_id": natural_idx(vf_path, "vicon"),
                "clip_frac": clip_frac,
                "mean_ms": mean_ms,
                "p95_ms": p95_ms,
                "max_ms": max_ms,
                "episode_file": ep_path.name,
                "vicon_file": vf_path.name,
            })

    if BAD:
        bad_df = pd.DataFrame(BAD).sort_values("max_ms", ascending=False)
        bad_df.to_csv(OUT_DIR / "bad_episodes.csv", index=False)
        print("Wrote bad episode report:", OUT_DIR / "bad_episodes.csv")

    print("Done.")
    print("Outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()