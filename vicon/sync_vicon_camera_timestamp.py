import re
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


# -----------------------------
# Config
# -----------------------------

VICON_DIR   = Path("/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon_v3/vicon_logs_to_csv/")     # contains vicon_1.csv ... vicon_257.csv
EPISODE_DIR = Path("/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon_v3/camera_timestamps/")   # contains episode_1.csv ... episode_256.csv
OUT_DIR     = Path("/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon_v3/aligned_vicon_files/")

TIME_MARGIN_SEC = 0.25   # crop margin around episode [start,end] when selecting vicon samples


# -----------------------------
# Helpers
# -----------------------------
def natural_idx(p: Path, prefix: str) -> int:
    m = re.search(rf"{re.escape(prefix)}_(\d+)\.csv$", p.name)
    return int(m.group(1)) if m else -1


def read_episode_timestamps(ep_path: Path) -> np.ndarray:
    df = pd.read_csv(ep_path)
    if "timestamp" not in df.columns:
        raise ValueError(f"{ep_path} has no 'timestamp' column. Found: {df.columns.tolist()}")
    t = df["timestamp"].to_numpy(dtype=np.float64)
    # Ensure sorted & unique (camera timestamps sometimes repeat)
    t = np.sort(t)
    t = t[np.isfinite(t)]
    return t


def read_vicon_df(vicon_path: Path) -> pd.DataFrame:
    df = pd.read_csv(vicon_path)

    # Basic checks
    if "Timestamp" not in df.columns:
        raise ValueError(f"{vicon_path} has no 'Timestamp' column. Found: {df.columns.tolist()}")

    # Sort by time and drop duplicate timestamps (keep last)
    df = df.copy()
    df["Timestamp"] = df["Timestamp"].astype(np.float64)
    df = df[np.isfinite(df["Timestamp"])]
    df = df.sort_values("Timestamp")
    df = df.drop_duplicates(subset=["Timestamp"], keep="last").reset_index(drop=True)

    return df


def file_time_range_vicon(vicon_path: Path) -> tuple[float, float]:
    # quick read only timestamps
    df = pd.read_csv(vicon_path, usecols=["Timestamp"])
    t = df["Timestamp"].to_numpy(dtype=np.float64)
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
    """
    Returns mapping: episode_file -> chosen_vicon_file
    Greedy assignment by max overlap (then smallest start-time difference), preventing re-use.
    """
    v_meta = []
    for vf in vicon_files:
        s, e = file_time_range_vicon(vf)
        v_meta.append((vf, s, e))

    ep_meta = []
    for ef in episode_files:
        s, e = file_time_range_episode(ef)
        ep_meta.append((ef, s, e))

    unused = set(vicon_files)
    mapping = {}

    for ef, es, ee in sorted(ep_meta, key=lambda x: x[1]):  # earliest episodes first
        best = None
        best_key = None

        for vf, vs, ve in v_meta:
            if vf not in unused:
                continue
            ov = overlap_seconds(es, ee, vs, ve)
            start_diff = abs(vs - es)
            # Key: maximize overlap, then minimize start diff
            key = (ov, -start_diff)

            if best is None or key > best_key:
                best = vf
                best_key = key

        if best is None:
            raise RuntimeError(f"No vicon file available for {ef.name}")

        mapping[ef] = best
        unused.remove(best)

    return mapping


def resample_vicon_to_episode_timestamps(vicon_df: pd.DataFrame, ep_t: np.ndarray) -> pd.DataFrame:
    """
    Inputs:
      - vicon_df: sorted by Timestamp, unique timestamps
      - ep_t: episode timestamps (sorted)
    Output:
      DataFrame with exactly len(ep_t) rows, indexed in same order as ep_t,
      containing interpolated position + SLERP quaternion.
    """
    vt = vicon_df["Timestamp"].to_numpy(dtype=np.float64)

    # Need at least 2 samples for interpolation/slerp
    if len(vt) < 2:
        raise ValueError("Vicon segment too small to resample (need >=2 timestamps).")

    # Ensure episode timestamps are within vicon time span (clip for safety)
    t_query = ep_t.copy()
    t_query = np.clip(t_query, vt[0], vt[-1])
    clipped = (ep_t < vt[0]) | (ep_t > vt[-1])
    clipped_frac = float(np.mean(clipped))

    # ---- Linear interp for position ----
    pos_cols = ["Pos_X", "Pos_Y", "Pos_Z"]
    for c in pos_cols:
        if c not in vicon_df.columns:
            raise ValueError(f"Missing {c} in vicon data. Columns: {vicon_df.columns.tolist()}")
    px = vicon_df["Pos_X"].to_numpy(dtype=np.float64)
    py = vicon_df["Pos_Y"].to_numpy(dtype=np.float64)
    pz = vicon_df["Pos_Z"].to_numpy(dtype=np.float64)

    pos_x = np.interp(t_query, vt, px)
    pos_y = np.interp(t_query, vt, py)
    pos_z = np.interp(t_query, vt, pz)

    # ---- SLERP for quaternion ----
    quat_cols = ["Rot_X", "Rot_Y", "Rot_Z", "Rot_W"]
    for c in quat_cols:
        if c not in vicon_df.columns:
            raise ValueError(f"Missing {c} in vicon data. Columns: {vicon_df.columns.tolist()}")

    # work on a copy of only needed cols
    qdf = vicon_df[["Timestamp"] + quat_cols].copy()

    # force numeric
    for c in quat_cols:
        qdf[c] = pd.to_numeric(qdf[c], errors="coerce")

    # mark invalid (non-finite or ~zero-norm)
    q = qdf[quat_cols].to_numpy(dtype=np.float64)
    finite = np.isfinite(q).all(axis=1)
    norms = np.linalg.norm(q, axis=1)
    valid = finite & (norms > 1e-8)

    # If some invalid rows exist, repair them by nearest-valid fill (ffill then bfill)
    if not np.all(valid):
        qdf.loc[~valid, quat_cols] = np.nan
        qdf[quat_cols] = qdf[quat_cols].ffill().bfill()
        q = qdf[quat_cols].to_numpy(dtype=np.float64)

        # re-check validity after repair
        norms = np.linalg.norm(q, axis=1)
        valid2 = np.isfinite(q).all(axis=1) & (norms > 1e-8)

        # If still invalid (e.g., entire file invalid), fail loudly
        if valid2.sum() < 2:
            raise ValueError(
                "Quaternion stream has <2 valid samples even after repair. "
                "Check Vicon export / object tracking dropouts."
            )

        # Keep only valid2 rows (in case leading/trailing stayed bad somehow)
        qdf = qdf.loc[valid2].reset_index(drop=True)
        vt_q = qdf["Timestamp"].to_numpy(dtype=np.float64)
        q = qdf[quat_cols].to_numpy(dtype=np.float64)
    else:
        vt_q = qdf["Timestamp"].to_numpy(dtype=np.float64)

    # normalize
    q_norm = np.linalg.norm(q, axis=1, keepdims=True)
    q = q / q_norm

    # hemisphere continuity (avoid long-way slerp)
    for i in range(1, len(q)):
        if np.dot(q[i-1], q[i]) < 0:
            q[i] *= -1

    rot = R.from_quat(q)  # [x,y,z,w]
    slerp = Slerp(vt_q, rot)

    # IMPORTANT: clip query timestamps to quaternion time span (vt_q may differ from vt)
    t_query_q = np.clip(t_query, vt_q[0], vt_q[-1])
    rot_q = slerp(t_query_q).as_quat()


    out = pd.DataFrame({
        "timestamp": ep_t,              # original episode timestamps (not clipped)
        "vicon_timestamp_used": t_query, # clipped timestamps actually queried
        "Pos_X": pos_x,
        "Pos_Y": pos_y,
        "Pos_Z": pos_z,
        "Rot_X": rot_q[:, 0],
        "Rot_Y": rot_q[:, 1],
        "Rot_Z": rot_q[:, 2],
        "Rot_W": rot_q[:, 3],
    })

    # clipping diagnostics (episode time outside available vicon range)
    out["clipped"] = clipped
    out["clipped_frac"] = clipped_frac  # constant per episode, repeated per row

    # -----------------------------
    # Nearest-Vicon timestamp diagnostics (how far are we from real samples?)
    # -----------------------------
    idx = np.searchsorted(vt, t_query, side="left")
    idx = np.clip(idx, 1, len(vt) - 1)

    left = vt[idx - 1]
    right = vt[idx]

    nearest = np.where(np.abs(t_query - left) <= np.abs(t_query - right), left, right)

    out["nearest_vicon_ts"] = nearest
    out["time_error_sec"] = out["timestamp"] - out["nearest_vicon_ts"]

    return out


# -----------------------------
# Main pipeline
# -----------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    aligned_dir = OUT_DIR / "aligned_vicon_to_episode"
    aligned_dir.mkdir(parents=True, exist_ok=True)

    vicon_files = sorted(VICON_DIR.glob("vicon_*.csv"), key=lambda p: natural_idx(p, "vicon"))
    episode_files = sorted(EPISODE_DIR.glob("episode_*.csv"), key=lambda p: natural_idx(p, "episode"))

    if len(episode_files) == 0 or len(vicon_files) == 0:
        raise RuntimeError("No input files found. Check VICON_DIR and EPISODE_DIR.")

    # 1) Build mapping episode -> vicon by timestamp overlap
    mapping = match_vicon_to_episodes(vicon_files, episode_files)

    # For inspection/debugging
    map_rows = []
    for ep, vf in mapping.items():
        es, ee = file_time_range_episode(ep)
        vs, ve = file_time_range_vicon(vf)
        map_rows.append({
            "episode_file": ep.name,
            "vicon_file": vf.name,
            "episode_start": es,
            "episode_end": ee,
            "vicon_start": vs,
            "vicon_end": ve,
            "overlap_sec": overlap_seconds(es, ee, vs, ve),
            "start_diff_sec": abs(vs - es),
        })
    map_df = pd.DataFrame(map_rows).sort_values("episode_start")
    map_df.to_csv(OUT_DIR / "episode_to_vicon_mapping.csv", index=False)

    # 2) For each episode: crop vicon around episode, resample to episode timestamps, save
    aligned_all = []
    episodes_all = []

    BAD_EPISODES = []
    P95_MS_THRESH = 10.0     # should be ~5 ms normally
    MAX_MS_THRESH = 50.0     # anything above this is severe
    CLIP_FRAC_THRESH = 0.01  # >1% of frames clipped => suspicious

    for ep_path in episode_files:
        vf_path = mapping[ep_path]

        ep_t = read_episode_timestamps(ep_path)
        if len(ep_t) < 2:
            print(f"Skipping {ep_path.name}: too few timestamps ({len(ep_t)})")
            continue

        vdf = read_vicon_df(vf_path)

        t0 = float(ep_t[0]) - TIME_MARGIN_SEC
        t1 = float(ep_t[-1]) + TIME_MARGIN_SEC
        seg = vdf[(vdf["Timestamp"] >= t0) & (vdf["Timestamp"] <= t1)].copy()

        # If margin crop ended up too small (rare), fall back to full vicon file
        if len(seg) < 2:
            seg = vdf

        aligned = resample_vicon_to_episode_timestamps(seg, ep_t)

        aligned.insert(0, "episode_id", natural_idx(ep_path, "episode"))
        aligned.insert(1, "vicon_id", natural_idx(vf_path, "vicon"))

        out_name = f"aligned_episode_{natural_idx(ep_path,'episode'):03d}.csv"
        aligned.to_csv(aligned_dir / out_name, index=False)


        # -----------------------------
        # Per-episode time alignment stats (diagnostic)
        # -----------------------------
        e = aligned["time_error_sec"].to_numpy(dtype=np.float64)
        abs_e = np.abs(e)

        print(
            f"[episode {natural_idx(ep_path,'episode'):03d} | vicon {natural_idx(vf_path,'vicon'):03d}] "
            f"mean={1e3*np.mean(e):.3f} ms, "
            f"p95={1e3*np.quantile(abs_e, 0.95):.3f} ms, "
            f"p99={1e3*np.quantile(abs_e, 0.99):.3f} ms, "
            f"max={1e3*np.max(abs_e):.3f} ms"
        )

        # -----------------------------
        # Flagging logic
        # -----------------------------
        e = aligned["time_error_sec"].to_numpy(dtype=np.float64)
        e = e[np.isfinite(e)]
        abs_e = np.abs(e)

        mean_ms = 1e3 * np.mean(e)
        p95_ms  = 1e3 * np.quantile(abs_e, 0.95)
        p99_ms  = 1e3 * np.quantile(abs_e, 0.99)
        max_ms  = 1e3 * np.max(abs_e)

        clip_frac = float(aligned["clipped_frac"].iloc[0]) if "clipped_frac" in aligned.columns else 0.0

        is_bad = (clip_frac > CLIP_FRAC_THRESH) or (p95_ms > P95_MS_THRESH) or (max_ms > MAX_MS_THRESH)

        if is_bad:
            print(f"  >>> BAD: clip_frac={100*clip_frac:.2f}%  p95={p95_ms:.1f}ms  max={max_ms:.1f}ms")
            BAD_EPISODES.append({
                "episode_id": natural_idx(ep_path, "episode"),
                "vicon_id": natural_idx(vf_path, "vicon"),
                "clip_frac": clip_frac,
                "mean_ms": mean_ms,
                "p95_ms": p95_ms,
                "p99_ms": p99_ms,
                "max_ms": max_ms,
                "episode_file": ep_path.name,
                "vicon_file": vf_path.name,
            })

        aligned_all.append(aligned)

        # Also store episode timestamps table (useful later)
        episodes_all.append(pd.DataFrame({
            "episode_id": natural_idx(ep_path, "episode"),
            "timestamp": ep_t
        }))

    # 3) Concatenate “big tables” (optional, but convenient)
    if aligned_all:
        aligned_big = pd.concat(aligned_all, ignore_index=True)
        e = aligned_big["time_error_sec"].to_numpy(dtype=np.float64)
        e = e[np.isfinite(e)]
        abs_e = np.abs(e)
        print("\n=== GLOBAL time_error_sec stats ===")
        print("mean (ms):", 1e3*np.mean(e))
        print("95% (ms):", 1e3*np.quantile(abs_e, 0.95))
        print("99% (ms):", 1e3*np.quantile(abs_e, 0.99))
        print("max (ms):", 1e3*np.max(abs_e))

        aligned_big.to_parquet(OUT_DIR / "aligned_vicon_all.parquet", index=False)
        aligned_big.to_csv(OUT_DIR / "aligned_vicon_all.csv", index=False)

    if episodes_all:
        ep_big = pd.concat(episodes_all, ignore_index=True)
        ep_big.to_parquet(OUT_DIR / "episode_timestamps_all.parquet", index=False)
        ep_big.to_csv(OUT_DIR / "episode_timestamps_all.csv", index=False)


    if BAD_EPISODES:
        bad_df = pd.DataFrame(BAD_EPISODES).sort_values(["max_ms"], ascending=False)
        bad_df.to_csv(OUT_DIR / "bad_episodes.csv", index=False)
        print(f"\nWrote bad episode report: {OUT_DIR / 'bad_episodes.csv'}")
    else:
        print("\nNo bad episodes flagged.")

    print("Done.")
    print(f"- Mapping CSV: {OUT_DIR / 'episode_to_vicon_mapping.csv'}")
    print(f"- Per-episode aligned CSVs: {aligned_dir}")
    print(f"- Concatenated aligned table: {OUT_DIR / 'aligned_vicon_all.parquet'}")


if __name__ == "__main__":
    main()
