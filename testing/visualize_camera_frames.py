#!/usr/bin/env python3
"""
Visualize RGB frames from a Diffusion Policy Zarr replay buffer (zarr.zip).

- Opens the zarr.zip
- Finds all keys like camera*_rgb
- Lets you iterate frame-by-frame (or autoplay) with keyboard controls
- Optionally jump by episode and show episode index / local frame index

Controls (in the OpenCV window):
  q / ESC : quit
  d / →   : next frame
  a / ←   : prev frame
  space   : toggle autoplay
  ]       : faster autoplay
  [       : slower autoplay
  n       : next episode (jump to first frame of next ep)
  p       : prev episode
  j       : prompt in terminal for absolute frame index to jump to
"""

import argparse
import re
import time
from typing import List, Tuple

import numpy as np
import zarr
import cv2

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs

register_codecs()


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def find_rgb_keys(data_group) -> List[str]:
    keys = list(data_group.keys())
    rgb_keys = []
    for k in keys:
        if k.startswith("camera") and k.endswith("_rgb"):
            rgb_keys.append(k)
    return sorted(rgb_keys, key=natural_key)


def build_episode_ranges(episode_ends: np.ndarray) -> List[Tuple[int, int]]:
    episode_ends = np.asarray(episode_ends, dtype=int)
    starts = np.concatenate(([0], episode_ends[:-1]))
    ranges = [(int(s), int(e)) for s, e in zip(starts, episode_ends)]
    return ranges


def locate_episode(frame_idx: int, ep_ranges: List[Tuple[int, int]]) -> Tuple[int, int]:
    # returns (ep_idx, local_idx)
    # linear scan is fine for typical sizes; can binary-search if huge
    for i, (s, e) in enumerate(ep_ranges):
        if s <= frame_idx < e:
            return i, frame_idx - s
    # out of bounds
    if frame_idx < 0:
        return 0, 0
    return len(ep_ranges) - 1, max(0, frame_idx - ep_ranges[-1][0])


def clamp(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", required=True, help="Path to dataset .zarr.zip")
    ap.add_argument("--camera", default=None,
                    help="RGB key to show (e.g. camera0_rgb). If not set, uses first camera*_rgb.")
    ap.add_argument("--scale", type=float, default=1.0, help="Display scale factor (e.g. 0.5)")
    ap.add_argument("--start", type=int, default=0, help="Start absolute frame index")
    ap.add_argument("--fps", type=float, default=20.0, help="Initial autoplay FPS")
    ap.add_argument("--window", default="zarr_rgb_viewer", help="OpenCV window name")
    args = ap.parse_args()

    with zarr.ZipStore(args.zarr, mode="r") as store:
        # Use ReplayBuffer for consistency with diffusion_policy datasets
        rb = ReplayBuffer.copy_from_store(src_store=store, store=zarr.MemoryStore())
        data = rb.data
        meta = rb.meta

    if "episode_ends" not in meta:
        raise KeyError("meta/episode_ends not found in this replay buffer.")
    episode_ends = np.asarray(meta["episode_ends"][:], dtype=int)
    T = int(episode_ends[-1]) if len(episode_ends) > 0 else 0
    if T <= 0:
        raise RuntimeError("Dataset has zero timesteps.")

    ep_ranges = build_episode_ranges(episode_ends)

    rgb_keys = find_rgb_keys(data)
    if len(rgb_keys) == 0:
        raise RuntimeError("No camera*_rgb keys found in rb.data. Available keys:\n  " + "\n  ".join(sorted(data.keys())))

    rgb_key = args.camera if args.camera is not None else rgb_keys[0]
    if rgb_key not in data:
        raise KeyError(f"Requested camera key '{rgb_key}' not found. Options: {rgb_keys}")

    arr = data[rgb_key]
    # Expect [T, H, W, 3] uint8
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"{rgb_key} has unexpected shape {arr.shape}. Expected [T,H,W,3].")

    print("Loaded:")
    print(f"  zarr: {args.zarr}")
    print(f"  key : {rgb_key}")
    print(f"  shape: {arr.shape}, dtype={arr.dtype}")
    print(f"  episodes: {len(ep_ranges)}, total frames: {T}")
    if len(rgb_keys) > 1:
        print(f"  other rgb keys: {rgb_keys}")

    idx = clamp(args.start, 0, T - 1)
    autoplay = False
    fps = max(1e-3, float(args.fps))
    delay_s = 1.0 / fps

    cv2.namedWindow(args.window, cv2.WINDOW_NORMAL)

    def show_frame(i: int):
        frame = arr[i]  # (H,W,3), RGB
        # OpenCV expects BGR
        bgr = frame[..., ::-1]

        if args.scale != 1.0:
            h, w = bgr.shape[:2]
            bgr = cv2.resize(bgr, (int(w * args.scale), int(h * args.scale)), interpolation=cv2.INTER_NEAREST)

        ep_i, local_i = locate_episode(i, ep_ranges)
        s, e = ep_ranges[ep_i]
        txt = f"frame {i}/{T-1} | ep {ep_i}/{len(ep_ranges)-1} (local {local_i}/{(e-s)-1}) | {rgb_key} | fps={fps:.1f} {'PLAY' if autoplay else 'PAUSE'}"
        cv2.putText(
            bgr, txt, (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
        )
        cv2.imshow(args.window, bgr)

    last_step_time = time.time()
    while True:
        show_frame(idx)

        # Use small waitKey to allow UI events; 1ms is fine
        key = cv2.waitKey(1) & 0xFF

        # Handle keypresses
        if key in (ord('q'), 27):  # q or ESC
            break
        elif key in (ord('d'), 83):  # d or right arrow (sometimes 83)
            idx = clamp(idx + 1, 0, T - 1)
        elif key in (ord('a'), 81):  # a or left arrow (sometimes 81)
            idx = clamp(idx - 1, 0, T - 1)
        elif key == ord(' '):  # space toggles autoplay
            autoplay = not autoplay
            last_step_time = time.time()
        elif key == ord(']'):
            fps = min(240.0, fps * 1.25)
            delay_s = 1.0 / fps
        elif key == ord('['):
            fps = max(0.5, fps / 1.25)
            delay_s = 1.0 / fps
        elif key == ord('n'):
            ep_i, _ = locate_episode(idx, ep_ranges)
            ep_i = clamp(ep_i + 1, 0, len(ep_ranges) - 1)
            idx = ep_ranges[ep_i][0]
        elif key == ord('p'):
            ep_i, _ = locate_episode(idx, ep_ranges)
            ep_i = clamp(ep_i - 1, 0, len(ep_ranges) - 1)
            idx = ep_ranges[ep_i][0]
        elif key == ord('j'):
            # terminal prompt (blocking)
            try:
                val = input("Jump to absolute frame index: ").strip()
                if val:
                    idx = clamp(int(val), 0, T - 1)
            except Exception as ex:
                print(f"Jump failed: {ex}")

        # Autoplay step based on time
        if autoplay:
            now = time.time()
            if (now - last_step_time) >= delay_s:
                idx += 1
                if idx >= T:
                    idx = T - 1
                    autoplay = False
                last_step_time = now

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()