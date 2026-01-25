import os, re, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

csv_dir = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon/vicon_quat/"
out_dir = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon/vicon_quat_trimmed_2/"
os.makedirs(out_dir, exist_ok=True)

SMOOTH_WIN = 5  # only for viewing

def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

csv_files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")), key=natural_key)

def find_header_idx(lines):
    # Find the line that looks like the CSV column header
    # (must contain Frame, TX, TY, TZ)
    for i, ln in enumerate(lines):
        low = ln.lower()
        if ("frame" in low) and ("tx" in low) and ("ty" in low) and ("tz" in low):
            return i
    raise RuntimeError("Could not find header line containing Frame, TX, TY, TZ")

def load_raw_preserve_exact(csv_path):
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    hdr_idx = find_header_idx(lines)

    # Assume next line is units row, and data starts after that.
    # If your file ever has no units row, change data_start = hdr_idx + 1.
    data_start = hdr_idx + 2

    header_lines = lines[:data_start]           # includes original header + units, no data rows
    colnames = [c.strip() for c in lines[hdr_idx].strip().split(",")]

    data_lines = lines[data_start:]
    df_raw = pd.read_csv(
        StringIO("".join(data_lines)),
        names=colnames,          # IMPORTANT: first data row is NOT treated as header
        header=None,
        dtype=str
    )

    return header_lines, df_raw, colnames, data_start

def load_numeric_for_plot(csv_path, colnames, data_start):
    df_num = pd.read_csv(
        csv_path,
        skiprows=list(range(data_start)),  # skip metadata + header + units
        names=colnames,
        header=None
    )
    for c in ["TX","TY","TZ"]:
        df_num[c] = pd.to_numeric(df_num[c], errors="coerce")
    for c in ["TX","TY","TZ"]:
        df_num[c+"_s"] = df_num[c].rolling(SMOOTH_WIN, center=True, min_periods=1).median()
    dx = df_num["TX_s"].diff()
    dy = df_num["TY_s"].diff()
    speed = np.sqrt(dx**2 + dy**2).fillna(0)
    return df_num, speed

def choose_trim_index_click(df_num, speed, title):
    chosen = {"idx": None}
    vlines = []

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(12, 8))
    axes[0].plot(df_num["TX_s"].to_numpy()); axes[0].set_ylabel("TX (mm)")
    axes[1].plot(df_num["TY_s"].to_numpy()); axes[1].set_ylabel("TY (mm)")
    axes[2].plot(df_num["TZ_s"].to_numpy()); axes[2].set_ylabel("TZ (mm)")
    axes[3].plot(speed.to_numpy());         axes[3].set_ylabel("Speed"); axes[3].set_xlabel("Frame index")
    axes[0].set_title(title)

    def onclick(event):
        if event.inaxes != axes[3] or event.xdata is None:
            return
        idx = int(round(event.xdata))
        idx = max(0, min(idx, len(df_num) - 1))
        chosen["idx"] = idx

        for ln in vlines:
            ln.remove()
        vlines.clear()

        for ax in axes:
            vlines.append(ax.axvline(idx, linestyle="--"))

        fig.canvas.draw_idle()
        print("Chosen trim_start index:", idx)

    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.tight_layout()
    plt.show()

    if chosen["idx"] is None:
        raise RuntimeError("No trim point selected. Click on the speed plot.")
    return chosen["idx"]

def save_trimmed_exact(out_path, header_lines, df_raw_trim):
    with open(out_path, "w", encoding="utf-8", errors="ignore") as f:
        # write original metadata + original header + original units exactly
        for ln in header_lines:
            f.write(ln)

        # write only data rows (NO header row here, because it's already in header_lines)
        df_raw_trim.to_csv(f, index=False, header=False)

for csv_path in csv_files:
    base = os.path.basename(csv_path)
    out_path = os.path.join(out_dir, base)

    print("\n==============================")
    print("File:", base)

    header_lines, df_raw, colnames, data_start = load_raw_preserve_exact(csv_path)
    df_num, speed = load_numeric_for_plot(csv_path, colnames, data_start)

    trim_start = choose_trim_index_click(
        df_num, speed,
        title=f"{base} — click SPEED (bottom) to choose trim_start, then close window"
    )

    # Trim using the SAME row indexing for raw and numeric (they now align by construction)
    df_raw_trim = df_raw.iloc[trim_start:].reset_index(drop=True)

    save_trimmed_exact(out_path, header_lines, df_raw_trim)
    print("Saved:", out_path)
