import os, re, glob
import numpy as np
import pandas as pd
from io import StringIO

csv_dir = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon/vicon_quat_trimmed/"
out_dir = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon/vicon_quat_trimmed_2/"
os.makedirs(out_dir, exist_ok=True)

def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

csv_files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")), key=natural_key)

def find_header_idx(lines):
    for i, ln in enumerate(lines):
        low = ln.lower()
        if ("frame" in low) and ("tx" in low) and ("ty" in low) and ("tz" in low):
            return i
    raise RuntimeError("Could not find header line containing Frame, TX, TY, TZ")

def load_raw_preserve_exact(csv_path):
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    hdr_idx = find_header_idx(lines)
    data_start = hdr_idx + 2  # header + units row

    header_lines = lines[:data_start]
    colnames = [c.strip() for c in lines[hdr_idx].strip().split(",")]

    data_lines = lines[data_start:]
    df_raw = pd.read_csv(
        StringIO("".join(data_lines)),
        names=colnames,
        header=None,
        dtype=str
    )
    return header_lines, df_raw, colnames, data_start

def load_numeric_ty_tz(csv_path, colnames, data_start):
    df_num = pd.read_csv(
        csv_path,
        skiprows=list(range(data_start)),
        names=colnames,
        header=None
    )
    df_num["TY"] = pd.to_numeric(df_num["TY"], errors="coerce")
    df_num["TZ"] = pd.to_numeric(df_num["TZ"], errors="coerce")
    return df_num

def choose_trim_index_min_yz(df_num):
    """
    Picks the index whose (TY, TZ) is closest to (min(TY), min(TZ)).
    This is usually what people mean by "minimum y and z point" when minima
    don't occur at the exact same frame.
    """
    valid = df_num[["TY", "TZ"]].notna().all(axis=1)
    if not valid.any():
        raise RuntimeError("No valid numeric TY/TZ rows found.")

    y = df_num.loc[valid, "TY"].to_numpy()
    z = df_num.loc[valid, "TZ"].to_numpy()

    y_min = np.min(y)
    z_min = np.min(z)

    # distance to the (y_min, z_min) point
    score = (y - y_min) ** 2 + (z - z_min) ** 2
    i_valid = np.argmin(score)

    # map back to original dataframe row index
    trim_idx = df_num.loc[valid].index[i_valid]
    return int(trim_idx), float(y_min), float(z_min)

def save_trimmed_exact(out_path, header_lines, df_raw_trim):
    with open(out_path, "w", encoding="utf-8", errors="ignore") as f:
        for ln in header_lines:
            f.write(ln)
        df_raw_trim.to_csv(f, index=False, header=False)

for csv_path in csv_files:
    base = os.path.basename(csv_path)
    out_path = os.path.join(out_dir, base)

    header_lines, df_raw, colnames, data_start = load_raw_preserve_exact(csv_path)
    df_num = load_numeric_ty_tz(csv_path, colnames, data_start)

    trim_end, y_min, z_min = choose_trim_index_min_yz(df_num)

    # Keep everything BEFORE trim_end, discard trim_end and after
    df_raw_trim = df_raw.iloc[:trim_end].reset_index(drop=True)

    save_trimmed_exact(out_path, header_lines, df_raw_trim)

    print("\n==============================")
    print("File:", base)
    print(f"Chosen trim_end index: {trim_end} (based on TY/TZ minima target y_min={y_min}, z_min={z_min})")
    print("Saved:", out_path)
