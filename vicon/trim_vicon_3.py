import os, re, glob
import pandas as pd
from io import StringIO

csv_dir = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon/vicon_quat_trimmed_2/"
out_dir = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon/vicon_quat_trimmed_3/"
os.makedirs(out_dir, exist_ok=True)

TRIM_START = 100  # keep data from this index onward (discard first 100 rows)

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
    return header_lines, df_raw

def save_trimmed_exact(out_path, header_lines, df_raw_trim):
    with open(out_path, "w", encoding="utf-8", errors="ignore") as f:
        for ln in header_lines:
            f.write(ln)
        df_raw_trim.to_csv(f, index=False, header=False)

for csv_path in csv_files:
    base = os.path.basename(csv_path)
    out_path = os.path.join(out_dir, base)

    header_lines, df_raw = load_raw_preserve_exact(csv_path)

    n = len(df_raw)
    if n <= TRIM_START:
        print(f"[WARN] {base}: only {n} rows, trimming at {TRIM_START} leaves 0 rows.")
        df_raw_trim = df_raw.iloc[0:0].copy()
    else:
        df_raw_trim = df_raw.iloc[TRIM_START:].reset_index(drop=True)

    save_trimmed_exact(out_path, header_lines, df_raw_trim)
    print(f"Saved: {out_path} (rows: {n} -> {len(df_raw_trim)})")