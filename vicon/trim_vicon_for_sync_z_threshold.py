#!/usr/bin/env python3
import os, re, glob
import numpy as np
import pandas as pd

# =========================
# CONFIG
# =========================
csv_dir = "/home/hisham246/uwaterloo/peg_in_hole_delta_umi/vicon_trimmed/"
out_dir = "/home/hisham246/uwaterloo/peg_in_hole_delta_umi/vicon_trimmed_2/"
os.makedirs(out_dir, exist_ok=True)

# column names in your CSV
Z_COL = "Pos_Z"

# Trimming mode:
# - False (recommended): keep rows BEFORE the min-Z index (exclude the min-Z row itself)
# - True: keep rows FROM the min-Z index to the end
KEEP_AFTER_MINZ = False

# Robustness options
IGNORE_NANS = True          # ignore NaN Z values when finding min
MIN_PREFIX_ROWS = 0         # if >0, only search for min-Z after this many initial rows (optional)
MIN_SUFFIX_ROWS = 0         # if >0, only search for min-Z before the last N rows (optional)

# If True, save exact original lines (preserves formatting) like your manual script.
# If False, re-write via pandas (may change float formatting).
PRESERVE_EXACT_LINES = True


# =========================
# HELPERS
# =========================
def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def load_lines(csv_path):
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    if len(lines) < 2:
        raise RuntimeError(f"{csv_path} has no data rows.")
    return lines[0], lines[1:]  # header, data_lines

def compute_min_z_index(csv_path):
    df = pd.read_csv(csv_path)

    if Z_COL not in df.columns:
        raise RuntimeError(f"Missing column '{Z_COL}' in {csv_path}. Columns: {list(df.columns)}")

    z = pd.to_numeric(df[Z_COL], errors="coerce")

    # define search window
    n = len(z)
    start = int(MIN_PREFIX_ROWS)
    end = n - int(MIN_SUFFIX_ROWS)
    start = max(0, min(start, n))
    end = max(start, min(end, n))

    z_win = z.iloc[start:end]

    if IGNORE_NANS:
        if z_win.notna().sum() == 0:
            raise RuntimeError(f"All Z values are NaN in search window for {csv_path}")
        local_idx = int(z_win.idxmin(skipna=True))
    else:
        if z_win.isna().any():
            raise RuntimeError(f"NaNs present in Z within search window for {csv_path} and IGNORE_NANS=False")
        local_idx = int(z_win.idxmin())

    # local_idx is the dataframe index (same as row index if default range)
    # Ensure it maps to 0..n-1 position
    # For standard CSVs, df index is 0..n-1, so:
    minz_pos = local_idx
    if not (0 <= minz_pos < n):
        # fallback: convert to positional index
        minz_pos = int(z.reset_index(drop=True).iloc[start:end].values.argmin()) + start

    minz_val = float(z.iloc[minz_pos]) if pd.notna(z.iloc[minz_pos]) else np.nan
    return minz_pos, minz_val, n

def save_trimmed_exact(out_path, header, data_lines_trimmed):
    with open(out_path, "w", encoding="utf-8", errors="ignore") as f:
        f.write(header)
        f.writelines(data_lines_trimmed)


# =========================
# MAIN
# =========================
csv_files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")), key=natural_key)

for csv_path in csv_files:
    base = os.path.basename(csv_path)
    out_path = os.path.join(out_dir, base)

    header, data_lines = load_lines(csv_path)

    minz_idx, minz_val, nrows = compute_min_z_index(csv_path)

    # data_lines corresponds to rows [0..nrows-1] (excluding header)
    if len(data_lines) != nrows:
        # If your files sometimes have blank lines, this warns you that "exact line slicing"
        # might not match pandas row count 1:1.
        print(f"[WARN] {base}: pandas rows={nrows}, raw data lines={len(data_lines)} (blank lines?)")

    if KEEP_AFTER_MINZ:
        trimmed = data_lines[minz_idx:]         # keep from min-Z row onward
        kept_range = f"[{minz_idx} .. end]"
    else:
        trimmed = data_lines[:minz_idx]         # keep before min-Z row (exclude min-Z row)
        kept_range = f"[0 .. {minz_idx-1}]"

    if PRESERVE_EXACT_LINES:
        save_trimmed_exact(out_path, header, trimmed)
    else:
        # re-write via pandas (format may change)
        df = pd.read_csv(csv_path)
        if KEEP_AFTER_MINZ:
            df_out = df.iloc[minz_idx:].copy()
        else:
            df_out = df.iloc[:minz_idx].copy()
        df_out.to_csv(out_path, index=False)

    print(f"{base}: minZ={minz_val:.6f} at idx={minz_idx} | kept {len(trimmed)} / {len(data_lines)} rows {kept_range} -> {out_path}")