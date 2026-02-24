import os, re, glob
import numpy as np
import pandas as pd

csv_dir = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon/vicon_quat_trimmed/"
out_dir = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon/vicon_quat_trimmed_2/"
os.makedirs(out_dir, exist_ok=True)

def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

csv_files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")), key=natural_key)

def choose_trim_index_min_yz(df_num, y_col="Pos_Y", z_col="Pos_Z"):
    """
    Picks the row index whose (y,z) is closest to (min(y), min(z)).
    Returns trim_end as a 0-based DATA-ROW index (0 = first row after header).
    """
    y = pd.to_numeric(df_num[y_col], errors="coerce")
    z = pd.to_numeric(df_num[z_col], errors="coerce")

    valid = y.notna() & z.notna()
    if not valid.any():
        raise RuntimeError(f"No valid numeric rows found in {y_col}/{z_col}.")

    yv = y[valid].to_numpy()
    zv = z[valid].to_numpy()

    y_min = float(np.min(yv))
    z_min = float(np.min(zv))

    score = (yv - y_min) ** 2 + (zv - z_min) ** 2
    i_valid = int(np.argmin(score))

    # Map back to original dataframe (data-row) index
    trim_end = int(df_num.index[valid][i_valid])
    return trim_end, y_min, z_min

for csv_path in csv_files:
    base = os.path.basename(csv_path)
    out_path = os.path.join(out_dir, base)

    # 1) Read numeric dataframe (normal CSV with header row)
    df_num = pd.read_csv(csv_path)

    # 2) Compute trim index using Pos_Y / Pos_Z
    trim_end, y_min, z_min = choose_trim_index_min_yz(df_num, "Pos_Y", "Pos_Z")

    # 3) Preserve exact original formatting by slicing raw lines:
    #    - lines[0] is header
    #    - data row k is lines[1 + k]
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    header = lines[0]
    data_lines = lines[1:]

    # Keep everything BEFORE trim_end (exclude trim_end)
    kept_data = data_lines[:trim_end]

    with open(out_path, "w", encoding="utf-8", errors="ignore") as f:
        f.write(header)
        f.writelines(kept_data)

    print("\n==============================")
    print("File:", base)
    print(f"Chosen trim_end (data-row index): {trim_end} based on y_min={y_min}, z_min={z_min}")
    print("Kept data rows:", len(kept_data))
    print("Saved:", out_path)