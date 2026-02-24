import os, re, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# csv_dir = "/home/hisham246/uwaterloo/cable_route_umi/aligned_vicon_files/aligned_vicon_to_episode/"
csv_dir = "/home/hisham246/uwaterloo/cable_route_umi/vicon_trimmed/"
out_dir = "/home/hisham246/uwaterloo/cable_route_umi/vicon_trimmed_2/"
os.makedirs(out_dir, exist_ok=True)

# for the speed plot only (does not affect saved values)
SMOOTH_WIN = 5

# choose trimming mode
KEEP_AFTER_CLICK = False   # True: keep rows from clicked idx to end
                          # False: keep rows before clicked idx (exclude clicked idx)

# columns to use for plotting/speed (adjust if your file uses different names)
X_COL, Y_COL, Z_COL = "Pos_X", "Pos_Y", "Pos_Z"

def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

csv_files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")), key=natural_key)

def load_lines_and_numeric(csv_path):
    # raw lines (for exact-preserving output)
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    if len(lines) < 2:
        raise RuntimeError(f"{csv_path} has no data rows.")

    header = lines[0]
    data_lines = lines[1:]

    # numeric df for plotting only
    df = pd.read_csv(csv_path)
    for c in [X_COL, Y_COL, Z_COL]:
        if c not in df.columns:
            raise RuntimeError(f"Missing column '{c}' in {csv_path}. Columns: {list(df.columns)}")

    x = pd.to_numeric(df[X_COL], errors="coerce")
    y = pd.to_numeric(df[Y_COL], errors="coerce")
    z = pd.to_numeric(df[Z_COL], errors="coerce")

    # smoothing only for visualization
    xs = x.rolling(SMOOTH_WIN, center=True, min_periods=1).median()
    ys = y.rolling(SMOOTH_WIN, center=True, min_periods=1).median()
    zs = z.rolling(SMOOTH_WIN, center=True, min_periods=1).median()

    # 3D speed (frame-to-frame)
    dx = xs.diff()
    dy = ys.diff()
    dz = zs.diff()
    speed = np.sqrt(dx**2 + dy**2 + dz**2).fillna(0.0)

    plot_df = pd.DataFrame({
        "x": xs.to_numpy(),
        "y": ys.to_numpy(),
        "z": zs.to_numpy(),
        "speed": speed.to_numpy(),
    })

    # sanity: data_lines count should match df rows in most cases
    # if your CSV sometimes has blank lines, keep output slicing based on data_lines length
    return header, data_lines, plot_df

def choose_trim_index_click(plot_df, title):
    chosen = {"idx": None}
    vlines = []

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(12, 8))
    axes[0].plot(plot_df["x"]);     axes[0].set_ylabel(X_COL)
    axes[1].plot(plot_df["y"]);     axes[1].set_ylabel(Y_COL)
    axes[2].plot(plot_df["z"]);     axes[2].set_ylabel(Z_COL)
    axes[3].plot(plot_df["speed"]); axes[3].set_ylabel("Speed"); axes[3].set_xlabel("Row index (0-based)")
    axes[0].set_title(title)

    def onclick(event):
        if event.inaxes != axes[3] or event.xdata is None:
            return
        idx = int(round(event.xdata))
        idx = max(0, min(idx, len(plot_df) - 1))
        chosen["idx"] = idx

        for ln in vlines:
            ln.remove()
        vlines.clear()

        for ax in axes:
            vlines.append(ax.axvline(idx, linestyle="--"))

        fig.canvas.draw_idle()
        print("Chosen trim index:", idx)

    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.tight_layout()
    plt.show()

    if chosen["idx"] is None:
        raise RuntimeError("No trim point selected. Click on the speed plot, then close the window.")
    return chosen["idx"]

def save_trimmed_exact(out_path, header, data_lines_trimmed):
    with open(out_path, "w", encoding="utf-8", errors="ignore") as f:
        f.write(header)
        f.writelines(data_lines_trimmed)


# for csv_path in csv_files:
#     base = os.path.basename(csv_path)
#     out_path = os.path.join(out_dir, base)

#     print("\n==============================")
#     print("File:", base)

#     header, data_lines, plot_df = load_lines_and_numeric(csv_path)

#     trim_idx = choose_trim_index_click(
#         plot_df,
#         title=f"{base} — click SPEED (bottom) to choose trim index, then close window"
#     )

#     if KEEP_AFTER_CLICK:
#         # keep clicked idx and everything after
#         trimmed = data_lines[trim_idx:]
#     else:
#         # keep everything before clicked idx (exclude clicked idx)
#         trimmed = data_lines[:trim_idx]

#     save_trimmed_exact(out_path, header, trimmed)
#     print("Saved:", out_path, "| kept rows:", len(trimmed), "of", len(data_lines))

# csv_path = '/home/hisham246/uwaterloo/cable_route_umi/aligned_vicon_files/aligned_vicon_to_episode/aligned_episode_073.csv'
csv_path = '/home/hisham246/uwaterloo/cable_route_umi/vicon_trimmed/aligned_episode_231.csv'

base = os.path.basename(csv_path)
out_path = os.path.join(out_dir, base)

print("\n==============================")
print("File:", base)

header, data_lines, plot_df = load_lines_and_numeric(csv_path)

trim_idx = choose_trim_index_click(
    plot_df,
    title=f"{base} — click SPEED (bottom) to choose trim index, then close window"
)

if KEEP_AFTER_CLICK:
    # keep clicked idx and everything after
    trimmed = data_lines[trim_idx:]
else:
    # keep everything before clicked idx (exclude clicked idx)
    trimmed = data_lines[:trim_idx]

save_trimmed_exact(out_path, header, trimmed)
print("Saved:", out_path, "| kept rows:", len(trimmed), "of", len(data_lines))