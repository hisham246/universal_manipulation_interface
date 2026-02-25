from pathlib import Path
import pandas as pd
import numpy as np

EP = Path("/home/hisham246/uwaterloo/peg_in_hole_delta_umi/timestamps/episode_4.csv")
VF = Path("/home/hisham246/uwaterloo/peg_in_hole_delta_umi/vicon_logs_to_csv/vicon_4.csv")

ep = pd.read_csv(EP)["timestamp"].to_numpy(np.float64)
vd = pd.read_csv(VF, usecols=["Timestamp"])["Timestamp"].to_numpy(np.float64)

print("EP start/end:", ep[0], ep[-1], "dur(s):", ep[-1]-ep[0])
print("V  start/end:", vd.min(), vd.max(), "dur(s):", vd.max()-vd.min())

print("EP - V start (s):", ep[0] - vd.min())
print("EP - V end   (s):", ep[-1] - vd.max())

# how close to 1 day?
d = np.median(ep) - np.median(vd)
print("median offset (s):", d, " offset/86400:", d/86400)