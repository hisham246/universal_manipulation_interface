import pickle

pkl_file = "/home/hisham246/uwaterloo/umi/surface_wiping_trial_1/slam_trial_1/dataset_plan.pkl"
with open(pkl_file, "rb") as f:
    dataset_plan = pickle.load(f)

# print(dataset_plan[0].keys())

print(dataset_plan[0]["episode_timestamps"])

# for i, episode in enumerate(dataset_plan):
#     timestamps = episode["episode_timestamps"]
    # print(f"Episode {i}:")
    # print(f"  Start time: {timestamps[0]:.3f} s")
    # print(f"  End time:   {timestamps[-1]:.3f} s")
    # print(f"  Duration:   {timestamps[-1] - timestamps[0]:.3f} s")
    # print(f"  Num frames: {len(timestamps)}\n")