import pickle
import pathlib
import pprint
import sys

ipath = pathlib.Path(sys.argv[1]).expanduser().resolve()
plan_path = ipath / "dataset_plan.pkl"
plan = pickle.load(plan_path.open("rb"))

i = 10  # episode index you want

ep = plan[i]
print(f"--- Episode {i} ---")
print("Type:", type(ep))

if isinstance(ep, dict):
    print("Keys:", sorted(ep.keys()))
    print("\nFull episode dict (depth-limited):")
    pprint.pprint(ep, depth=8, compact=True, width=140)

    # Optional: show just the big parts in a readable way
    if "grippers" in ep:
        print(f"\nGrippers: {len(ep['grippers'])}")
        for gidx, g in enumerate(ep["grippers"]):
            print(f"  gripper[{gidx}] keys:", sorted(g.keys()))
            if "tcp_pose" in g:
                print("    tcp_pose shape:", getattr(g["tcp_pose"], "shape", None))
            if "gripper_width" in g:
                print("    gripper_width shape:", getattr(g["gripper_width"], "shape", None))

    if "cameras" in ep:
        print(f"\nCameras: {len(ep['cameras'])}")
        for cidx, c in enumerate(ep["cameras"]):
            print(f"  camera[{cidx}] keys:", sorted(c.keys()))
            if "video_path" in c:
                print("    video_path:", c["video_path"])
            if "video_start_end" in c:
                print("    video_start_end:", c["video_start_end"])
else:
    pprint.pprint(ep, depth=8, width=140)
