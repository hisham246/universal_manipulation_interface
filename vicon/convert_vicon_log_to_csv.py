import os, re, glob, csv

def natural_key(path: str):
    return [int(s) if s.isdigit() else s.lower()
            for s in re.split(r'(\d+)', os.path.basename(path))]

src_path = "/home/hisham246/uwaterloo/cable_route_umi_with_vicon/vicon_logs/"
dst_path = "/home/hisham246/uwaterloo/cable_route_umi_with_vicon/vicon_logs_to_csv/"
os.makedirs(dst_path, exist_ok=True)

# fields after Object_Name in the log
pose_fields = ["Pos_X", "Pos_Y", "Pos_Z", "Rot_X", "Rot_Y", "Rot_Z", "Rot_W"]

files = sorted(glob.glob(os.path.join(src_path, "*.log")), key=natural_key)

# Simple check: line starts with a float timestamp
ts_line_re = re.compile(r"^\s*-?\d+(?:\.\d+)?\s*,")

def discover_objects(log_path, max_lines=5000):
    """Find unique Object_Name values (usually 2) by scanning the file head."""
    objs = []
    seen = set()
    with open(log_path, "r", errors="ignore") as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            line = line.strip()
            if not line or not ts_line_re.match(line):
                continue
            parts = [p.strip() for p in line.split(",") if p.strip() != ""]
            if len(parts) < 3:
                continue
            obj = parts[2]
            if obj not in seen:
                seen.add(obj)
                objs.append(obj)
            if len(objs) >= 2:   # stop early if you only expect 2
                break
    return objs

for in_file in files:
    base = os.path.splitext(os.path.basename(in_file))[0]
    out_file = os.path.join(dst_path, f"{base}.csv")

    # Option A: auto-detect the two objects from the file
    objects = discover_objects(in_file)

    # Option B (recommended if you want strict ordering):
    # objects = ["cable_station", "umi_cable_route"]

    if len(objects) < 2:
        print(f"[WARN] Found <2 objects in {in_file}: {objects}. Will still write what exists.")

    # Build CSV header: Timestamp, Frame, then per-object pose fields
    headers = ["Timestamp", "Frame"]
    for obj in objects:
        headers += [f"{obj}_{fld}" for fld in pose_fields]

    with open(in_file, "r", errors="ignore") as f_in, open(out_file, "w", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(headers)

        current_key = None  # (Timestamp, Frame)
        row_data = {}       # obj -> [pos/quat 7 values]

        def flush():
            """Write buffered frame to CSV (pad missing objects with blanks)."""
            if current_key is None:
                return
            ts, frame = current_key
            out_row = [ts, frame]
            for obj in objects:
                vals = row_data.get(obj, [""] * len(pose_fields))
                out_row.extend(vals)
            writer.writerow(out_row)

        for line in f_in:
            line = line.strip()
            if not line or not ts_line_re.match(line):
                continue

            parts = [p.strip() for p in line.split(",") if p.strip() != ""]
            # expected: ts, frame, obj, 7 pose fields
            if len(parts) < 3 + len(pose_fields):
                continue

            ts = parts[0]
            frame = parts[1]
            obj = parts[2]
            pose = parts[3:3 + len(pose_fields)]

            key = (ts, frame)

            # new frame -> flush previous
            if current_key is not None and key != current_key:
                flush()
                row_data = {}

            current_key = key
            row_data[obj] = pose

        # flush last buffered frame
        flush()

    print(f"Wrote: {out_file}")