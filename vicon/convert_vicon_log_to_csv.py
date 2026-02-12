import os, re, glob, csv

def natural_key(path: str):
    # Sort like: file_2.log < file_10.log
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', os.path.basename(path))]

src_path = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon_v3/vicon_logs/"
dst_path = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon_v3/vicon_logs_to_csv/"
os.makedirs(dst_path, exist_ok=True)

headers = [
    "Timestamp", "Frame", "Object_Name",
    "Pos_X", "Pos_Y", "Pos_Z",
    "Rot_X", "Rot_Y", "Rot_Z", "Rot_W"
]

# Loop over .log files (like your attached vicon_257.log)
files = sorted(glob.glob(os.path.join(src_path, "*.log")), key=natural_key)

# Simple check: line starts with a float timestamp
ts_line_re = re.compile(r"^\s*-?\d+(?:\.\d+)?\s*,")

for in_file in files:
    base = os.path.splitext(os.path.basename(in_file))[0]
    out_file = os.path.join(dst_path, f"{base}.csv")

    with open(in_file, "r", errors="ignore") as f_in, open(out_file, "w", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(headers)

        for line in f_in:
            line = line.strip()
            if not line:
                continue

            # Skip any non-data lines (if your logs ever contain headers/messages)
            if not ts_line_re.match(line):
                continue

            # Your .log lines look like:
            # 1770824938.651, 1246412, umi_gripper_peg_in_hole, ..., 0.99989...,   (often trailing comma)
            row = [field.strip() for field in line.split(",") if field.strip() != ""]

            # Keep only the first 10 fields (drop any extra caused by trailing commas)
            if len(row) >= len(headers):
                row = row[:len(headers)]
                writer.writerow(row)
            # If some lines are shorter, you can either skip or pad; here we skip
            # else:
            #     continue

    print(f"Wrote: {out_file}")
