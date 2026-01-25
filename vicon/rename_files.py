#!/usr/bin/env python3
import os
import re
import sys
import glob
import uuid

csv_dir = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon/vicon/"

# Matches: "peg_umi 123.csv"
PATTERN = re.compile(r"^peg_umi_quat\s+(\d+)\.csv$", re.IGNORECASE)

def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def main() -> int:
    # Allow overriding the directory via CLI, otherwise use the hardcoded one
    directory = sys.argv[1] if len(sys.argv) > 1 else csv_dir
    if not os.path.isdir(directory):
        print(f"Error: '{directory}' is not a directory.")
        return 1

    # Your retrieval snippet (glob + natural sort)
    csv_files = sorted(glob.glob(os.path.join(directory, "*.csv")), key=natural_key)

    # Build index -> filepath map, but only for peg_umi <n>.csv files
    files = {}  # {idx: fullpath}
    for fp in csv_files:
        name = os.path.basename(fp)
        m = PATTERN.match(name)
        if not m:
            continue
        idx = int(m.group(1))
        files[idx] = fp

    if not files:
        print("No files matching 'peg_umi_quat <n>.csv' found.")
        return 1

    # Expect: 118 missing, and files exist for 1..117 and 119..250
    if 118 in files:
        print("Error: 'peg_umi 118.csv' exists. This script assumes 118 is missing.")
        return 1

    # Rename plan: old_idx >= 119 -> new_idx = old_idx - 1 (119->118 ... 250->249)
    plan = []
    for old_idx in sorted(files.keys()):
        if old_idx <= 117:
            continue
        if old_idx >= 119:
            src_path = files[old_idx]
            new_idx = old_idx - 1
            dst_name = f"peg_umi_{new_idx}.csv"
            dst_path = os.path.join(directory, dst_name)
            plan.append((src_path, dst_path))

    if not plan:
        print("Nothing to rename (no files >= 119 found).")
        return 0

    # Safety checks
    target_paths = [dst for _, dst in plan]
    if len(target_paths) != len(set(target_paths)):
        print("Error: rename plan has duplicate targets. Aborting.")
        return 1

    existing_names = {os.path.basename(p) for p in csv_files}
    move_set_sources = {os.path.basename(src) for src, _ in plan}
    conflicts = []
    for _, dst in plan:
        dst_name = os.path.basename(dst)
        if dst_name in existing_names and dst_name not in move_set_sources:
            conflicts.append(dst_name)

    if conflicts:
        print("Error: these target filenames already exist and are not part of the move set:")
        for n in sorted(set(conflicts), key=natural_key):
            print("  ", n)
        print("Aborting.")
        return 1

    print(f"Directory: {directory}")
    print(f"Planned renames: {len(plan)}")
    for src, dst in plan[:10]:
        print(f"  {os.path.basename(src)} -> {os.path.basename(dst)}")
    if len(plan) > 10:
        print("  ...")

    # Two-step rename to avoid collisions
    token = uuid.uuid4().hex[:8]
    temp_map = []

    # Step 1: rename to temp files
    for src, dst in plan:
        src_name = os.path.basename(src)
        tmp_name = f".__tmp__{token}__{src_name}"
        tmp_path = os.path.join(directory, tmp_name)
        os.rename(src, tmp_path)
        temp_map.append((tmp_path, dst))

    # Step 2: rename temps to final destinations
    for tmp, dst in temp_map:
        os.rename(tmp, dst)

    print("Done. Files are now numbered 1..249 (for the matched set).")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
