import os
import pathlib
import argparse

def rename_videos(directory):
    directory = pathlib.Path(directory).resolve()
    if not directory.exists() or not directory.is_dir():
        print(f"Directory does not exist: {directory}")
        return

    mp4_files = sorted([f for f in directory.glob("*.mp4")] + [f for f in directory.glob("*.MP4")], key=lambda x: x.name)

    if not mp4_files:
        print(f"No MP4 files found in {directory}")
        return

    print(f"Found {len(mp4_files)} MP4 files. Renaming...")

    # Step 1: Rename all to temporary names to avoid name clashes
    temp_names = []
    for i, f in enumerate(mp4_files):
        tmp_name = directory / f"tmp_{i}.mp4"
        f.rename(tmp_name)
        temp_names.append(tmp_name)

    # Step 2: Rename to final numbered names
    for i, tmp_path in enumerate(temp_names, start=1):
        final_name = directory / f"{i}.mp4"
        tmp_path.rename(final_name)
        print(f"Renamed to {final_name.name}")

    print("Renaming complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename MP4 videos by filename order to 1.mp4, 2.mp4, ...")
    parser.add_argument("directory", help="Path to the directory containing videos")
    args = parser.parse_args()

    rename_videos(args.directory)
