"""
python scripts_slam_pipeline/00_process_videos.py data_workspace/toss_objects/20231113
"""
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import pathlib
import click
import shutil
from exiftool import ExifToolHelper
from umi.common.timecode_util import mp4_get_start_datetime

# %%
@click.command(help='Session directories. Assumming mp4 videos are in <session_dir>/raw_videos')
@click.argument('session_dir', nargs=-1)
def main(session_dir):
    for session in session_dir:
        session = pathlib.Path(os.path.expanduser(session)).absolute()
        # hardcode subdirs
        input_dir = session.joinpath('raw_videos')
        output_dir = session.joinpath('demos')
        
        # create raw_videos if don't exist
        if not input_dir.is_dir():
            input_dir.mkdir()
            print(f"{input_dir.name} subdir don't exits! Creating one and moving all mp4 videos inside.")
            for mp4_path in list(session.glob('**/*.MP4')) + list(session.glob('**/*.mp4')):
                out_path = input_dir.joinpath(mp4_path.name)
                shutil.move(mp4_path, out_path)
        
        # create mapping video if don't exist
        mapping_vid_path = input_dir.joinpath('mapping.mp4')
        if (not mapping_vid_path.exists()) and not(mapping_vid_path.is_symlink()):
            max_size = -1
            max_path = None
            for mp4_path in list(input_dir.glob('**/*.MP4')) + list(input_dir.glob('**/*.mp4')):
                size = mp4_path.stat().st_size
                if size > max_size:
                    max_size = size
                    max_path = mp4_path
            shutil.move(max_path, mapping_vid_path)
            print(f"raw_videos/mapping.mp4 don't exist! Renaming largest file {max_path.name}.")
        
        # create gripper calibration video if don't exist
        gripper_cal_dir = input_dir.joinpath('gripper_calibration')
        if not gripper_cal_dir.is_dir():
            gripper_cal_dir.mkdir()
            print("raw_videos/gripper_calibration don't exist! Creating one with the first video of each camera serial.")
            
            serial_start_dict = dict()
            serial_path_dict = dict()
            with ExifToolHelper() as et:
                for mp4_path in list(input_dir.glob('**/*.MP4')) + list(input_dir.glob('**/*.mp4')):
                    if mp4_path.name.startswith('map'):
                        continue
                    
                    start_date = mp4_get_start_datetime(str(mp4_path))
                    meta = list(et.get_metadata(str(mp4_path)))[0]
                    cam_serial = meta['QuickTime:CameraSerialNumber']
                    
                    if cam_serial in serial_start_dict:
                        if start_date < serial_start_dict[cam_serial]:
                            serial_start_dict[cam_serial] = start_date
                            serial_path_dict[cam_serial] = mp4_path
                    else:
                        serial_start_dict[cam_serial] = start_date
                        serial_path_dict[cam_serial] = mp4_path
            
            for serial, path in serial_path_dict.items():
                print(f"Selected {path.name} for camera serial {serial}")
                out_path = gripper_cal_dir.joinpath(path.name)
                shutil.move(path, out_path)

        # look for mp4 video in all subdirectories in input_dir
        input_mp4_paths = list(input_dir.glob('**/*.MP4')) + list(input_dir.glob('**/*.mp4'))
        print(f'Found {len(input_mp4_paths)} MP4 videos')

        with ExifToolHelper() as et:
            for mp4_path in input_mp4_paths:
                if mp4_path.is_symlink():
                    print(f"Skipping {mp4_path.name}, already moved.")
                    continue

                start_date = mp4_get_start_datetime(str(mp4_path))
                meta = list(et.get_metadata(str(mp4_path)))[0]
                cam_serial = meta['QuickTime:CameraSerialNumber']
                out_dname = 'demo_' + cam_serial + '_' + start_date.strftime(r"%Y.%m.%d_%H.%M.%S.%f")

                # special folders
                if mp4_path.name.startswith('mapping'):
                    out_dname = "mapping"
                elif mp4_path.name.startswith('gripper_cal') or mp4_path.parent.name.startswith('gripper_cal'):
                    out_dname = "gripper_calibration_" + cam_serial + '_' + start_date.strftime(r"%Y.%m.%d_%H.%M.%S.%f")
                
                # create directory
                this_out_dir = output_dir.joinpath(out_dname)
                this_out_dir.mkdir(parents=True, exist_ok=True)
                
                # move videos
                vfname = 'raw_video.mp4'
                out_video_path = this_out_dir.joinpath(vfname)
                shutil.move(mp4_path, out_video_path)

                # create symlink back from original location
                # relative_to's walk_up argument is not avaliable until python 3.12
                dots = os.path.join(*['..'] * len(mp4_path.parent.relative_to(session).parts))
                rel_path = str(out_video_path.relative_to(session))
                symlink_path = os.path.join(dots, rel_path)                
                mp4_path.symlink_to(symlink_path)

# %%
if __name__ == '__main__':
    if len(sys.argv) == 1:
        main.main(['--help'])
    else:
        main()
        

# """
# python scripts_slam_pipeline/00_process_videos.py data_workspace/toss_objects/20231113
# """
# # %%
# import sys
# import os

# ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
# sys.path.append(ROOT_DIR)
# os.chdir(ROOT_DIR)

# # %%
# import pathlib
# import click
# import shutil
# from exiftool import ExifToolHelper
# from umi.common.timecode_util import mp4_get_start_datetime

# # %%
# def rename_all_mp4_sequential(input_dir):
#     # Rename mp4 files to 1.mp4, 2.mp4, ... (excluding mapping if it already exists)
#     mp4_files = sorted(
#         [f for f in input_dir.glob("*.mp4")] + [f for f in input_dir.glob("*.MP4")],
#         key=lambda x: x.name
#     )
#     temp_names = []

#     # Step 1: Rename all to temporary names to avoid name clashes
#     for i, f in enumerate(mp4_files):
#         tmp_name = input_dir / f"tmp_{i}.mp4"
#         f.rename(tmp_name)
#         temp_names.append(tmp_name)

#     # Step 2: Final renaming
#     for i, tmp_path in enumerate(temp_names, start=1):
#         final_path = input_dir / f"{i}.mp4"
#         tmp_path.rename(final_path)
#         print(f"Renamed {tmp_path.name} -> {final_path.name}")

# # %%
# @click.command(help='Session directories. Assuming mp4 videos are in <session_dir>/raw_videos')
# @click.argument('session_dir', nargs=-1)
# def main(session_dir):
#     for session in session_dir:
#         session = pathlib.Path(os.path.expanduser(session)).absolute()
#         input_dir = session.joinpath('raw_videos')
#         output_dir = session.joinpath('demos')

#         # Create raw_videos if missing
#         if not input_dir.is_dir():
#             input_dir.mkdir()
#             print(f"{input_dir.name} subdir didn't exist. Creating and moving all mp4 videos inside.")
#             for mp4_path in list(session.glob('**/*.MP4')) + list(session.glob('**/*.mp4')):
#                 out_path = input_dir.joinpath(mp4_path.name)
#                 shutil.move(mp4_path, out_path)

#         # Rename all to 1.mp4, 2.mp4, ...
#         rename_all_mp4_sequential(input_dir)

#         # Create mapping.mp4 if it doesn't exist
#         mapping_vid_path = input_dir.joinpath('mapping.mp4')
#         if not mapping_vid_path.exists():
#             largest_mp4 = max(
#                 list(input_dir.glob('*.mp4')),
#                 key=lambda p: p.stat().st_size
#             )
#             shutil.move(largest_mp4, mapping_vid_path)
#             print(f"mapping.mp4 not found. Renamed largest file {largest_mp4.name} to mapping.mp4.")

#         # Create gripper_calibration directory and move 2.mp4 there
#         gripper_cal_dir = input_dir.joinpath('gripper_calibration')
#         if not gripper_cal_dir.is_dir():
#             gripper_cal_dir.mkdir()
#             print("Creating raw_videos/gripper_calibration with video 2.mp4")
#             cal_vid = input_dir / "2.mp4"
#             if cal_vid.exists():
#                 shutil.move(cal_vid, gripper_cal_dir / cal_vid.name)
#             else:
#                 print("Warning: 2.mp4 not found for gripper calibration.")

#         # Final MP4 scan (including subdirs)
#         input_mp4_paths = list(input_dir.glob('**/*.MP4')) + list(input_dir.glob('**/*.mp4'))
#         print(f'Found {len(input_mp4_paths)} MP4 videos')

#         with ExifToolHelper() as et:
#             for mp4_path in input_mp4_paths:
#                 if mp4_path.is_symlink():
#                     print(f"Skipping {mp4_path.name}, already symlinked.")
#                     continue

#                 start_date = mp4_get_start_datetime(str(mp4_path))
#                 meta = list(et.get_metadata(str(mp4_path)))[0]
#                 cam_serial = meta.get('QuickTime:CameraSerialNumber', 'unknown')

#                 out_dname = 'demo_' + cam_serial + '_' + start_date.strftime(r"%Y.%m.%d_%H.%M.%S.%f")

#                 # Handle mapping / gripper_cal special naming
#                 if mp4_path.name.startswith('mapping'):
#                     out_dname = "mapping"
#                 elif mp4_path.name.startswith('gripper_cal') or mp4_path.parent.name.startswith('gripper_cal'):
#                     out_dname = "gripper_calibration_" + cam_serial + '_' + start_date.strftime(r"%Y.%m.%d_%H.%M.%S.%f")

#                 this_out_dir = output_dir.joinpath(out_dname)
#                 this_out_dir.mkdir(parents=True, exist_ok=True)

#                 out_video_path = this_out_dir.joinpath('raw_video.mp4')
#                 shutil.move(mp4_path, out_video_path)

#                 # Symlink to old location
#                 dots = os.path.join(*['..'] * len(mp4_path.parent.relative_to(session).parts))
#                 rel_path = str(out_video_path.relative_to(session))
#                 symlink_path = os.path.join(dots, rel_path)
#                 mp4_path.symlink_to(symlink_path)

# # %%
# if __name__ == '__main__':
#     if len(sys.argv) == 1:
#         main.main(['--help'])
#     else:
#         main()

