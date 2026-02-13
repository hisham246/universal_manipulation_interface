import sys
import os

# Set up Root Directory
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import pathlib
import click
import pickle
import numpy as np
import json
import math
import collections
import scipy.ndimage as sn
import pandas as pd
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import av
from exiftool import ExifToolHelper
from umi.common.timecode_util import mp4_get_start_datetime
from umi.common.pose_util import pose_to_mat, mat_to_pose
from umi.common.cv_util import get_gripper_width
from umi.common.interpolation_util import (
    get_gripper_calibration_interpolator, 
    get_interp1d,
    PoseInterpolator
)
from collections import Counter

# %% Helper Functions
def get_bool_segments(bool_seq):
    bool_seq = np.array(bool_seq, dtype=bool)
    if len(bool_seq) == 0:
        return [], np.array([], dtype=bool)
    segment_ends = (np.nonzero(np.diff(bool_seq))[0] + 1).tolist()
    segment_bounds = [0] + segment_ends + [len(bool_seq)]
    segments = list()
    segment_type = list()
    for i in range(len(segment_bounds) - 1):
        start = segment_bounds[i]
        end = segment_bounds[i+1]
        this_type = bool_seq[start]
        segments.append(slice(start, end))
        segment_type.append(this_type)
    segment_type = np.array(segment_type, dtype=bool)
    return segments, segment_type

def pose_interp_from_df(df, start_timestamp=0.0, tx_base_slam=None):    
    timestamp_sec = df['timestamp'].to_numpy() + start_timestamp
    cam_pos = df[['x', 'y', 'z']].to_numpy()
    cam_rot_quat_xyzw = df[['q_x', 'q_y', 'q_z', 'q_w']].to_numpy()
    cam_rot = Rotation.from_quat(cam_rot_quat_xyzw)
    cam_pose = np.zeros((cam_pos.shape[0], 4, 4), dtype=np.float32)
    cam_pose[:,3,3] = 1
    cam_pose[:,:3,3] = cam_pos
    cam_pose[:,:3,:3] = cam_rot.as_matrix()
    tx_slam_cam = cam_pose
    tx_base_cam = tx_slam_cam
    if tx_base_slam is not None:
        tx_base_cam = tx_base_slam @ tx_slam_cam
    pose_interp = PoseInterpolator(
        t=timestamp_sec, x=mat_to_pose(tx_base_cam))
    return pose_interp

def get_x_projection(tx_tag_this, tx_tag_other):
    """
    Calculates projection of 'other' camera onto the 'right' vector of 'this' camera.
    Fixed to handle both single 4x4 matrices and batches of [N, 4, 4].
    """
    # Detect if single 2D matrix [4,4] or 3D batch [N,4,4]
    is_single = tx_tag_this.ndim == 2
    if is_single:
        tx_tag_this = tx_tag_this[None, ...]
        tx_tag_other = tx_tag_other[None, ...]

    t_this_other = tx_tag_other[:,:3,3] - tx_tag_this[:,:3,3]
    v_this_forward = tx_tag_this[:,:3,2]
    v_up = np.array([0.,0.,1.])
    
    # Cross product to find the 'right' vector in global frame
    v_this_right = np.cross(v_this_forward, v_up)
    proj_other_right = np.sum(v_this_right * t_this_other, axis=-1)
    
    return proj_other_right[0] if is_single else proj_other_right

# %% Main Pipeline
@click.command()
@click.option('-i', '--input', required=True, help='Project directory')
@click.option('-o', '--output', default=None)
@click.option('-to', '--tcp_offset', type=float, default=0.205)
@click.option('-ts', '--tx_slam_tag', default=None)
@click.option('-nz', '--nominal_z', type=float, default=0.072)
@click.option('-ml', '--min_episode_length', type=int, default=24)
@click.option('--ignore_cameras', type=str, default=None)
def main(input, output, tcp_offset, tx_slam_tag,
         nominal_z, min_episode_length, ignore_cameras):
    
    input_path = pathlib.Path(os.path.expanduser(input)).absolute()
    demos_dir = input_path.joinpath('demos')
    if output is None:
        output = input_path.joinpath('dataset_plan_gripper_only.pkl')

    # Calibration constants
    cam_to_center_height = 0.086 
    cam_to_mount_offset = 0.01465 
    cam_to_tip_offset = cam_to_mount_offset + tcp_offset
    tx_cam_tcp = pose_to_mat(np.array([0, cam_to_center_height, cam_to_tip_offset, 0,0,0]))
        
    # SLAM map to table tag
    if tx_slam_tag is None:
        path = demos_dir.joinpath('mapping', 'tx_slam_tag.json')
        tx_slam_tag = np.array(json.load(open(path, 'r'))['tx_slam_tag']) if path.is_file() else np.eye(4)
    else:
        tx_slam_tag = np.array(json.load(open(os.path.expanduser(tx_slam_tag), 'r'))['tx_slam_tag'])
    tx_tag_slam = np.linalg.inv(tx_slam_tag)

    # Gripper calibration maps
    gripper_id_gripper_cal_map = dict()
    cam_serial_gripper_cal_map = dict()
    with ExifToolHelper() as et:
        for gripper_cal_path in demos_dir.glob("gripper*/gripper_range.json"):
            mp4_path = gripper_cal_path.parent.joinpath('raw_video.mp4')
            if not mp4_path.is_file(): continue
            meta = list(et.get_metadata(str(mp4_path)))[0]
            cam_serial = meta.get('QuickTime:CameraSerialNumber', 'unknown')
            g_data = json.load(gripper_cal_path.open('r'))
            interp = get_gripper_calibration_interpolator(
                aruco_measured_width=[g_data['min_width'], g_data['max_width']],
                aruco_actual_width=[g_data['min_width'], g_data['max_width']]
            )
            gripper_id_gripper_cal_map[g_data['gripper_id']] = interp
            cam_serial_gripper_cal_map[cam_serial] = interp

    # %% Stage 1: Video Extraction
    video_dirs = sorted([x.parent for x in demos_dir.glob('demo_*/raw_video.mp4')])
    ignore_serials = set(ignore_cameras.split(',')) if ignore_cameras else set()
    fps, rows = None, []
    
    with ExifToolHelper() as et:
        for video_dir in video_dirs:            
            mp4_path = video_dir.joinpath('raw_video.mp4')
            meta = list(et.get_metadata(str(mp4_path)))[0]
            cam_serial = meta.get('QuickTime:CameraSerialNumber', 'unknown')
            if cam_serial in ignore_serials: continue
            
            start_ts = mp4_get_start_datetime(str(mp4_path)).timestamp()
            with av.open(str(mp4_path), 'r') as container:
                stream = container.streams.video[0]
                n_frames = stream.frames
                if fps is None: fps = stream.average_rate
            
            rows.append({
                'video_dir': video_dir, 'camera_serial': cam_serial, 'n_frames': n_frames,
                'fps': fps, 'start_timestamp': start_ts, 'end_timestamp': start_ts + float(n_frames / fps)
            })
    
    if not rows: return
    video_meta_df = pd.DataFrame(data=rows)

    # %% Stage 2: Temporal Demo Matching
    serial_count = video_meta_df['camera_serial'].value_counts()
    n_cameras = len(serial_count)
    events = []
    for vid_idx, row in video_meta_df.iterrows():
        events.append({'vid_idx': vid_idx, 't': row['start_timestamp'], 'is_start': True, 'cs': row['camera_serial']})
        events.append({'vid_idx': vid_idx, 't': row['end_timestamp'], 'is_start': False, 'cs': row['camera_serial']})
    events = sorted(events, key=lambda x: x['t'])
    
    demo_list, on_vids, on_cams, t_demo_start = [], set(), set(), None
    for event in events:
        if event['is_start']:
            on_vids.add(event['vid_idx']); on_cams.add(event['cs'])
        else:
            on_vids.remove(event['vid_idx']); on_cams.remove(event['cs'])
        
        if len(on_cams) == n_cameras:
            t_demo_start = event['t']
        elif t_demo_start is not None:
            v_idxs = set(on_vids); v_idxs.add(event['vid_idx'])
            demo_list.append({"video_idxs": sorted(v_idxs), "start_timestamp": t_demo_start, "end_timestamp": event['t']})
            t_demo_start = None

    # %% Stage 3: Gripper Hardware ID
    vid_gripper_id_map = {}
    for vid_idx, row in video_meta_df.iterrows():
        pkl_path = row['video_dir'].joinpath('tag_detection.pkl')
        if not pkl_path.is_file():
            vid_gripper_id_map[vid_idx] = 0
            continue
        tag_data = pickle.load(pkl_path.open('rb'))
        counts = Counter([k for f in tag_data for k in f['tag_dict'].keys()])
        stats = {k: v / len(tag_data) for k, v in counts.items()}
        
        best_id, max_p = 0, 0.0
        for g_id in range(5):
            p = min(stats.get(g_id*6, 0), stats.get(g_id*6+1, 0))
            if p > max_p: best_id, max_p = g_id, p
        vid_gripper_id_map[vid_idx] = best_id

    video_meta_df['gripper_hardware_id'] = pd.Series(vid_gripper_id_map)
    cam_gripper_id_map = video_meta_df.groupby('camera_serial')['gripper_hardware_id'].agg(lambda x: x.value_counts().index[0]).to_dict()

    # %% Stage 4: L/R Disambiguation
    grip_serials = [cs for cs, gi in cam_gripper_id_map.items() if gi >= 0]
    n_gripper_cams = len(grip_serials)
    other_serials = sorted([cs for cs, gi in cam_gripper_id_map.items() if gi < 0])
    
    cam_idx_map = {cs: n_gripper_cams + i for i, cs in enumerate(other_serials)}
    disambiguation_votes = collections.defaultdict(list)

    for demo in demo_list:
        p_interps, demo_cs = [], []
        for v_idx in demo['video_idxs']:
            row = video_meta_df.loc[v_idx]
            if row.gripper_hardware_id < 0: continue
            
            csv_path = row['video_dir'].joinpath('camera_trajectory.csv')
            if csv_path.is_file(): 
                df = pd.read_csv(csv_path).query('~is_lost') 
            else: 
                df = pd.DataFrame({'timestamp': [0.0, 1000.0], # Arbitrary spread
                        'x': [0, 0], 'y': [0, 0], 'z': [0, 0], 
                        'q_x': [0, 0], 'q_y': [0, 0], 'q_z': [0, 0], 'q_w': [1, 1]
                    })            
            p_interps.append(pose_interp_from_df(df, start_timestamp=row['start_timestamp'], tx_base_slam=tx_tag_slam))
            demo_cs.append(row['camera_serial'])
        
        if len(p_interps) == n_gripper_cams and n_gripper_cams > 0:
            t_mid = (demo['start_timestamp'] + demo['end_timestamp']) / 2
            current_poses = [pose_to_mat(interp(t_mid)) for interp in p_interps]
            
            x_projs = []
            for i in range(len(current_poses)):
                # Calculate projection against all other cameras in this specific demo
                projs = [get_x_projection(current_poses[i], current_poses[j]) for j in range(len(current_poses)) if i != j]
                x_projs.append(np.mean(projs) if projs else 0.0)
            
            # Smallest x_proj (right-most) to largest (left-most)
            for i, c_idx in enumerate(np.argsort(x_projs)):
                disambiguation_votes[demo_cs[c_idx]].append(i)

    for cs in grip_serials:
        counts = Counter(disambiguation_votes[cs])
        cam_idx_map[cs] = counts.most_common(1)[0][0] if counts else 0
    
    video_meta_df['camera_idx'] = video_meta_df['camera_serial'].map(cam_idx_map)

    # %% Stage 6: Generation
    all_plans = []
    for demo in demo_list:
        demo_vids = video_meta_df.loc[demo['video_idxs']].set_index('camera_idx').sort_index()
        dt = 1 / demo_vids.iloc[0]['fps']
        n_frames = int((demo['end_timestamp'] - demo['start_timestamp']) / dt)
        
        all_poses, all_widths = [], []
        for c_idx, row in demo_vids.iterrows():
            if c_idx >= n_gripper_cams: continue
            
            # Robust Trajectory Handling
            csv_path = row['video_dir'].joinpath('camera_trajectory.csv')
            if csv_path.is_file():
                df = pd.read_csv(csv_path).iloc[:n_frames]
                if len(df) < n_frames:
                    pad = pd.DataFrame(index=range(n_frames - len(df)), columns=df.columns).fillna(0)
                    pad['q_w'] = 1; df = pd.concat([df, pad])
            else:
                # Provide a full array of identical rows to match n_frames
                # This ensures the matrix operations later don't fail on shape
                df = pd.DataFrame({
                    'x': 0.0, 'y': 0.0, 'z': 0.0, 
                    'q_x': 0.0, 'q_y': 0.0, 'q_z': 0.0, 'q_w': 1.0
                }, index=range(n_frames))
            
            c_mat = np.zeros((n_frames, 4, 4)); c_mat[:,3,3] = 1
            c_mat[:,:3,3] = df[['x','y','z']].values
            c_mat[:,:3,:3] = Rotation.from_quat(df[['q_x','q_y','q_z','q_w']].fillna({'q_w':1}).values).as_matrix()
            all_poses.append(mat_to_pose(tx_tag_slam @ c_mat @ tx_cam_tcp))

            # Robust Gripper Handling
            pkl_path = row['video_dir'].joinpath('tag_detection.pkl')
            w_arr = np.zeros(n_frames)
            if pkl_path.is_file():
                tags = pickle.load(open(pkl_path, 'rb'))[:n_frames]
                cal = cam_serial_gripper_cal_map.get(row['camera_serial'], lambda x: x if x else 0.0)
                for i, td in enumerate(tags):
                    w = get_gripper_width(td['tag_dict'], left_id=row['gripper_hardware_id']*6, right_id=row['gripper_hardware_id']*6+1, nominal_z=nominal_z)
                    w_arr[i] = cal(w) if w is not None else 0.0
            all_widths.append(w_arr)

        grippers = [{"tcp_pose": all_poses[i], "gripper_width": all_widths[i]} for i in range(len(all_poses))]
        cameras = [{"video_path": "raw_video.mp4", "video_start_end": (0, n_frames)} for _ in range(len(demo_vids))]

        all_plans.append({
            "episode_timestamps": np.arange(n_frames) * dt + demo['start_timestamp'],
            "grippers": grippers, "cameras": cameras
        })

    pickle.dump(all_plans, output.open('wb'))
    print(f"Success! {len(all_plans)} episodes processed.")

if __name__ == "__main__":
    main()