import sys
import os

# Standard UMI Workspace Setup
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import pathlib
import click
import zarr
import pickle
import numpy as np
from diffusion_policy.common.replay_buffer import ReplayBuffer

@click.command()
@click.argument('input', nargs=-1)
@click.option('-o', '--output', required=True, help='Zarr path (e.g. dataset.zarr.zip)')
def main(input, output):
    # 1. Overwrite check
    if os.path.isfile(output):
        if not click.confirm(f'Output file {output} exists! Overwrite?', default=False):
            raise click.Abort()
        
    # Create empty ReplayBuffer in memory
    out_replay_buffer = ReplayBuffer.create_empty_zarr(
        storage=zarr.MemoryStore())
    
    total_episodes = 0
    max_grippers_found = 0
    
    for ipath_str in input:
        ipath = pathlib.Path(os.path.expanduser(ipath_str)).absolute()
        
        # Look for the gripper-only plan first
        plan_path = ipath.joinpath('dataset_plan_gripper_only.pkl')
        if not plan_path.is_file():
            plan_path = ipath.joinpath('dataset_plan.pkl')
            if not plan_path.is_file():
                print(f"Skipping {ipath.name}: no dataset plan found.")
                continue
        
        print(f"Loading plan from: {plan_path}")
        with open(plan_path, 'rb') as f:
            plan = pickle.load(f)
        
        for plan_episode in plan:
            grippers = plan_episode['grippers']
            current_n_grippers = len(grippers)
            
            # Update global gripper count for reporting
            max_grippers_found = max(max_grippers_found, current_n_grippers)
                
            episode_data = dict()
            
            # Loop through available grippers
            for gripper_id, gripper in enumerate(grippers):    
                gripper_widths = gripper['gripper_width']
                
                robot_name = f'robot{gripper_id}'
                
                # Standard UMI/Diffusion Policy format: (T, 1)
                episode_data[robot_name + '_gripper_width'] = np.expand_dims(
                    gripper_widths, axis=-1).astype(np.float32)
            
            # Add timestamp (defines the 'T' dimension for the ReplayBuffer)
            episode_data['timestamp'] = plan_episode['episode_timestamps'].astype(np.float64)
            
            # Add to buffer
            try:
                out_replay_buffer.add_episode(data=episode_data, compressors=None)
                total_episodes += 1
            except Exception as e:
                print(f"Error adding episode from {ipath.name}: {e}")

    print(f"Successfully added {total_episodes} episodes.")
    print(f"Max grippers detected across dataset: {max_grippers_found}")
    
    # Save to disk as a ZipStore
    print(f"Saving ReplayBuffer to {output}")
    if os.path.exists(output):
        os.remove(output) # Clean start for the zip file
        
    with zarr.ZipStore(output, mode='w') as zip_store:
        out_replay_buffer.save_to_store(store=zip_store)
    
    print("Done!")

if __name__ == "__main__":
    main()