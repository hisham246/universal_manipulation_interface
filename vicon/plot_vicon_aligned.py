import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_all_trajectories(directory='.', file_pattern='episode_*.csv'):
    # Find all matching files in the directory
    files = glob.glob(os.path.join(directory, file_pattern))
    files.sort()
    
    if not files:
        print(f"No files matching '{file_pattern}' found in '{directory}'.")
        return

    # Initialize the 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Loop over each file and plot its trajectory
    for f in files:
        try:
            df = pd.read_csv(f)
            # Ensure the required columns exist
            required_cols = ["robot0_eef_pos_0", "robot0_eef_pos_1", "robot0_eef_pos_2"]
            if all(col in df.columns for col in required_cols):
                ax.plot(df['robot0_eef_pos_0'], df['robot0_eef_pos_1'], df['robot0_eef_pos_2'], label=os.path.basename(f))
            else:
                print(f"Skipping {f}: Missing coordinate columns.")
        except Exception as e:
            print(f"Error reading {f}: {e}")

    # Configure plot labels and style
    ax.set_xlabel('Pos_X')
    ax.set_ylabel('Pos_Y')
    ax.set_zlabel('Pos_Z')
    ax.set_title('3D Trajectories from Episodes')
    # ax.legend(loc='upper right', fontsize='small')
    
    # Save or show the plot
    plt.tight_layout()
    plt.show()
    # plt.savefig('trajectories_3d_combined.png')

def plot_one_by_one(directory='.', file_pattern='episode_*.csv'):
    files = glob.glob(os.path.join(directory, file_pattern))
    files.sort()
    
    for f in files:
        df = pd.read_csv(f)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the single trajectory
        ax.plot(df['robot0_eef_pos_0'], df['robot0_eef_pos_1'], df['robot0_eef_pos_2'])
        
        ax.set_title(f"Trajectory: {os.path.basename(f)}")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        plt.show(block=False) # Show without blocking code execution
        print(f"Showing {f}. Click the plot or press a key for next...")
        
        plt.waitforbuttonpress() 
        plt.close(fig) # Close current window to open the next one

if __name__ == "__main__":
    directory = '/home/hisham246/uwaterloo/peg_in_hole_delta_umi/VIDP_data/dataset_with_vicon_trimmed_filtered_2'
    # directory = '/home/hisham246/uwaterloo/VIDP_IROS2026/peg_in_hole/vicon'
    plot_all_trajectories(directory=directory)
    plot_one_by_one(directory=directory)