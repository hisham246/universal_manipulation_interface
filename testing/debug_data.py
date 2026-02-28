import os
import pandas as pd
import matplotlib.pyplot as plt

def load_all_stiffness_data(directory_path):
    """Loads and concatenates stiffness data from all CSVs in a directory."""
    all_data = []
    # Adjust the file extension or naming convention if needed
    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            filepath = os.path.join(directory_path, filename)
            df = pd.read_csv(filepath)
            # Assuming columns are named Kx, Ky, Kz
            if all(col in df.columns for col in ['Kx', 'Ky', 'Kz']):
                all_data.append(df[['Kx', 'Ky', 'Kz']])
    
    if not all_data:
        raise ValueError(f"No valid CSV files found in {directory_path}")
        
    return pd.concat(all_data, ignore_index=True)

def compare_datasets(old_dir, new_dir):
    print("Loading old dataset...")
    df_old = load_all_stiffness_data(old_dir)
    print("Loading new dataset...")
    df_new = load_all_stiffness_data(new_dir)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    components = ['Kx', 'Ky', 'Kz']
    colors_old = 'blue'
    colors_new = 'orange'

    for i, comp in enumerate(components):
        # Plot old dataset
        axes[i].hist(df_old[comp], bins=50, color=colors_old, alpha=0.5, 
                     density=True, label='Old Dataset (Worked)')
        # Plot new dataset
        axes[i].hist(df_new[comp], bins=50, color=colors_new, alpha=0.5, 
                     density=True, label='New Dataset (Fails)')
        
        axes[i].set_title(f'{comp} Distribution Comparison')
        axes[i].set_xlabel(f'Stiffness ({comp})')
        axes[i].set_ylabel('Density')
        axes[i].legend()

    plt.suptitle('Stiffness Action Distribution: Old vs. New Dataset', fontsize=16)
    plt.tight_layout()
    plt.savefig('dataset_comparison.png')
    plt.show()

# --- RUN THE COMPARISON ---
# Replace these with the actual paths to your dataset folders containing the CSVs
OLD_DATASET_DIR = "/home/hisham246/uwaterloo/VIDP_IROS2026/stiffness_profiles_peg_in_hole" 
NEW_DATASET_DIR = "/home/hisham246/uwaterloo/VIDP_IROS2026/stiffness_profiles_cable_route_no_blue_station"

compare_datasets(OLD_DATASET_DIR, NEW_DATASET_DIR)