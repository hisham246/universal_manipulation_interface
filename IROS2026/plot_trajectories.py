import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import re
import random
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D

# --- Publication Quality Settings ---
plt.rcParams.update({
    'text.usetex': True,                      # Use LaTeX to write all text
    'font.family': 'serif',
    'text.latex.preamble': r'\usepackage{mathpazo}', # Load mathpazo for Palatino font
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

def natural_key(path):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', path)]

data_dir = "/home/hisham246/uwaterloo/peg_in_hole_umi_with_vicon_v3/vicon_final"

episode_files = sorted([f for f in os.listdir(data_dir) if f.startswith("episode_") and f.endswith(".csv")], key=natural_key)

# --- Random Selection with Fixed Seed ---
random.seed(42)  # Change this integer to try different subsets, but keep it fixed once you like the visual
num_trajectories_to_plot = min(10, len(episode_files))
selected_files = random.sample(episode_files, num_trajectories_to_plot)

fig = plt.figure(figsize=(4.5, 3.5)) # Standard single-column width for IEEE is ~3.5 inches
ax = fig.add_subplot(111, projection='3d')

# --- Plot the Selected Trajectories ---
for i, file in enumerate(selected_files):
    file_path = os.path.join(data_dir, file)
    df = pd.read_csv(file_path)

    if all(col in df.columns for col in ["robot0_eef_pos_0", "robot0_eef_pos_1", "robot0_eef_pos_2"]):
        # Downsample data to keep vector graphics small (e.g., every 5th point)
        x = df["robot0_eef_pos_0"].iloc[::5]
        y = df["robot0_eef_pos_1"].iloc[::5]
        z = df["robot0_eef_pos_2"].iloc[::5]
        
        # Use a consistent color (e.g., standard blue) with low alpha to show density
        ax.plot(x, y, z, color='#1f77b4', alpha=0.4, linewidth=1.2)
        
        # --- Plot the Task Frames (Start and End Poses) ---
        # Start Pose (Frame j=1): Green Circle
        ax.scatter(x.iloc[0], y.iloc[0], z.iloc[0], 
                   color='limegreen', marker='o', s=30, alpha=0.9, 
                   edgecolors='black', linewidth=0.5, zorder=5)
        
        # End Pose (Frame j=2): Red Square
        ax.scatter(x.iloc[-1], y.iloc[-1], z.iloc[-1], 
                   color='crimson', marker='s', s=30, alpha=0.9, 
                   edgecolors='black', linewidth=0.5, zorder=5)
    else:
        print(f"Skipping {file}: position columns not found.")

# --- Axis Formatting ---
# Limit the number of ticks to prevent clutter
ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
ax.zaxis.set_major_locator(MaxNLocator(nbins=4)) 

# Use mathtext for LaTeX rendering of xi
ax.set_xlabel(r'$\xi_1$ [m]', labelpad=5)
ax.set_ylabel(r'$\xi_2$ [m]', labelpad=5)
ax.set_zlabel(r'$\xi_3$ [m]', labelpad=5)

# Force equal aspect ratio so the physical space is not distorted
limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
spans = limits[:, 1] - limits[:, 0]
ax.set_box_aspect(spans)

# Clean up pane colors for a cleaner look
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

# Soften the Grid Lines
grid_style = {'color': 'gray', 'alpha': 0.2, 'linewidth': 0.5, 'linestyle': '--'}
ax.xaxis._axinfo['grid'].update(grid_style)
ax.yaxis._axinfo['grid'].update(grid_style)
ax.zaxis._axinfo['grid'].update(grid_style)

# # Add a custom legend to explain the markers
legend_elements = [
    Line2D([0], [0], color='#1f77b4', lw=1.2, alpha=0.8, label='Demonstrations'),
    Line2D([0], [0], marker='o', color='w', label='Start Pose', markerfacecolor='limegreen', markersize=6, markeredgecolor='black', markeredgewidth=0.5),
    Line2D([0], [0], marker='s', color='w', label='End Pose', markerfacecolor='crimson', markersize=6, markeredgecolor='black', markeredgewidth=0.5)
]

# # --- NEW: Horizontal legend above the plot ---
# ax.legend(handles=legend_elements, 
#           loc='lower center',          # The anchor point on the legend itself
#           bbox_to_anchor=(0.5, 1.05),  # Position (x, y) relative to the axes (1.05 is just above)
#           ncol=3,                      # Spread items across 3 columns (horizontal)
#           frameon=False, 
#           handletextpad=0.4,           # Reduce space between marker and text
#           columnspacing=1.0)           # Reduce space between the legend items

# --- NEW: Save the legend to the 'leg' variable ---
leg = ax.legend(handles=legend_elements, 
                loc='lower center',          
                bbox_to_anchor=(0.5, 0.95),  
                ncol=3,                      
                frameon=False, 
                handletextpad=0.4,           
                columnspacing=1.0)

# plt.tight_layout()

# Save as PDF for LaTeX integration
# plt.savefig("IROS2026/fig1_subplotA.pdf", format='pdf', bbox_inches='tight', pad_inches=0.05)
# --- NEW: Force matplotlib to include the legend in the crop ---
plt.savefig("IROS2026/fig1_subplotA.pdf", 
            format='pdf', 
            bbox_inches='tight', 
            bbox_extra_artists=(leg,),  # <--- This prevents the legend from being cropped
            pad_inches=0.3)             # Gave it slightly more padding
plt.show()