import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as mcolors

# --- Publication Quality Settings ---
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'text.latex.preamble': r'\usepackage{mathpazo}', 
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


def plot_ellipsoid(ax, mu, sigma, color, alpha, label=None, n_std=2.0):
    """Plots a 3D covariance ellipsoid."""
    # Compute eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eigh(sigma)
    
    # Sort eigenvalues and vectors to ensure consistent plotting
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    
    # Radii corresponding to n standard deviations
    radii = np.sqrt(eigvals) * n_std
    
    # Generate data for a unit sphere
    u = np.linspace(0.0, 2.0 * np.pi, 30)
    v = np.linspace(0.0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    
    # Pack into a single matrix, stretch by radii, rotate by eigenvectors
    sphere_pts = np.vstack((x.flatten(), y.flatten(), z.flatten()))
    ellipsoid_pts = (eigvecs @ np.diag(radii)) @ sphere_pts
    
    # Translate to mean
    X = ellipsoid_pts[0, :].reshape(x.shape) + mu[0]
    Y = ellipsoid_pts[1, :].reshape(y.shape) + mu[1]
    Z = ellipsoid_pts[2, :].reshape(z.shape) + mu[2]
    
    # --- FIX: Use linewidth=0 instead of edgecolors='none' ---
    ax.plot_surface(X, Y, Z, color=color, alpha=alpha, linewidth=0, antialiased=True)

# --- Setup Figure ---
fig = plt.figure(figsize=(4.5, 3.5))
ax = fig.add_subplot(111, projection='3d')

# --- 1. Generate Synthetic Intersecting Trajectories ---
# Trajectory 1: Moving Forward (+Y direction)
t = np.linspace(-1, 1, 50)
traj1_x = 0.1 * np.sin(np.pi * t)
traj1_y = t
traj1_z = 0.1 * np.cos(np.pi * t)
ax.plot(traj1_x, traj1_y, traj1_z, color='black', linestyle='-', linewidth=1.5, alpha=0.8, zorder=4)
# Add an arrowhead to indicate direction
ax.quiver(traj1_x[24], traj1_y[24], traj1_z[24], 
          traj1_x[25]-traj1_x[24], traj1_y[25]-traj1_y[24], traj1_z[25]-traj1_z[24], 
          color='black', length=0.2, normalize=True, zorder=5)

# Trajectory 2: Moving Backward (-Y direction, spatially overlapping)
traj2_x = 0.1 * np.sin(np.pi * t) + 0.05
traj2_y = -t
traj2_z = 0.1 * np.cos(np.pi * t) - 0.02
ax.plot(traj2_x, traj2_y, traj2_z, color='black', linestyle='--', linewidth=1.5, alpha=0.8, zorder=4)
# Add an arrowhead to indicate direction
ax.quiver(traj2_x[24], traj2_y[24], traj2_z[24], 
          traj2_x[25]-traj2_x[24], traj2_y[25]-traj2_y[24], traj2_z[25]-traj2_z[24], 
          color='black', length=0.2, normalize=True, zorder=5)

# --- 2. Define and Plot Covariances (Replace with your actual Algorithm 1 outputs) ---
# Center of the bottleneck
mu_center = np.array([0.025, 0.0, 0.04])

# Standard TP-GMM Covariance (Fails to distinguish direction, creates a fat sphere)
sigma_gmm = np.array([
    [0.05, 0.0, 0.0],
    [0.0, 0.5, 0.0],
    [0.0, 0.0, 0.05]
])
plot_ellipsoid(ax, mu_center, sigma_gmm, color='crimson', alpha=0.15)

# TP-DAMM Covariance 1 (Strictly follows Trajectory 1)
sigma_damm1 = np.array([
    [0.005, 0.01, 0.0],
    [0.01, 0.2, 0.0],
    [0.0, 0.0, 0.005]
])
plot_ellipsoid(ax, np.array([0, 0, 0.1]), sigma_damm1, color='dodgerblue', alpha=0.6)

# TP-DAMM Covariance 2 (Strictly follows Trajectory 2)
sigma_damm2 = np.array([
    [0.005, -0.01, 0.0],
    [-0.01, 0.2, 0.0],
    [0.0, 0.0, 0.005]
])
plot_ellipsoid(ax, np.array([0.05, 0, -0.02]), sigma_damm2, color='dodgerblue', alpha=0.6)

# --- Axis Formatting ---
ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
ax.zaxis.set_major_locator(MaxNLocator(nbins=4))

ax.set_xlabel(r'$\xi_1$ [m]', labelpad=5)
ax.set_ylabel(r'$\xi_2$ [m]', labelpad=5)
ax.set_zlabel(r'$\xi_3$ [m]', labelpad=5)

# Force equal aspect ratio
limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
spans = limits[:, 1] - limits[:, 0]
ax.set_box_aspect(spans)

# Clean up pane colors
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

# Soften the Grid Lines
grid_style = {'color': 'gray', 'alpha': 0.2, 'linewidth': 0.5, 'linestyle': '--'}
ax.xaxis._axinfo['grid'].update(grid_style)
ax.yaxis._axinfo['grid'].update(grid_style)
ax.zaxis._axinfo['grid'].update(grid_style)

# --- Legend ---
legend_elements = [
    Line2D([0], [0], color='black', lw=1.5, label='Expert Trajectories'),
    Patch(facecolor='crimson', alpha=0.3, label='Standard TP-GMM'),
    Patch(facecolor='dodgerblue', alpha=0.6, label='TP-DAMM (Ours)')
]

leg = ax.legend(handles=legend_elements, 
                loc='lower center',          
                bbox_to_anchor=(0.5, 1.05),  
                ncol=3,                      
                frameon=False, 
                handletextpad=0.4,           
                columnspacing=1.0)           

# --- Save Figure ---
plt.savefig("IROS2026/fig1_subplotB.pdf", 
            format='pdf', 
            bbox_inches='tight', 
            bbox_extra_artists=(leg,), 
            pad_inches=0.1)
plt.show()