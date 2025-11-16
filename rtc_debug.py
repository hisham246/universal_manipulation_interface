import numpy as np
import matplotlib.pyplot as plt

def compute_W(H, s, d, schedule="exp"):
    """
    Reproduce your _get_prefix_weights logic (only for prefix region).

    H      : horizon length
    s      : executed steps from previous chunk
    d      : delay forecast
    schedule: "ones", "linear", "exp" (matches your code)
    """
    total = H
    start = d
    end = H - s  # prefix_attention_horizon

    indices = np.arange(total, dtype=np.float32)

    if schedule == "ones":
        w = np.ones(total, dtype=np.float32)
    elif schedule == "zeros":
        w = (indices < start).astype(np.float32)
    elif schedule in ("linear", "exp"):
        # same formula as your _get_prefix_weights
        w = (start - 1 - indices) / (end - start + 1) + 1.0
        w = np.clip(w, 0.0, 1.0)
        if schedule == "exp":
            w = w * (np.exp(w) - 1.0) / (np.e - 1.0)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")

    # zero out beyond prefix_h = end
    w = np.where(indices >= end, 0.0, w)
    return w


def guidance_schedule(tau_grid, max_guidance_weight=5.0):
    """
    Plot the guidance schedule we talked about:

        gw_raw = (1 - τ) / (τ * r^2),  r^2 = τ^2 + (1-τ)^2

    clipped at max_guidance_weight.
    """
    eps = 1e-6
    t = tau_grid
    r2 = t**2 + (1.0 - t)**2

    gw_raw = (1.0 - t) / ((t + eps) * (r2 + eps))
    gw_clip = np.minimum(gw_raw, max_guidance_weight)

    return gw_raw, gw_clip


def debug_example():
    # Example: your typical case
    H = 8
    s = 5
    d = 2   # NOTE: use d < H-s to get a non-empty exponential region

    print(f"Example: H={H}, s={s}, d={d}")
    for sched in ["ones", "linear", "exp"]:
        W = compute_W(H, s, d, schedule=sched)
        print(f"{sched} -> W:", W)

        plt.figure()
        plt.stem(np.arange(H), W, use_line_collection=True)
        plt.ylim([-0.1, 1.1])
        plt.xlabel("time index i")
        plt.ylabel("W[i]")
        plt.title(f"W over horizon (H={H}, s={s}, d={d}, schedule={sched})")
        plt.grid(True)

    # Guidance schedule
    tau_grid = np.linspace(0.05, 0.95, 200)
    gw_raw, gw_clip = guidance_schedule(tau_grid, max_guidance_weight=5.0)

    plt.figure()
    plt.plot(tau_grid, gw_raw, label="gw_raw(τ)")
    plt.plot(tau_grid, gw_clip, label="gw_clipped(τ≤5)", linestyle="--")
    plt.xlabel("τ (flow time)")
    plt.ylabel("guidance weight")
    plt.title("ΠGDM guidance schedule vs τ")
    plt.legend()
    plt.grid(True)

    plt.show()


def plot_guidance():

    log = np.load("/home/hisham246/uwaterloo/rtc_debug_reaching/rtc_guidance_log.npy", allow_pickle=True)
    tau, gw_raw, gw_clip = log[:,0], log[:,1], log[:,2]

    plt.figure()
    plt.scatter(tau, gw_clip, s=5)
    plt.xlabel("τ")
    plt.ylabel("guidance (clipped)")
    plt.title("ΠGDM guidance during real run")
    plt.grid(True)
    plt.show()

def load_and_stack_W(path):
    """
    Loads rtc_W_log.npy and stacks it into [N_chunks, H].
    Handles object arrays (variable-length) by padding to max H.
    """
    W_list = np.load(path, allow_pickle=True)

    # if already numeric 2D array, just return
    if isinstance(W_list, np.ndarray) and W_list.dtype != object:
        return W_list

    # otherwise, assume it's an array of 1D arrays
    W_list = list(W_list)
    H = max(w.shape[0] for w in W_list)
    N = len(W_list)

    W_mat = np.zeros((N, H), dtype=np.float32)
    for i, w in enumerate(W_list):
        W_mat[i, :len(w)] = w
    return W_mat

def plot_W_heatmap(W_mat):
    """
    W_mat: [N_chunks, H]
    """
    N, H = W_mat.shape
    plt.figure(figsize=(8, 4))
    plt.imshow(W_mat, aspect='auto', origin='lower',
               interpolation='nearest')
    plt.colorbar(label='W[i]')
    plt.xlabel('Horizon index i')
    plt.ylabel('Chunk index k')
    plt.title(f'RTC prefix mask W over chunks (N={N}, H={H})')
    plt.tight_layout()

def plot_W_stems(W_mat, num_examples=5):
    """
    Plot stem plots for a few chunks to inspect W manually.
    """
    N, H = W_mat.shape
    num_examples = min(num_examples, N)

    plt.figure(figsize=(10, 2 * num_examples))
    for j in range(num_examples):
        idx = int(j * (N-1) / max(num_examples-1, 1))  # spread across log
        plt.subplot(num_examples, 1, j+1)
        plt.stem(np.arange(H), W_mat[idx], use_line_collection=True)
        plt.ylim([-0.1, 1.1])
        plt.ylabel(f'chunk {idx}')
        if j == 0:
            plt.title('Example W profiles across chunks')
        if j == num_examples - 1:
            plt.xlabel('Horizon index i')
        plt.grid(True)

    plt.tight_layout()    

if __name__ == "__main__":
    # debug_example()
    plot_guidance()
    path = "/home/hisham246/uwaterloo/rtc_debug_reaching/rtc_W_log.npy"
    W_mat = load_and_stack_W(path)
    plot_W_heatmap(W_mat)
    plot_W_stems(W_mat, num_examples=8)
    plt.show()