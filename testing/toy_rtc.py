import numpy as np

def make_chunk(horizon, chunk_id):
    """
    Fake 'policy' output: a chunk of length H whose elements encode
    which inference it came from, so we can visually track blending.

    Example for chunk_id=2, H=8:
    [200, 201, 202, 203, 204, 205, 206, 207]
    """
    base = 100 * chunk_id
    return np.arange(base, base + horizon)

def rtc_cycle(prev_chunk, curr_chunk, s_horizon, d_forecast, cycle_idx):
    """
    One RTC planning cycle:
      - prev_chunk: old horizon (world space)
      - curr_chunk: new horizon (world space)
      - s_horizon: how many steps we will execute before next inference
      - d_forecast: how many steps must be continuous with prev
    """
    H = len(prev_chunk)
    H_world = min(len(prev_chunk), len(curr_chunk), H)
    s_horizon = min(s_horizon, H_world)
    d = min(d_forecast, s_horizon)

    part_prev = prev_chunk[:d]
    part_curr = curr_chunk[d:s_horizon]
    horizon = np.concatenate([part_prev, part_curr], axis=0)

    print(f"\n=== RTC cycle {cycle_idx} ===")
    print(f"H = {H}, s_horizon = {s_horizon}, d_forecast = {d_forecast}, d = {d}")
    print(f"prev_chunk: {prev_chunk}")
    print(f"curr_chunk: {curr_chunk}")
    print(f"execution horizon (prev[0:{d}] + curr[{d}:{s_horizon}]):")
    print(horizon)

    # Annotate where each action came from for clarity
    for i, a in enumerate(horizon):
        if i < d:
            src = "prev"
            src_id = prev_chunk[0] // 100
        else:
            src = "curr"
            src_id = curr_chunk[0] // 100
        print(f"  step {i:2d}: action={a:3d}  (from {src} chunk #{src_id})")

    return horizon

# --- parameters, analogous to your real setup ---

H = 8                # policy.action_horizon
steps_per_inference = 4   # like your smin / s_horizon
d_forecast = 2       # pretend the delay predictor says "freeze 2 steps"

# initial chunk (think of this as prev_raw_chunk/world_chunk from inference #0)
prev_chunk = make_chunk(H, chunk_id=0)

executed_actions = []

# simulate a few cycles
num_cycles = 3
for cycle in range(1, num_cycles + 1):
    curr_chunk = make_chunk(H, chunk_id=cycle)   # fake new policy chunk

    horizon = rtc_cycle(
        prev_chunk=prev_chunk,
        curr_chunk=curr_chunk,
        s_horizon=steps_per_inference,
        d_forecast=d_forecast,
        cycle_idx=cycle
    )

    # "Execute" all of them in order — THIS is what exec_actions should do
    executed_actions.extend(list(horizon))

    # For the next cycle, the "previous" chunk is now the current model output
    prev_chunk = curr_chunk

print("\n=== Full executed sequence over time ===")
print(executed_actions)
