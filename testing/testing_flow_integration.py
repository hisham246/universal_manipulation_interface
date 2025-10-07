"""
Code from https://github.com/Physical-Intelligence/real-time-chunking-kinetix/issues/4#issuecomment-3049756738
"""
import jax
import jax.numpy as jnp
from typing import Callable


def pinv_corrected_velocity(
    v_t_fn: Callable, # ([ah ad], float) -> [ah ad]
    x_t: jax.Array, # [b ah ad]
    t: float,
    prefix_actions: jax.Array, # [b ah ad]
    inference_delay: int,
    prefix_attention_horizon: int,
    max_guidance_weight: float,
) -> jax.Array: # [b ah ad]
    @jax.vmap
    def _pinv_corrected_velocity(
        x_t: jax.Array, # [ah ad]
        y: jax.Array, # [ah a   d]
    ) -> jax.Array: # [ah ad]
        def denoiser(x_t: jax.Array) -> tuple[jax.Array, jax.Array]: # [ah ad]
            v_t = v_t_fn(x_t, t)
            return x_t - v_t * t, v_t

        x_0, vjp_fun, v_t = jax.vjp(denoiser, x_t, has_aux=True)
        error = (y - x_0) * get_prefix_weights(inference_delay, prefix_attention_horizon, prefix_actions.shape[1])[:, None]
        pinv_correction = vjp_fun(error)[0]
        # constants from paper
        inv_r2 = (t**2 + (1 - t) ** 2) / (t**2)
        c = jnp.nan_to_num(t / (1 - t), posinf=max_guidance_weight)
        guidance_weight = jnp.minimum(c * inv_r2, max_guidance_weight)
        return v_t - guidance_weight * pinv_correction

    return _pinv_corrected_velocity(x_t, prefix_actions)

def pinv_corrected_velocity(obs, x_t, y, t):
    def denoiser(x_t):
        v_t = self(obs[None], x_t[None], t)[0]
        return x_t + v_t * (1 - t), v_t

    x_1, vjp_fun, v_t = jax.vjp(denoiser, x_t, has_aux=True)
    weights = get_prefix_weights(
        inference_delay, prefix_attention_horizon, self.action_chunk_size, prefix_attention_schedule
    )
    error = (y - x_1) * weights[:, None]
    pinv_correction = vjp_fun(error)[0]
    # constants from paper
    inv_r2 = (t**2 + (1 - t) ** 2) / ((1 - t) ** 2)
    c = jnp.nan_to_num((1 - t) / t, posinf=max_guidance_weight)
    guidance_weight = jnp.minimum(c * inv_r2, max_guidance_weight)
    return v_t + guidance_weight * pinv_correction


"""
Code from https://github.com/Physical-Intelligence/real-time-chunking-kinetix/issues/4#issuecomment-3065876338
"""

def sample_actions_rtc(
    self,
    rng: at.KeyArrayLike,
    observation: _model.Observation,
    prefix_actions: jax.Array,
    inference_delay: int,
    prefix_attention_horizon: int,
    max_guidance_weight: float,
    *,
    num_steps: int | at.Int[at.Array, ""] = 10,
) -> _model.Actions:
    observation = _model.preprocess_observation(None, observation, train=False)
    # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
    # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
    dt = -1.0 / num_steps
    batch_size = observation.state.shape[0]
    noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
    
    # Add a batch dim to prefix_actions # e.g. (30, 14) -> (1, 30, 32)
    prefix_actions = prefix_actions[None, ...]
    prefix_actions = jnp.concatenate([prefix_actions, jnp.zeros((batch_size, self.action_horizon, self.action_dim-prefix_actions.shape[-1]))], axis=2)

    # first fill KV cache with a forward pass of the prefix
    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

    def get_prefix_weights(start: int, end: int, total: int, schedule: PrefixAttentionSchedule) -> jax.Array:
        """With start=2, end=6, total=10, the output will be:
        1  1  4/5 3/5 2/5 1/5 0  0  0  0
            ^              ^
            start           end
        `start` (inclusive) is where the chunk starts being allowed to change. `end` (exclusive) is where the chunk stops
        paying attention to the prefix. if start == 0, then the entire chunk is allowed to change. if end == total, then the
        entire prefix is attended to.

        `end` takes precedence over `start` in the sense that, if `end < start`, then `start` is pushed down to `end`. Thus,
        if `end` is 0, then the entire prefix will always be ignored.
        """
        start = jnp.minimum(start, end)
        if schedule == "ones":
            w = jnp.ones(total)
        elif schedule == "zeros":
            w = (jnp.arange(total) < start).astype(jnp.float32)
        elif schedule == "linear" or schedule == "exp":
            w = jnp.clip((start - 1 - jnp.arange(total)) / (end - start + 1) + 1, 0, 1)
            if schedule == "exp":
                w = w * jnp.expm1(w) / (jnp.e - 1)
        else:
            raise ValueError(f"Invalid schedule: {schedule}")
        return jnp.where(jnp.arange(total) >= end, 0, w)

    def pinv_corrected_velocity(
        v_t_fn: Callable, # ([ah ad], float) -> [ah ad]
        x_t: jax.Array, # [b ah ad]
        t: float,
        prefix_actions: jax.Array, # [b ah ad]
        inference_delay: int,
        prefix_attention_horizon: int,
        max_guidance_weight: float,
    ) -> jax.Array: # [b ah ad]
        @jax.vmap
        def _pinv_corrected_velocity(
            x_t: jax.Array, # [ah ad]
            y: jax.Array, # [ah ad]
        ) -> jax.Array: # [ah ad]
            def denoiser(x_t: jax.Array) -> tuple[jax.Array, jax.Array]: # [ah ad]
                v_t = v_t_fn(x_t, t)
                return x_t - v_t * t, v_t

            x_0, vjp_fun, v_t = jax.vjp(denoiser, x_t, has_aux=True)
            error = (y - x_0) * get_prefix_weights(inference_delay, prefix_attention_horizon, prefix_actions.shape[1], 'exp')[:, None]
            pinv_correction = vjp_fun(error)[0]
            # constants from paper
            inv_r2 = (t**2 + (1 - t) ** 2) / (t**2)
            c = jnp.nan_to_num(t / (1 - t), posinf=max_guidance_weight)
            guidance_weight = jnp.minimum(c * inv_r2, max_guidance_weight)
            return v_t - guidance_weight * pinv_correction

        return _pinv_corrected_velocity(x_t, prefix_actions)

    def v_t_step(
            x_t: jax.Array, # [ah ad]
            time: jax.Array, # []
            ):
        # TODO: find better way to support jax.vmap
        x_t = x_t[None, ...]
        time = time[None, ...]

        suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(
            observation, x_t, time
        )
        # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
        # other
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
        # prefix tokens
        prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
        # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
        full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
        assert full_attn_mask.shape == (
            batch_size,
            suffix_tokens.shape[1],
            prefix_tokens.shape[1] + suffix_tokens.shape[1],
        )
        # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens], mask=full_attn_mask, positions=positions, kv_cache=kv_cache
        )
        assert prefix_out is None
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return v_t[0, ...] # TODO: remove this since it's not super vectorized

    def rtc_step(carry):
        x_t, time = carry
        guided_vt = pinv_corrected_velocity(v_t_step, x_t, time, prefix_actions, inference_delay, prefix_attention_horizon, max_guidance_weight)
        return x_t + dt * guided_vt, time + dt

    def cond(carry):
        x_t, time = carry
        # robust to floating-point error
        return time >= -dt / 2

    x_0, _ = jax.lax.while_loop(cond, rtc_step, (noise, 1.0))
    return x_0