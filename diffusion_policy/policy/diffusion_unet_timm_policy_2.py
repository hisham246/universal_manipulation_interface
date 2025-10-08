import sched
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.timm_obs_encoder import TimmObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.flow_adapter_1 import DiffusionAsFlow

def taus_from_scheduler(scheduler, device, dtype):
    """
    Return τ for every id in `scheduler.timesteps`, mirroring diffusers' handling of the last step.
    τ = 1 - η, where η = s/(a+s), a = sqrt(ᾱ), s = sqrt(1-ᾱ).
    """
    # Gather ᾱ on target device/dtype
    ab_full = scheduler.alphas_cumprod.to(device=device, dtype=dtype)  # [T_train]
    tids = scheduler.timesteps.to(device='cpu')  # ids used for this inference pass
    taus = []

    # Identify the last inference id
    last_id = int(tids[-1].item()) if len(tids) > 0 else 0
    use_final = bool(getattr(scheduler.config, "set_alpha_to_one", False))
    fa = getattr(scheduler, "final_alpha_cumprod", None)  # e.g., 1.0 for DDIM

    for t in tids.tolist():
        if use_final and t == last_id and (fa is not None):
            ab_t = torch.as_tensor(fa, device=device, dtype=dtype)
        else:
            ab_t = ab_full[int(t)]
        # numerics
        ab_t = torch.clamp(ab_t, 1e-12, 1.0)  # keep in (0,1]
        a = torch.sqrt(ab_t)
        s = torch.sqrt(torch.clamp(1.0 - ab_t, 0.0, 1.0))
        eta = s / (a + s + 1e-12)
        tau = 1.0 - eta
        taus.append(tau)

    return torch.stack(taus)  # [len(timesteps)]

class DiffusionUnetTimmPolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            obs_encoder: TimmObsEncoder,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            input_pertub=0.1,
            inpaint_fixed_action_prefix=False,
            train_diffusion_n_samples=1,
            # parameters passed to step
            **kwargs
        ):
        super().__init__()

        # parse shapes
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        action_horizon = shape_meta['action']['horizon']
        # get feature dim
        obs_feature_dim = np.prod(obs_encoder.output_shape())


        # create diffusion model
        assert obs_as_global_cond
        input_dim = action_dim
        global_cond_dim = obs_feature_dim

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon # used for training
        self.obs_as_global_cond = obs_as_global_cond
        self.input_pertub = input_pertub
        self.inpaint_fixed_action_prefix = inpaint_fixed_action_prefix
        self.train_diffusion_n_samples = int(train_diffusion_n_samples)
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        self.flow_adapter = DiffusionAsFlow(
            model=self.model,
            noise_scheduler=self.noise_scheduler,
            pred_type=self.noise_scheduler.config.prediction_type  # 'epsilon' or 'sample'
        )
        self.flow_n_steps = int(self.num_inference_steps)  # good default; can tune later for latency/quality

    # ========= inference  ============
    def conditional_sample(self, 
            condition_data,
            condition_mask,
            local_cond=None,
            global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
        ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor], fixed_action_prefix: torch.Tensor=None) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        fixed_action_prefix: unnormalized action prefix
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # print("Normalizer keys:", list(self.normalizer.params_dict.keys()))
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0]

        # condition through global feature
        global_cond = self.obs_encoder(nobs)

        # empty data for action
        cond_data = torch.zeros(size=(B, self.action_horizon, self.action_dim), device=self.device, dtype=self.dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        if fixed_action_prefix is not None and self.inpaint_fixed_action_prefix:
            n_fixed_steps = fixed_action_prefix.shape[1]
            cond_data[:, :n_fixed_steps] = fixed_action_prefix
            cond_mask[:, :n_fixed_steps] = True
            cond_data = self.normalizer['action'].normalize(cond_data)


        # run sampling
        nsample = self.conditional_sample(
            condition_data=cond_data, 
            condition_mask=cond_mask,
            local_cond=None,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        assert nsample.shape == (B, self.action_horizon, self.action_dim)
        action_pred = self.normalizer['action'].unnormalize(nsample)
        
        result = {
            'action': action_pred,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        
        assert self.obs_as_global_cond
        global_cond = self.obs_encoder(nobs)

        # train on multiple diffusion samples per obs
        if self.train_diffusion_n_samples != 1:
            # repeat obs features and actions multiple times along the batch dimension
            # each sample will later have a different noise sample, effecty training 
            # more diffusion steps per each obs encoder forward pass
            global_cond = torch.repeat_interleave(global_cond, 
                repeats=self.train_diffusion_n_samples, dim=0)
            nactions = torch.repeat_interleave(nactions, 
                repeats=self.train_diffusion_n_samples, dim=0)

        trajectory = nactions
        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        # input perturbation by adding additonal noise to alleviate exposure bias
        # reference: https://github.com/forever208/DDPM-IP
        noise_new = noise + self.input_pertub * torch.randn(trajectory.shape, device=trajectory.device)

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (nactions.shape[0],), device=trajectory.device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise_new, timesteps)
        
        # Predict the noise residual
        pred = self.model(
            noisy_trajectory,
            timesteps, 
            local_cond=None,
            global_cond=global_cond
        )

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()

        return loss
    
    # Flow Policy
    # @torch.no_grad()
    # def sample_chunk_flow(self, global_cond, generator=None, n_steps=None):
    #     """
    #     Deterministic flow sampler.
    #     Iterates reverse diffusion indices (high->low) and uses z <- z - (1/n) * v,
    #     with v = (ε - x0), which is equivalent to stepping + (x0 - ε).
    #     Returns a normalized action chunk [B, H, D].
    #     """
    #     if n_steps is None: n_steps = self.flow_n_steps
    #     B = global_cond.shape[0]
    #     H, D = self.action_horizon, self.action_dim

    #     # Start from standard Normal in normalized action space
    #     z = torch.randn((B, H, D), dtype=self.dtype, device=global_cond.device, generator=generator)

    #     # Use the SAME discrete scheduler timesteps you already use for DDIM/DDPM
    #     self.noise_scheduler.set_timesteps(n_steps)
    #     for k, t_id in enumerate(self.noise_scheduler.timesteps):
    #         # one Euler step on the flow ODE
    #         v = self.flow_adapter.velocity(z, t_id, global_cond)   # [B,H,D]
    #         z = z + (1.0 / n_steps) * v

    #     return z  # still normalized; unnormalize in predict_action

    @torch.no_grad()
    def _fm_ddim_step(self, z_t, t_id, t_id_next, global_cond):
        # α, σ, η, τ at current/next ids
        a_t, s_t, eta_t, tau_t = self.flow_adapter._alpha_sigma_eta(int(t_id),   z_t.device, z_t.dtype)
        a_s, s_s, eta_s, tau_s = self.flow_adapter._alpha_sigma_eta(int(t_id_next), z_t.device, z_t.dtype)

        # model → (ε, x0)
        v = self.flow_adapter.velocity(z_t, t_id, global_cond)   # your preferred direction

        # Euler in reparametrized state ẑ := z / (α+σ), step by Δη = (η_t - η_s) ≥ 0
        delta = (eta_t - eta_s)                               # positive along backward ids
        z_tilde = z_t / (a_t + s_t)
        z_tilde_next = z_tilde + v * delta
        z_next = z_tilde_next * (a_s + s_s)

        return z_next, float(tau_t), float(tau_s), v

    @torch.no_grad()
    def sample_chunk_flow(self, global_cond, generator=None, n_steps=None):
        """
        OT flow sampler using forward integration.
        Uses backward scheduler timesteps (45→0) but forward flow time (0.1→1.0).
        """
        if n_steps is None: 
            n_steps = self.flow_n_steps
        B = global_cond.shape[0]
        H, D = self.action_horizon, self.action_dim

        # Start from standard Normal (noise)
        z = torch.randn((B, H, D), dtype=self.dtype, device=global_cond.device, generator=generator)

        # Get scheduler timesteps (backward: 45→0)
        self.noise_scheduler.set_timesteps(n_steps)

        t_seq = self.noise_scheduler.timesteps
        taus  = taus_from_scheduler(self.noise_scheduler, device=z.device, dtype=z.dtype)   # [N]
        # Build ᾱ sequence aligned with t_seq, applying the same final override as diffusers
        ab_full = self.noise_scheduler.alphas_cumprod.to(z.device, z.dtype)
        ab_seq = []
        last_id = int(t_seq[-1].item())
        use_final = bool(getattr(self.noise_scheduler.config, "set_alpha_to_one", False))
        fa = getattr(self.noise_scheduler, "final_alpha_cumprod", None)
        for tid in t_seq.tolist():
            if use_final and tid == last_id and (fa is not None):
                ab_seq.append(torch.as_tensor(fa, device=z.device, dtype=z.dtype))
            else:
                ab_seq.append(ab_full[int(tid)])
        ab_seq = torch.stack(ab_seq)  # [N]

        def _as_from_ab(ab):
            ab = torch.clamp(ab, 1e-12, 1.0)
            a  = ab.sqrt()
            s  = torch.sqrt(torch.clamp(1.0 - ab, 0.0, 1.0))
            return a, s

        print(f"\n{'Step':<5} {'t':<5} {'s':<5} {'tau':<8} {'z_before':<10} {'|v|':<10} {'z_after':<10}")
        print("-"*70)

        for k in range(len(t_seq)-1):
            t = int(t_seq[k].item())
            s = int(t_seq[k+1].item())
            tau_t = float(taus[k].item())
            tau_s = float(taus[k+1].item())

            # ᾱ→(a,s) for current/next ids, using the overridden sequence
            a_t, s_t = _as_from_ab(ab_seq[k])
            a_s, s_s = _as_from_ab(ab_seq[k+1])

            # Δη = η_t - η_s = (τ_s - τ_t)  (positive along backward ids)
            delta_eta = (tau_s - tau_t)

            # velocity from your adapter at id t
            t_b = torch.full((z.shape[0],), t, dtype=torch.long, device=z.device)
            v = self.flow_adapter.velocity(z, t_b, global_cond)   # [B,H,D]

            # FM/DDIM step in reparametrized space ẑ = z / (a+σ)
            z_before = z.norm().item()
            z_tilde = z / (a_t + s_t)
            z_tilde_next = z_tilde + v * delta_eta
            z = (z_tilde_next * (a_s + s_s)).detach()
            z_after = z.norm().item()

            if k % max(1, n_steps // 10) == 0 or k == len(t_seq)-2:
                print(f"{k:<5} {t:<5} {s:<5} {tau_t:<8.4f} {z_before:<10.4f} {v.norm().item():<10.4f} {z_after:<10.4f}")

        # final τ row (use taus[-1] so it respects final_alpha_cumprod)
        tau_final = float(taus[-1].item())
        print(f"{len(t_seq)-1:<5} {int(t_seq[-1]):<5} {'—':<5} {tau_final:<8.4f} "
            f"{z.norm().item():<10.4f} {'-':<10} {z.norm().item():<10.4f}")


        with torch.no_grad():
            sched = self.noise_scheduler
            unet  = self.model
            cond  = global_cond
            t_seq = sched.timesteps

            # # disable clipping for fair parity
            # clip_backup = sched.config.clip_sample
            # sched.config.clip_sample = False

            # init (reuse dtype/device)
            z0 = torch.randn((cond.shape[0], self.action_horizon, self.action_dim),
                            dtype=self.dtype, device=cond.device)
            z_ddim = z0.clone()
            z_fm   = z0.clone()

            # prefetch alphas table once
            ab = sched.alphas_cumprod.to(cond.device, z0.dtype)

            # # A) DDIM (eta=0)
            # for t in t_seq:
            #     t_idx = int(t.item())
            #     t_b = torch.full((z_ddim.shape[0],), t_idx, dtype=torch.long, device=z_ddim.device)
            #     eps = unet(z_ddim, t_b, local_cond=None, global_cond=cond)

            #     t_cpu = torch.full((z_ddim.shape[0],), t_idx, dtype=torch.long, device='cpu')
            #     z_ddim = sched.step(eps, t_cpu, z_ddim, eta=0.0).prev_sample   # <- CPU tensor index

            # B) FM pairwise + explicit DDIM step parity
            max_step_abs = 0.0
            z_tmp = z0.clone()
            for k in range(len(t_seq)-1):
                t = int(t_seq[k].item()); s = int(t_seq[k+1].item())

                # FM step (your conversion)
                z_fm, *_ = self._fm_ddim_step(z_fm, t, s, cond)

                # Explicit DDIM step from z_tmp
                sqrtab_t   = ab[t].sqrt(); sqrt1mab_t = (1.0 - ab[t]).sqrt()
                sqrtab_s   = ab[s].sqrt(); sqrt1mab_s = (1.0 - ab[s]).sqrt()
                t_b = torch.full((z_tmp.shape[0],), t, dtype=torch.long, device=z_tmp.device)
                eps_t = unet(z_tmp, t_b, local_cond=None, global_cond=cond)
                x0_hat = (z_tmp - sqrt1mab_t * eps_t) / sqrtab_t
                z_prev_ddim = sqrtab_s * x0_hat + sqrt1mab_s * eps_t
                z_tmp = z_prev_ddim  # advance explicit-DDIM state

                # per-step diff
                step_abs = (z_prev_ddim - z_fm).abs().max().item()
                max_step_abs = max(max_step_abs, step_abs)
                if step_abs > 1e-5:
                    print(f"[step {k} t={t}->s={s}] max|Δ| = {step_abs:.3e}")

            l2 = (z_tmp - z_fm).pow(2).sum().sqrt().item()
            print(f"[DDIM parity] L2(explicit_ddim, z_fm) = {l2:.3e} | max per-step abs = {max_step_abs:.3e}")

            # restore
            # sched.config.clip_sample = clip_backup


                
        # # dt for flow integration
        # dt = 1.0 / n_steps
        
        # print(f"\n{'='*60}")
        # print(f"FLOW SAMPLING DEBUG")
        # print(f"{'='*60}")
        # print(f"n_steps: {n_steps}, dt: {dt:.4f}")
        # print(f"Initial z (noise): norm={z.norm().item():.4f}")
        
        # timesteps = self.noise_scheduler.timesteps
        # print(f"Timesteps: {timesteps[:3].cpu().numpy()}...{timesteps[-3:].cpu().numpy()}")
        # print(f"Direction: {'BACKWARD' if timesteps[0] > timesteps[-1] else 'FORWARD'}")
        
        # print(f"\n{'Step':<5} {'t_id':<6} {'time':<8} {'z_norm':<10} {'v_norm':<10} {'x0_norm':<10}")
        # print("-" * 60)

        # t_seq = self.noise_scheduler.timesteps  # e.g., [45, 42, ..., 0]

        # print(f"\n{'Step':<5} {'t':<5} {'t->s':<8} {'|z|':<10} {'|v|':<10} {'τ':<8}")
        # print("-"*60)

        # for k in range(len(t_seq)-1):
        #     t = int(t_seq[k].item()); s = int(t_seq[k+1].item())
        #     z, tau_t, tau_s, v = self._fm_ddim_step(z, t, s, global_cond)
        #     if k % max(1, n_steps // 5) == 0 or k == len(t_seq)-2:
        #         print(f"{k:<5} {t:<5} {s:<8} {z.norm().item():<10.4f} {v.norm().item():<10.4f} {tau_s:<8.4f}")

        # t_final = int(t_seq[-1].item())
        # _, _, _, tau_final = self.flow_adapter._alpha_sigma_eta(
        #     t_final, z.device, z.dtype
        # )

        # # optional: show a final row aligned with your table
        # print(f"{len(t_seq)-1:<5} {t_final:<5} {'—':<5} {tau_final:<8.4f} "
        #     f"{z.norm().item():<10.4f} {'-':<10} {'-':<10} "
        #     f"{z.norm().item():<10.4f} {'-':<10}")
        


        
        # for k, t_id in enumerate(self.noise_scheduler.timesteps):
        #     # Get continuous flow time
        #     time = self.flow_adapter._discrete_timestep_to_continuous_time(t_id, z.device).item()
            
        #     # Get velocity and intermediate predictions
        #     eps_hat, x0_hat = self.flow_adapter._eps_x0_from_model(z, t_id, global_cond)
        #     v = self.flow_adapter.velocity(z, t_id, global_cond)
            
        #     # Euler integration step
        #     z = z + dt * v
            
        #     # Print progress
        #     if k % max(1, n_steps // 5) == 0 or k == n_steps - 1:
        #         print(f"{k:<5} {t_id.item():<6} {time:<8.4f} {z.norm().item():<10.4f} "
        #             f"{v.norm().item():<10.4f} {x0_hat.norm().item():<10.4f}")
        
        print(f"\nFinal z (clean): norm={z.norm().item():.4f}")
        print(f"{'='*60}\n")
        
        return z  # normalized; unnormalize in predict_action

    def predict_action_flow(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert 'past_action' not in obs_dict
        nobs = self.normalizer.normalize(obs_dict)
        global_cond = self.obs_encoder(nobs)  # [B, feat]

        # Flow sampling in normalized space, then unnormalize
        nsample = self.sample_chunk_flow(global_cond=global_cond, n_steps=self.flow_n_steps)
        action_pred = self.normalizer['action'].unnormalize(nsample)

        return {'action': action_pred, 'action_pred': action_pred}
    
    # def realtime_action(
    #     self,
    #     obs_dict: Dict[str, torch.Tensor],
    #     prev_action_chunk: torch.Tensor,
    #     inference_delay: int,
    #     prefix_attention_horizon: int,
    #     prefix_attention_schedule: str = "exp",
    #     max_guidance_weight: float = 5.0,
    #     n_steps: int = None,
    # ) -> Dict[str, torch.Tensor]:
    #     """Real-time chunking with inpainting."""
    #     if n_steps is None:
    #         n_steps = self.flow_n_steps
        
    #     nobs = self.normalizer.normalize(obs_dict)
    #     global_cond = self.obs_encoder(nobs)
    #     B = global_cond.shape[0]
    #     H, D = self.action_horizon, self.action_dim
        
    #     # Normalize previous chunk
    #     y_prev = self.normalizer['action'].normalize(prev_action_chunk)
        
    #     # Compute soft masking weights
    #     weights = self._get_prefix_weights(
    #         inference_delay, 
    #         prefix_attention_horizon, 
    #         H, 
    #         prefix_attention_schedule
    #     )
        
    #     # Start from noise
    #     z = torch.randn((B, H, D), dtype=self.dtype, device=self.device)

    #     print(f"\n{'='*60}")
    #     print(f"FLOW INTEGRATION DEBUG (prediction_type={self.noise_scheduler.config.prediction_type})")
    #     print(f"{'='*60}")
    #     print(f"n_steps: {n_steps}, dt: {1.0/n_steps:.4f}")
    #     print(f"Initial z (noise) norm: {z.norm().item():.4f}")
        
    #     # Flow integration with guidance
    #     self.noise_scheduler.set_timesteps(n_steps)
    #     dt = 1.0 / n_steps

    #     timesteps = self.noise_scheduler.timesteps.cpu().numpy()
    #     print(f"Timesteps: {timesteps[:3]}...{timesteps[-3:]}")
    #     print("Timesteps length:", len(timesteps))
    #     print(f"Direction: {'HIGH→LOW (backward)' if timesteps[0] > timesteps[-1] else 'LOW→HIGH (forward)'}")

    #     alphas = self.noise_scheduler.alphas_cumprod.cpu().numpy()
    #     print(f"\nAlpha_bar at first timestep ({timesteps[0]}): {alphas[timesteps[0]]:.6f}")
    #     print(f"Alpha_bar at last timestep ({timesteps[-1]}): {alphas[timesteps[-1]]:.6f}")
        
    #     for k, t_id in enumerate(self.noise_scheduler.timesteps):
    #         # tau = k / n_steps
            
    #         alpha_bar_t = self.noise_scheduler.alphas_cumprod[t_id].item()
    #         tau = alpha_bar_t  # 0 at noise, ~1 at data

    #         # Check what the model predicts
    #         with torch.no_grad():
    #             eps_hat, x0_hat = self.flow_adapter._eps_x0_from_model(
    #                 z, t_id, global_cond
    #             )
    #             v = self.flow_adapter.velocity(z, t_id, global_cond)
            
    #         z_norm_before = z.norm().item()
    #         eps_norm = eps_hat.norm().item()
    #         x0_norm = x0_hat.norm().item()
    #         v_norm = v.norm().item()
            
    #         # Compute guided velocity
    #         v_guided = self._pinv_corrected_velocity(
    #             z, t_id, global_cond, y_prev, weights, tau, max_guidance_weight
    #         )
            
    #         # Euler step and detach to break graph
    #         z = (z + dt * v_guided).detach()
            
    #         z_norm_after = z.norm().item()
            
    #         # Print row
    #         print(f"{k:<5} {t_id.item():<5} {tau:<7.4f} {alpha_bar_t:<10.6f} {z_norm_before:<8.4f} {eps_norm:<9.4f} {x0_norm:<8.4f} {v_norm:<8.4f} {z_norm_after:<8.4f}")
            
    #         # # Compute guided velocity
    #         # v_guided = self._pinv_corrected_velocity(
    #         #     z, t_id, global_cond, y_prev, weights, tau, max_guidance_weight
    #         # )
            
    #         # # Euler step and detach to break graph
    #         # z = (z + dt * v_guided).detach()

    #         # if k in [0, n_steps//2, n_steps-1]:
    #         #     print(f"  After step (z + dt*v): {z.norm().item():.4f}")
        
    #     print(f"\nFinal z norm: {z.norm().item():.4f}")
    #     print(f"Expected range for normalized actions: ~1-2")
    #     print(f"{'='*60}\n")
    #     # Unnormalize
    #     action_pred = self.normalizer['action'].unnormalize(z)
    #     return {'action': action_pred, 'action_pred': action_pred}

    # def realtime_action(
    #     self,
    #     obs_dict: Dict[str, torch.Tensor],
    #     prev_action_chunk: torch.Tensor,
    #     inference_delay: int,
    #     prefix_attention_horizon: int,
    #     prefix_attention_schedule: str = "exp",
    #     max_guidance_weight: float = 5.0,
    #     n_steps: int = None,
    # ) -> Dict[str, torch.Tensor]:
    #     """Real-time chunking with inpainting."""
    #     if n_steps is None:
    #         n_steps = self.flow_n_steps
        
    #     nobs = self.normalizer.normalize(obs_dict)
    #     global_cond = self.obs_encoder(nobs)
    #     B = global_cond.shape[0]
    #     H, D = self.action_horizon, self.action_dim
        
    #     # Normalize previous chunk
    #     y_prev = self.normalizer['action'].normalize(prev_action_chunk)
        
    #     # Compute soft masking weights
    #     weights = self._get_prefix_weights(
    #         inference_delay, 
    #         prefix_attention_horizon, 
    #         H, 
    #         prefix_attention_schedule
    #     )
        
    #     # Start from noise
    #     z = torch.randn((B, H, D), dtype=self.dtype, device=self.device)

    #     print(f"\n{'='*60}")
    #     print(f"RTC FLOW INTEGRATION (prediction_type={self.noise_scheduler.config.prediction_type})")
    #     print(f"{'='*60}")
        
    #     # Flow integration with guidance
    #     self.noise_scheduler.set_timesteps(n_steps)

    #     taus = taus_from_scheduler(self.noise_scheduler, device=z.device, dtype=z.dtype)

    #     dt = 1.0 / n_steps

    #     timesteps = self.noise_scheduler.timesteps
    #     print(f"n_steps: {n_steps}, dt: {dt:.4f}")
    #     print(f"Timesteps: {timesteps[:3].cpu().numpy()}...{timesteps[-3:].cpu().numpy()}")
    #     print(f"Direction: {'BACKWARD (high→low)' if timesteps[0] > timesteps[-1] else 'FORWARD (low→high)'}")
    #     print(f"Initial z norm: {z.norm().item():.4f}")
        
    #     print(f"\n{'Step':<5} {'t_id':<5} {'time':<8} {'z_before':<10} {'v_norm':<10} {'z_after':<10} {'guidance':<10}")
    #     print("-" * 75)

    #     t_seq = self.noise_scheduler.timesteps
    #     print(f"\n{'Step':<5} {'t':<5} {'s':<5} {'tau':<8} {'z_before':<10} {'|v_base|':<10} {'|v_corr|':<10} {'z_after':<10} {'guidance':<10}")
    #     print("-"*95)

    #     for k in range(len(t_seq)-1):
    #         t = int(t_seq[k].item()); s = int(t_seq[k+1].item())

    #         # --- before norms
    #         z_before = z.norm().item()

    #         # base FM/DDIM proposal
    #         z_prop, tau_t, tau_s, v_base = self._fm_ddim_step(z, t, s, global_cond)

    #         # ΠGDM correction at current state/time τ_t
    #         v_corr = self._pinv_corrected_velocity(
    #             z, torch.tensor(t), global_cond, y_prev, weights, tau_t, max_guidance_weight
    #         )

    #         # Δη = (η_t-η_s) = (τ_s-τ_t)
    #         delta = (tau_s - tau_t)

    #         # apply correction in reparam space (same as your code)
    #         a_t, s_t, _, _ = self.flow_adapter._alpha_sigma_eta(t, z.device, z.dtype)
    #         a_s, s_s, _, _ = self.flow_adapter._alpha_sigma_eta(s, z.device, z.dtype)
    #         z_tilde = z / (a_t + s_t)
    #         z_tilde_corr = z_tilde + v_corr * delta
    #         z = (z_tilde_corr * (a_s + s_s)).detach()

    #         # --- after norms
    #         z_after = z.norm().item()

    #         # guidance weight (same formula you use in _pinv_corrected_velocity)
    #         inv_r2 = (tau_t**2 + (1 - tau_t)**2) / ((1 - tau_t)**2)
    #         c = (1 - tau_t) / max(tau_t, 1e-8)   # protect τ=0
    #         guidance = min(c * inv_r2, max_guidance_weight)

    #         if k % max(1, n_steps // 10) == 0 or k == len(t_seq)-2:
    #             print(f"{k:<5} {t:<5} {s:<5} {tau_t:<8.4f} {z_before:<10.4f} "
    #                 f"{v_base.norm().item():<10.4f} {v_corr.norm().item():<10.4f} "
    #                 f"{z_after:<10.4f} {guidance:<10.4f}")

    #     t_final = int(t_seq[-1].item())
    #     _, _, _, tau_final = self.flow_adapter._alpha_sigma_eta(
    #         t_final, z.device, z.dtype
    #     )

    #     # optional: show a final row aligned with your table
    #     print(f"{len(t_seq)-1:<5} {t_final:<5} {'—':<5} {tau_final:<8.4f} "
    #         f"{z.norm().item():<10.4f} {'-':<10} {'-':<10} "
    #         f"{z.norm().item():<10.4f} {'-':<10}")
    #     # for k, t_id in enumerate(self.noise_scheduler.timesteps):
    #     #     # Get continuous time for this timestep
    #     #     time = self.flow_adapter._discrete_timestep_to_continuous_time(t_id, z.device)
    #     #     time_val = time.item()
            
    #     #     z_norm_before = z.norm().item()
            
    #     #     # Compute guided velocity
    #     #     v_guided = self._pinv_corrected_velocity(
    #     #         z, t_id, global_cond, y_prev, weights, time_val, max_guidance_weight
    #     #     )
            
    #     #     v_norm = v_guided.norm().item()
            
    #     #     # Euler step
    #     #     z = (z + dt * v_guided).detach()
            
    #     #     z_norm_after = z.norm().item()
            
    #     #     # Print every few steps
    #     #     if k % max(1, n_steps // 10) == 0 or k == n_steps - 1:
    #     #         # Get guidance weight for display
    #     #         inv_r2 = (time_val**2 + (1 - time_val)**2) / ((1 - time_val)**2)
    #     #         c = (1 - time_val) / (time_val)
    #     #         guidance_weight = min(c * inv_r2, max_guidance_weight)
                
    #     #         print(f"{k:<5} {t_id.item():<5} {time_val:<8.4f} {z_norm_before:<10.4f} "
    #     #             f"{v_norm:<10.4f} {z_norm_after:<10.4f} {guidance_weight:<10.4f}")
        
    #     print(f"\nFinal z norm: {z.norm().item():.4f}")
    #     print(f"{'='*60}\n")
        
    #     # Unnormalize
    #     action_pred = self.normalizer['action'].unnormalize(z)
    #     return {'action': action_pred, 'action_pred': action_pred}

    @torch.no_grad()
    def realtime_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        prev_action_chunk: torch.Tensor,
        inference_delay: int,
        prefix_attention_horizon: int,
        prefix_attention_schedule: str = "exp",
        max_guidance_weight: float = 5.0,
        n_steps: int = None,
    ) -> Dict[str, torch.Tensor]:
        if n_steps is None:
            n_steps = self.flow_n_steps

        # Encode + normalize
        nobs = self.normalizer.normalize(obs_dict)
        global_cond = self.obs_encoder(nobs)
        B = global_cond.shape[0]
        H, D = self.action_horizon, self.action_dim
        y_prev = self.normalizer['action'].normalize(prev_action_chunk)

        # weights for guidance
        weights = self._get_prefix_weights(
            inference_delay, prefix_attention_horizon, H, prefix_attention_schedule
        )

        # init latent on the correct device/dtype
        z = torch.randn((B, H, D), dtype=self.dtype, device=global_cond.device)

        print(f"\n{'='*60}")
        print(f"RTC FLOW INTEGRATION (prediction_type={self.noise_scheduler.config.prediction_type})")
        print(f"{'='*60}")

        # Scheduler timesteps and taus
        sched = self.noise_scheduler
        sched.set_timesteps(n_steps)
        t_seq = sched.timesteps
        taus  = taus_from_scheduler(sched, device=z.device, dtype=z.dtype)  # [N]

        # Build ᾱ sequence aligned with t_seq, honoring set_alpha_to_one/final_alpha_cumprod
        ab_full = sched.alphas_cumprod.to(z.device, z.dtype)
        ab_seq = []
        last_id = int(t_seq[-1].item())
        use_final = bool(getattr(sched.config, "set_alpha_to_one", False))
        fa = getattr(sched, "final_alpha_cumprod", None)

        for tid in t_seq.tolist():
            if use_final and tid == last_id and (fa is not None):
                ab_seq.append(torch.as_tensor(fa, device=z.device, dtype=z.dtype))
            else:
                ab_seq.append(ab_full[int(tid)])
        ab_seq = torch.stack(ab_seq)  # [N]

        def _as_from_ab(ab_t: torch.Tensor):
            ab_t = torch.clamp(ab_t, 1e-12, 1.0)
            a = ab_t.sqrt()
            s = torch.sqrt(torch.clamp(1.0 - ab_t, 0.0, 1.0))
            return a, s

        # Pretty logging
        print(f"n_steps: {n_steps}, dt (Δη avg): ~{(taus[-1]-taus[0]).item()/max(1,(len(t_seq)-1)):.4f}")
        print(f"Timesteps: {t_seq[:3].cpu().numpy()}...{t_seq[-3:].cpu().numpy()}")
        print(f"Direction: {'BACKWARD (high→low)' if t_seq[0] > t_seq[-1] else 'FORWARD (low→high)'}")
        print(f"Initial z norm: {z.norm().item():.4f}\n")
        print(f"{'Step':<5} {'t':<5} {'s':<5} {'tau':<8} {'z_before':<10} {'|v_base|':<10} {'|v_corr|':<10} {'z_after':<10} {'guidance':<10}")
        print("-"*95)

        # FM/DDIM + ΠGDM in the same (a,s,τ) geometry as sample_chunk_flow
        for k in range(len(t_seq)-1):
            t = int(t_seq[k].item()); s = int(t_seq[k+1].item())
            tau_t = float(taus[k].item()); tau_s = float(taus[k+1].item())
            delta_eta = tau_s - tau_t  # Δη

            # before
            z_before = z.norm().item()

            # (a,s) for current/next indices from ab_seq
            a_t, s_t = _as_from_ab(ab_seq[k])
            a_s, s_s = _as_from_ab(ab_seq[k+1])

            # base velocity at id t (device-safe long tensor)
            t_b = torch.full((B,), t, dtype=torch.long, device=z.device)
            v_base = self.flow_adapter.velocity(z, t_b, global_cond)

            # ΠGDM correction at τ_t (pass the same t_b; _pinv_... expects torch.LongTensor for t_id)
            v_corr = self._pinv_corrected_velocity(
                z, t_b, global_cond, y_prev, weights, tau_t, max_guidance_weight
            )

            # Reparam step in ẑ with corrected velocity
            z_tilde = z / (a_t + s_t)
            z_tilde_corr = z_tilde + v_corr * delta_eta
            z = (z_tilde_corr * (a_s + s_s)).detach()

            # after + guidance number (same formula as inside _pinv_corrected_velocity)
            z_after = z.norm().item()
            inv_r2 = (tau_t**2 + (1 - tau_t)**2) / ((1 - tau_t)**2 + 1e-12)
            c = (1 - tau_t) / (tau_t + 1e-12)
            guidance = min(c * inv_r2, max_guidance_weight)

            if k % max(1, n_steps // 10) == 0 or k == len(t_seq)-2:
                print(f"{k:<5} {t:<5} {s:<5} {tau_t:<8.4f} {z_before:<10.4f} "
                    f"{v_base.norm().item():<10.4f} {v_corr.norm().item():<10.4f} "
                    f"{z_after:<10.4f} {guidance:<10.4f}")

        # final τ row (use taus[-1] which respects final_alpha_cumprod)
        print(f"{len(t_seq)-1:<5} {int(t_seq[-1]):<5} {'—':<5} {float(taus[-1].item()):<8.4f} "
            f"{z.norm().item():<10.4f} {'-':<10} {'-':<10} {z.norm().item():<10.4f} {'-':<10}")

        print(f"\nFinal z norm: {z.norm().item():.4f}")
        print(f"{'='*60}\n")

        # unnormalize back to action space
        action_pred = self.normalizer['action'].unnormalize(z)
        return {'action': action_pred, 'action_pred': action_pred}

    # def _pinv_corrected_velocity(
    #     self,
    #     z_t: torch.Tensor,
    #     t_id: torch.Tensor,
    #     global_cond: torch.Tensor,
    #     y_prev: torch.Tensor,
    #     weights: torch.Tensor,
    #     tau: float,
    #     max_guidance_weight: float
    # ) -> torch.Tensor:
    #     """ΠGDM-corrected velocity (Eq. 2 from paper)."""
    #     B, H, D = z_t.shape
        
    #     # Detach global_cond to prevent graph accumulation
    #     global_cond_detached = global_cond.detach()
        
    #     # Create fresh tensor with gradients enabled
    #     z_t_copy = z_t.detach().clone().requires_grad_(True)
        
    #     # Compute velocity and denoiser with isolated graph
    #     with torch.enable_grad():
    #         v_t = self.flow_adapter.velocity(z_t_copy, t_id, global_cond_detached)
    #         x_1_hat = z_t_copy + (1 - tau) * v_t
        
    #     # Weighted error
    #     error = (y_prev - x_1_hat) * weights[None, :, None]  # [B, H, D]
        
    #     # Compute gradient
    #     if z_t_copy.grad is not None:
    #         z_t_copy.grad.zero_()
        
    #     x_1_hat.backward(error)
    #     pinv_correction = z_t_copy.grad.detach().clone()
        
    #     # Guidance weight (Eq. 4)
    #     inv_r2 = (tau**2 + (1 - tau)**2) / ((1 - tau)**2 + 1e-12)
    #     c = (1 - tau) / (tau + 1e-12)
    #     guidance_weight = min(c * inv_r2, max_guidance_weight)
        
    #     # Return corrected velocity (everything detached)
    #     return v_t.detach() + guidance_weight * pinv_correction

    def _pinv_corrected_velocity(
        self,
        z_t: torch.Tensor,
        t_id: torch.Tensor,
        global_cond: torch.Tensor,
        y_prev: torch.Tensor,
        weights: torch.Tensor,
        time: float,  # Now receives continuous time directly
        max_guidance_weight: float
    ) -> torch.Tensor:
        """
        ΠGDM-corrected velocity for RTC (Pokle et al. Eq. 2).
        
        Args:
            time: Continuous flow time ∈ [0, 1], where 0=noise, 1=data
        """
        B, H, D = z_t.shape
        
        # Detach to prevent graph accumulation
        global_cond_detached = global_cond.detach()
        z_t_copy = z_t.detach().clone().requires_grad_(True)
        
        # Compute velocity and predicted clean action
        with torch.enable_grad():
            v_t = self.flow_adapter.velocity(z_t_copy, t_id, global_cond_detached)
            # Denoising estimate: x_1 = z_t + (1-time) * v_t
            # This is Â_c^1 from Pokle Eq. 3
            x_1_hat = z_t_copy + (1 - time) * v_t
        
        # Weighted error against previous chunk
        error = (y_prev - x_1_hat) * weights[None, :, None]  # [B, H, D]
        
        # Compute gradient via backprop
        if z_t_copy.grad is not None:
            z_t_copy.grad.zero_()
        
        x_1_hat.backward(error)
        pinv_correction = z_t_copy.grad.detach().clone()
        
        # Guidance weight (Pokle Eq. 2 and Eq. 4)
        inv_r2 = (time**2 + (1 - time)**2) / ((1 - time)**2)
        c = (1 - time) / (time)
        guidance_weight = min(c * inv_r2, max_guidance_weight)

        # Return corrected velocity
        return v_t.detach() + guidance_weight * pinv_correction
    
    def _get_prefix_weights(
        self,
        start: int,
        end: int,
        total: int,
        schedule: str
    ) -> torch.Tensor:
        """Soft masking weights (Eq. 5 from paper)."""
        start = min(start, end)
        indices = torch.arange(total, dtype=torch.float32, device=self.device)
        
        if schedule == "ones":
            w = torch.ones(total, device=self.device)
        elif schedule == "zeros":
            w = (indices < start).float()
        elif schedule == "linear" or schedule == "exp":
            w = torch.clamp((start - 1 - indices) / (end - start + 1) + 1, 0, 1)
            if schedule == "exp":
                # Exponential shaping: w * (e^w - 1) / (e - 1)
                w = w * (torch.exp(w) - 1) / (np.e - 1)
        else:
            raise ValueError(f"Invalid schedule: {schedule}")
        
        # Zero out everything past 'end'
        w = torch.where(indices >= end, torch.zeros_like(w), w)
        return w
    
    def forward(self, batch):
        return self.compute_loss(batch)
    
    # def _pinv_corrected_velocity(
    #     self,
    #     z_t: torch.Tensor,
    #     t_id: torch.Tensor,
    #     global_cond: torch.Tensor,
    #     y_prev: torch.Tensor,
    #     weights: torch.Tensor,
    #     max_guidance_weight: float
    # ) -> torch.Tensor:
    #     """ΠGDM-corrected velocity with consistent time variable."""
    #     B, H, D = z_t.shape
        
    #     # Derive tau from the scheduler's timestep
    #     # For DDPM: tau = t_id / num_train_timesteps where tau=1 is noise, tau=0 is data
    #     num_train_timesteps = self.noise_scheduler.config.num_train_timesteps
    #     if isinstance(t_id, torch.Tensor):
    #         tau = t_id.float().mean().item() / num_train_timesteps
    #     else:
    #         tau = float(t_id) / num_train_timesteps
        
    #     # Now tau matches the noise level: tau=1 at pure noise, tau=0 at clean data

    #     # print(f"[DEBUG] t_id: {t_id if not isinstance(t_id, torch.Tensor) else t_id.item()}, "
    #     #   f"num_train_timesteps: {num_train_timesteps}, tau: {tau:.4f}")

    #     global_cond_detached = global_cond.detach()
    #     z_t_copy = z_t.detach().clone().requires_grad_(True)
        
    #     with torch.enable_grad():
    #         v_t = self.flow_adapter.velocity(z_t_copy, t_id, global_cond_detached)
    #         # Predict clean data: x_0 = x_tau - v * tau
    #         x_0_hat = z_t_copy - v_t * tau
        
    #     # Error against target clean data
    #     error = (y_prev - x_0_hat) * weights[None, :, None]
        
    #     if z_t_copy.grad is not None:
    #         z_t_copy.grad.zero_()
        
    #     x_0_hat.backward(error)
    #     pinv_correction = z_t_copy.grad.detach().clone()
        
    #     # Guidance weight (corrected formulas from GitHub issue)
    #     inv_r2 = (tau**2 + (1 - tau)**2) / (tau**2 + 1e-12)
    #     c = tau / ((1 - tau) + 1e-12)
    #     guidance_weight = min(c * inv_r2, max_guidance_weight)

    #     # print(f"[DEBUG] c={c:.4f}, inv_r2={inv_r2:.4f}, "
    #     #       f"weight={guidance_weight:.4f}, max={max_guidance_weight}")
        
    #     # Subtract correction (from GitHub issue)
    #     return v_t.detach() - guidance_weight * pinv_correction
