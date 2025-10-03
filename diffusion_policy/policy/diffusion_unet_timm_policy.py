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
from diffusion_policy.common.flow_adapter import DiffusionAsFlow

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
    @torch.no_grad()
    def sample_chunk_flow(self, global_cond, generator=None, n_steps=None):
        """
        Deterministic flow sampler.
        Iterates reverse diffusion indices (high->low) and uses z <- z - (1/n) * v,
        with v = (ε - x0), which is equivalent to stepping + (x0 - ε).
        Returns a normalized action chunk [B, H, D].
        """
        if n_steps is None: n_steps = self.flow_n_steps
        B = global_cond.shape[0]
        H, D = self.action_horizon, self.action_dim

        # Start from standard Normal in normalized action space
        z = torch.randn((B, H, D), dtype=self.dtype, device=global_cond.device, generator=generator)

        # Use the SAME discrete scheduler timesteps you already use for DDIM/DDPM
        self.noise_scheduler.set_timesteps(n_steps)
        for k, t_id in enumerate(self.noise_scheduler.timesteps):
            # one Euler step on the flow ODE
            v = self.flow_adapter.velocity(z, t_id, global_cond)   # [B,H,D]
            z = z - (1.0 / n_steps) * v

        return z  # still normalized; unnormalize in predict_action

    def predict_action_flow(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert 'past_action' not in obs_dict
        nobs = self.normalizer.normalize(obs_dict)
        global_cond = self.obs_encoder(nobs)  # [B, feat]

        # Flow sampling in normalized space, then unnormalize
        nsample = self.sample_chunk_flow(global_cond=global_cond, n_steps=self.flow_n_steps)
        action_pred = self.normalizer['action'].unnormalize(nsample)

        return {'action': action_pred, 'action_pred': action_pred}
    
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
        """Real-time chunking with inpainting."""
        if n_steps is None:
            n_steps = self.flow_n_steps
        
        nobs = self.normalizer.normalize(obs_dict)
        global_cond = self.obs_encoder(nobs)
        B = global_cond.shape[0]
        H, D = self.action_horizon, self.action_dim
        
        # Normalize previous chunk
        y_prev = self.normalizer['action'].normalize(prev_action_chunk)
        
        # Compute soft masking weights
        weights = self._get_prefix_weights(
            inference_delay, 
            prefix_attention_horizon, 
            H, 
            prefix_attention_schedule
        )
        
        # Start from noise
        z = torch.randn((B, H, D), dtype=self.dtype, device=self.device)
        
        # Flow integration with guidance
        self.noise_scheduler.set_timesteps(n_steps)
        dt = 1.0 / n_steps
        
        for k, t_id in enumerate(self.noise_scheduler.timesteps):
            tau = k / n_steps
            
            # Compute guided velocity
            v_guided = self._pinv_corrected_velocity(
                z, t_id, global_cond, y_prev, weights, tau, max_guidance_weight
            )
            
            # Euler step and detach to break graph
            z = (z - dt * v_guided).detach()
        
        # Unnormalize
        action_pred = self.normalizer['action'].unnormalize(z)
        return {'action': action_pred, 'action_pred': action_pred}

    def _pinv_corrected_velocity(
        self,
        z_t: torch.Tensor,
        t_id: torch.Tensor,
        global_cond: torch.Tensor,
        y_prev: torch.Tensor,
        weights: torch.Tensor,
        tau: float,
        max_guidance_weight: float
    ) -> torch.Tensor:
        """ΠGDM-corrected velocity (Eq. 2 from paper)."""
        B, H, D = z_t.shape
        
        # Detach global_cond to prevent graph accumulation
        global_cond_detached = global_cond.detach()
        
        # Create fresh tensor with gradients enabled
        z_t_copy = z_t.detach().clone().requires_grad_(True)
        
        # Compute velocity and denoiser with isolated graph
        with torch.enable_grad():
            v_t = self.flow_adapter.velocity(z_t_copy, t_id, global_cond_detached)
            x_1_hat = z_t_copy + (1 - tau) * v_t
        
        # Weighted error
        error = (y_prev - x_1_hat) * weights[None, :, None]  # [B, H, D]
        
        # Compute gradient
        if z_t_copy.grad is not None:
            z_t_copy.grad.zero_()
        
        x_1_hat.backward(error)
        pinv_correction = z_t_copy.grad.detach().clone()
        
        # Guidance weight (Eq. 4)
        inv_r2 = (tau**2 + (1 - tau)**2) / ((1 - tau)**2 + 1e-12)
        c = (1 - tau) / (tau + 1e-12)
        guidance_weight = min(c * inv_r2, max_guidance_weight)
        
        # Return corrected velocity (everything detached)
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