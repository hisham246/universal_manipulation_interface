import torch

class DiffusionAsFlow:
    """
    Wraps your diffusion UNet as a flow for OT (Optimal Transport) sampling.
    Works with DDIM/DDPM schedulers that iterate BACKWARD (high t_id → low t_id).
    """
    def __init__(self, model, noise_scheduler, pred_type: str):
        self.model = model
        self.scheduler = noise_scheduler
        assert pred_type in ("epsilon", "sample")
        self.pred_type = pred_type
        # self.max_time = 0.999

    def _alpha_sigma_eta(self, t_id, device, dtype):
        ab = self.scheduler.alphas_cumprod.to(device=device, dtype=dtype)  # [T]
        a  = ab[t_id].sqrt()
        s  = (1.0 - ab[t_id]).sqrt()
        eta = s / (a + s)                 # ∈(0,1), decreases toward data
        tau = 1.0 - eta                   # ∈(0,1), increases toward data
        return a, s, eta, tau

    def _discrete_timestep_to_continuous_time(self, t_id, device):
        # Keep signature, but now return τ = 1 - η
        if torch.is_tensor(t_id):
            t_id_int = t_id.item() if t_id.ndim == 0 else t_id[0].item()
        else:
            t_id_int = int(t_id)
        _, _, _, tau = self._alpha_sigma_eta(t_id_int, device, torch.float32)
        # tau = float(min(max(tau, 0.0), self.max_time))

        return torch.tensor(tau, dtype=torch.float32, device=device)

    # def _discrete_timestep_to_continuous_time(self, t_id, device):
    #     """
    #     Convert discrete scheduler timestep to continuous flow time ∈ [0, 1].
        
    #     DDIM: t_id ∈ [45, 42, ..., 3, 0] (backward)
    #     Flow time: time ∈ [0.1, 0.16, ..., 0.94, 1.0] (forward)
        
    #     Formula: time = 1 - (t_id / num_train_timesteps)
    #     - t_id=45 → time=0.1 (mostly noise, start)
    #     - t_id=0  → time=1.0 (clean data, end)
    #     """
    #     if torch.is_tensor(t_id):
    #         t_id_int = t_id.item() if t_id.ndim == 0 else t_id[0].item()
    #     else:
    #         t_id_int = int(t_id)
        
    #     num_train_timesteps = self.scheduler.config.num_train_timesteps
    #     time = 1.0 - (t_id_int / num_train_timesteps)

    #     time = min(max(time, 0.0), self.max_time)  # Clamp to [0, max_time]

    #     return torch.tensor(time, dtype=torch.float32, device=device)

    def _eps_x0_from_model(self, z_t, t_id, global_cond):
        """
        Get epsilon and x0 predictions from the diffusion model.
        This part stays the same - just calls your existing model.
        """
        B = z_t.shape[0]

        # Prepare timestep tensor
        if torch.is_tensor(t_id):
            if t_id.ndim == 0:
                t_ids = t_id.to(dtype=torch.long, device=z_t.device).expand(B)
            else:
                t_ids = t_id.to(dtype=torch.long, device=z_t.device)
        else:
            t_ids = torch.full((B,), int(t_id), dtype=torch.long, device=z_t.device)
        
        # Model forward pass
        model_out = self.model(z_t, t_ids, local_cond=None, global_cond=global_cond)

        # Get alpha_bar for this timestep
        alphas_cumprod = self.scheduler.alphas_cumprod.to(z_t.device).to(z_t.dtype)
        a_bar = alphas_cumprod.index_select(0, t_ids).view(B, 1, 1)
        sqrt_ab = torch.sqrt(a_bar.clamp(min=1e-8))
        sqrt_one_minus_ab = torch.sqrt((1.0 - a_bar).clamp(min=1e-8))

        # Convert model output to (epsilon, x0) based on prediction type
        if self.pred_type == 'epsilon':
            eps_hat = model_out
            x0_hat = (z_t - sqrt_one_minus_ab * eps_hat) / sqrt_ab
        else:  # 'sample'
            x0_hat = model_out
            eps_hat = (z_t - sqrt_ab * x0_hat) / sqrt_one_minus_ab

        return eps_hat, x0_hat

    def velocity(self, z_t, t_id, global_cond):
        """
        Rectified Flow velocity (no singularity).
        
        This is the correct formula for OT flows when:
        - Path: z_t = (1-t)*noise + t*x0
        - Velocity: v = d(z_t)/dt = x0 - noise
        """
        eps_hat, x0_hat = self._eps_x0_from_model(z_t, t_id, global_cond)
        
        # Rectified flow velocity (Liu et al. 2022)
        # This is what the JAX code uses!

        v = x0_hat - eps_hat
    
        return v

# import torch

# class DiffusionAsFlow:
#     """
#     Wraps your diffusion UNet as a flow: exposes velocity v(z_t, τ).
#     Works for pred_type in {'epsilon','sample'} (diffusers' 'v_prediction' not used here).
#     """
#     def __init__(self, model, noise_scheduler, pred_type: str):
#         self.model = model
#         self.scheduler = noise_scheduler
#         assert pred_type in ("epsilon", "sample")
#         self.pred_type = pred_type

#         # Precompute the sqrt(ᾱ_t) tables on the right device at runtime
#         # We'll index with the scheduler's discrete timestep ids (same ones you already pass to UNet).
#     def _eps_x0_from_model(self, z_t, t_id, global_cond):
#         """
#         z_t: [B,H,D] current noisy actions (normalized space)
#         t_id: scalar int or 1D tensor of ints from scheduler.timesteps
#         """
#         B = z_t.shape[0]

#         # batch timestep handling
#         if torch.is_tensor(t_id):
#             if t_id.ndim == 0:
#                 t_ids = t_id.to(dtype=torch.long, device=z_t.device).expand(B)
#             else:
#                 assert t_id.shape[0] == B, "t_id must be scalar or shape [B]"
#                 t_ids = t_id.to(dtype=torch.long, device=z_t.device)
#         else:
#             t_ids = torch.full((B,), int(t_id), dtype=torch.long, device=z_t.device)
        
#         # 1) UNet forward at this timestep
#         model_out = self.model(z_t, t_ids, local_cond=None, global_cond=global_cond)

#         # 2) Convert to (ε̂, x̂0) using the scheduler's ᾱ_t
#         #    We fetch ᾱ_t with the SAME discrete id 't_id' you're already using.
#         alphas_cumprod = self.scheduler.alphas_cumprod.to(z_t.device).to(z_t.dtype)
#         a_bar = alphas_cumprod.index_select(0, t_ids).view(B, 1, 1)
#         # sqrt_ab = torch.sqrt(a_bar)
#         # sqrt_one_minus_ab = torch.sqrt(1.0 - a_bar)
#         sqrt_ab = torch.sqrt(a_bar.clamp(min=1e-8))
#         sqrt_one_minus_ab = torch.sqrt((1.0 - a_bar).clamp(min=1e-8))

#         if self.pred_type == 'epsilon':         # model predicts ε̂
#             eps_hat = model_out
#             x0_hat = (z_t - sqrt_one_minus_ab * eps_hat) / (sqrt_ab)
#         else:                                   # 'sample' — model predicts x̂0
#             x0_hat = model_out
#             eps_hat = (z_t - sqrt_ab * x0_hat) / (sqrt_one_minus_ab)

#         return eps_hat, x0_hat

#     def velocity(self, z_t, t_id, global_cond):
#         eps_hat, x0_hat = self._eps_x0_from_model(z_t, t_id, global_cond)
#         return x0_hat - eps_hat  # the flow velocity field
    

        # if isinstance(t_id, torch.Tensor):
        #     t_int = int(t_id[0].item())
        # else:
        #     t_int = int(t_id)

        # alphas_cumprod = self.scheduler.alphas_cumprod.to(z_t.device).to(z_t.dtype)
        # a_bar = alphas_cumprod[t_int].view(1,1,1)
        # sqrt_ab = torch.sqrt(a_bar)
        # # sqrt_one_minus_ab = torch.sqrt(1.0 - a_bar + 1e-12)
        # sqrt_one_minus_ab = torch.sqrt(1.0 - a_bar)
