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

        return torch.tensor(tau, dtype=torch.float32, device=device)

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
        
        v = x0_hat - eps_hat
    
        return v
    