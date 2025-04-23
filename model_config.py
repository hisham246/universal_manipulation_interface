import dill
import torch

ckpt_path = '/home/hisham246/uwaterloo/diffusion_policy_models/test_policy.ckpt'
# ckpt_path = '/home/hisham246/uwaterloo/diffusion_policy_models/pickplace_trial_2.ckpt'

payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
cfg = payload['cfg']
print(cfg)
