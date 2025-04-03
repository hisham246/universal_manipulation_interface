import torch
import dill 
from pprint import pprint

# Load the checkpoint (adjust path and map_location as needed)
ckpt_path = '/home/hisham246/uwaterloo/test_policy.ckpt'
payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)

# Extract the model (this depends on how it was saved)
model = payload.get('model', None)

if model is None:
    # Try loading via workspace
    cfg = payload['cfg']
    import hydra
    from diffusion_policy.workspace.base_workspace import BaseWorkspace

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    model = workspace.model


# Count parameters and check GPU memory
if model is not None:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')

    # Load model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    torch.cuda.empty_cache()  # Clear any previous allocations

    # Report GPU memory
    allocated = torch.cuda.memory_allocated(device) / 1024**2  # MB
    reserved = torch.cuda.max_memory_reserved(device) / 1024**2  # MB

    print(f"Allocated GPU memory: {allocated:.2f} MB")
    print(f"Max reserved GPU memory: {reserved:.2f} MB")

else:
    print("Model could not be loaded from the checkpoint.")