import torch
import numpy as np

def to_gpu_tensor(data, device='cuda'):
    """
    Convert numpy array or CPU tensor to GPU tensor.
    
    Args:
        data: numpy array or torch tensor (CPU or GPU)
        device: target device ('cuda', 'cuda:0', etc.)
    
    Returns:
        torch.Tensor on specified GPU device
    """
    # Handle numpy arrays
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    # Handle torch tensors
    elif isinstance(data, torch.Tensor):
        tensor = data
    else:
        raise TypeError(f"Expected numpy array or torch tensor, got {type(data)}")
    
    # Move to GPU if not already there
    if not tensor.is_cuda:
        tensor = tensor.to(device)
    
    return tensor

def tensor_to_numpy(tensor):
    # Handle tensors that require gradients
    if tensor.requires_grad:
        tensor = tensor.detach()
    
    # Move to CPU if on CUDA
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    return tensor.numpy()