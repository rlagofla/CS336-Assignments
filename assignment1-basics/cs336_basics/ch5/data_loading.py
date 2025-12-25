import numpy as np
import numpy.typing as npt
import torch

def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
):
    idx = torch.randint(len(dataset) - context_length, (batch_size,))
    x = torch.stack([torch.from_numpy(dataset[i:i+context_length].astype(np.int64)) for i in idx])
    y = torch.stack([torch.from_numpy(dataset[i+1:i+1+context_length].astype(np.int64)) for i in idx])
    return x.to(device), y.to(device)