import torch
from typing import Mapping
import numpy as np
from torch.utils.data.dataloader import default_collate


def qformer_collate_fn(batch: list):
    """Discard None images in a batch when using torch DataLoader
    
    Args:
        batch (list): list of samples
    """
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)
