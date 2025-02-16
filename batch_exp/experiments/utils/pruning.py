"""
Author: Anonymous during review process
Institution: Anonymous during review process
Date: Anonymous during review process
Pruning function for PBT.
"""
import torch

def pbt_pruning(binary_mask, model):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in binary_mask:
                param.data.mul_(binary_mask[name])  # Element-wise multiplication

            else:
                raise KeyError(f"Binary mask for parameter '{name}' not found.")

    return model