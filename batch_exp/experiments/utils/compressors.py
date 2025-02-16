"""
Author: Anonymous during review process
Institution: Anonymous during review process
Date: Anonymous during review process
Compression algorithms for model pruning.
"""

import os

from tqdm import tqdm
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim

from collections import defaultdict
from typing import Tuple
import torch

SKIP_LAYERS = ["bias", "linear", "bn"]  # Layers to skip from sparsification

DAMP = 1e-5

def mask_generation(
    mask_batch: int,
    comp_class: str,
    model: nn.Module,
    device: torch.device,
    mask_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    count_mask: Dict[str, torch.Tensor],
    sparsity: float = 0.5,
    mask_type: str = "global",
    seed: int = 0,
    dataset_class: str = "cifar10",
    model_class: str = "resnet18",
    warmup: int = 0,
    exp_class: str = "pbt"
) -> Dict[str, torch.Tensor]:

    if not 0 <= sparsity <= 1:
        raise ValueError("Sparsity must be between 0 and 1.")
    if mask_type not in {"global", "layer"}:
        raise ValueError("Invalid mask_type. Must be 'global' or 'layer'.")

    # Early exit for edge cases
    if sparsity == 1:
        return {name: torch.zeros_like(param) for name, param in model.named_parameters()}
    elif sparsity == 0:
        return {name: torch.ones_like(param) for name, param in model.named_parameters()}

    edge_case = mask_batch > 4096

    os.makedirs(f"{save_saliency_dict}/{dataset_class}/{model_class}/seed_{seed}", exist_ok=True)
    precomputed_mask_dir = f"{save_saliency_dict}/{dataset_class}/{model_class}/seed_{seed}/{comp_class}_{mask_batch}_warmup_{warmup}.pt"

    if comp_class == "random":
        compression_mask = random_compressor(model, sparsity=sparsity, mask_type=mask_type)

    elif comp_class == "magnitude":
        compression_mask = magnitude_compressor(model, sparsity=sparsity, mask_type=mask_type)

    elif comp_class == "grad_norm":
        print(f"Computing: {precomputed_mask_dir}")
        mean_mask = grad_mean(model, device, mask_loader, criterion, optimizer, edge_case=edge_case)
        compression_mask = mask2binary(mean_mask, sparsity=sparsity, mask_type=mask_type)

    elif comp_class == "fisher_diag":
        print(f"Computing: {precomputed_mask_dir}")
        fisher_mask = fisher_diag(model, device, mask_loader, criterion, optimizer, edge_case=edge_case)
        compression_mask = mask2binary(fisher_mask, sparsity=sparsity, mask_type=mask_type)

    elif comp_class == "fisher_pruner":
        print(f"Computing: {precomputed_mask_dir}")
        fisher_mask = fisher_diag(model, device, mask_loader, criterion, optimizer, edge_case=edge_case)
        fisher_saliency = saliency_score(model, fisher_mask)
        compression_mask = mask2binary(fisher_saliency, sparsity=sparsity, mask_type=mask_type)

    elif comp_class == "snip":
        print(f"Computing: {precomputed_mask_dir}")
        snip_mask = snip_sensitivity(model, device, mask_loader, criterion)
        compression_mask = mask2binary(snip_mask, sparsity=sparsity, mask_type=mask_type)

    elif comp_class == "grasp":
        print(f"Computing: {precomputed_mask_dir}")
        grasp_mask = grasp_sensitivity(model, device, mask_loader, criterion)
        compression_mask = mask2binary(grasp_mask, sparsity=sparsity, mask_type=mask_type)

    elif comp_class == "fts":
        print(f"Computing: {precomputed_mask_dir}")
        fd_taylor_mask = fd_taylor(model, device, mask_loader, criterion, optimizer, edge_case=edge_case)
        compression_mask = mask2binary(fd_taylor_mask, sparsity=sparsity, mask_type=mask_type)

    elif comp_class == "fbss":
        print(f"Computing: {precomputed_mask_dir}")
        fd_obs_taylor_mask = fd_obs_taylor(model, device, mask_loader, criterion, optimizer, edge_case=edge_case)
        compression_mask = mask2binary(fd_obs_taylor_mask, sparsity=sparsity, mask_type=mask_type)
    return compression_mask

def saliency_score(model, hessian_diagonal):
    saliency = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
    for name, param in model.named_parameters():
        if name in hessian_diagonal:
            # multiply square of the parameter with the hessian diagonal and divide by 2
            saliency[name] = torch.square(param) * hessian_diagonal[name] / 2
    return saliency

def mask2binary(
    score_mask: Dict[str, torch.Tensor], sparsity: float, mask_type: str = "global"
) -> Dict[str, torch.Tensor]:
    """
    Convert importance values to binary masks based on sparsity and type.

    Args:
        score_mask (Dict[str, torch.Tensor]): Dictionary of importance values for each parameter.
        sparsity (float): Proportion of parameters to zero out (0 to 1).
        mask_type (str): Type of sparsity - 'global' or 'layer'.

    Returns:
        Dict[str, torch.Tensor]: Dictionary of binary masks for each parameter.
    """

    # Initialize binary mask
    binary_mask = {name: torch.zeros_like(param) for name, param in score_mask.items()}

    if mask_type == "global":
        # Concatenate all data, excluding specified layers
        all_data = torch.cat(
            [param.view(-1) for name, param in score_mask.items() if not any(skip in name for skip in SKIP_LAYERS)]
        )
        k = int((1 - sparsity) * len(all_data))
        k = max(k, 1)

        _, global_idx = torch.topk(all_data, k, largest=True, sorted=False)

        # Map global indices back to individual parameter masks
        current_index = 0
        for name, param in score_mask.items():
            if any(skip in name for skip in SKIP_LAYERS):
                binary_mask[name] = torch.ones_like(param)
            else:
                param_size = param.numel()
                mask_indices = global_idx[(global_idx >= current_index) & (global_idx < current_index + param_size)] - current_index
                binary_mask[name].view(-1)[mask_indices] = 1
                current_index += param_size

    elif mask_type == "layer":
        # Layer-wise mode
        for name, param in score_mask.items():
            if any(skip in name for skip in SKIP_LAYERS):
                binary_mask[name] = torch.ones_like(param)
            else:
                data = param.view(-1)
                k = int((1 - sparsity) * len(data))
                k = max(k, 1)

                _, idx = torch.topk(data, k, largest=True, sorted=False)
                binary_mask[name].view(-1)[idx] = 1

    return binary_mask

def random_compressor(
    model: nn.Module,
    sparsity: float = 0.5,
    mask_type: str = "global"
    )-> Dict[str, torch.Tensor]:

    """
    Generate random binary masks for model parameters.

    Args:
        model (torch.nn.Module): The model whose parameters need random masking.
        sparsity (float): Proportion of parameters to zero out (0 to 1).
        mask_type (str): Type of masking - 'global' or 'layer'.

    Returns:
        Dict[str, torch.Tensor]: Random binary masks for each parameter.
    """

    random_mask = {name: torch.zeros_like(param) for name, param in model.named_parameters()}

    if mask_type == "global":
        # Concatenate all parameters, excluding specified layers
        all_params = torch.cat(
            [param.view(-1) for name, param in model.named_parameters() if not any(skip in name for skip in SKIP_LAYERS)]
        )
        k = int((1 - sparsity) * len(all_params))
        k = max(k, 1)
        
        # Create a binary mask with random selection of indices
        selected_indices = torch.randperm(len(all_params))[:k]
        global_mask = torch.zeros_like(all_params)
        global_mask[selected_indices] = 1
        
        # Map global mask to individual layers
        current_index = 0
        for name, param in model.named_parameters():
            if any(skip in name for skip in SKIP_LAYERS):
                random_mask[name] = torch.ones_like(param)
            else:
                param_size = param.numel()
                if param_size > 0:  # Handle empty parameters
                    random_mask[name].view(-1).copy_(global_mask[current_index:current_index + param_size])
                current_index += param_size

    elif mask_type == "layer":
        # Layer-wise random masking
        for name, param in model.named_parameters():
            if any(skip in name for skip in SKIP_LAYERS):
                random_mask[name] = torch.ones_like(param)
            else:
                param_size = param.numel()
                if param_size > 0:  # Handle empty parameters
                    k = int((1 - sparsity) * param_size)
                    k = max(k, 1)

                    # Randomly select indices within the layer
                    selected_indices = torch.randperm(param_size)[:k]
                    layer_mask = torch.zeros_like(param.view(-1))
                    layer_mask[selected_indices] = 1
                    random_mask[name].view(-1).copy_(layer_mask)

    return random_mask

def magnitude_compressor(
    model: nn.Module,
    sparsity: float = 0.5,
    mask_type: str = "global"
) -> Dict[str, torch.Tensor]:
    """
    Generate a binary mask based on the magnitude of the weights.

    Args:
        model (torch.nn.Module): The model whose weights are evaluated.
        sparsity (float): Proportion of weights to zero out (0 to 1).
        mask_type (str): Type of masking - 'global' or 'layer'.

    Returns:
        Dict[str, torch.Tensor]: Binary masks for each parameter based on magnitude.
    """

    magnitude_mask = {name: torch.zeros_like(param) for name, param in model.named_parameters()}

    if mask_type == "global":
        # Concatenate all weight magnitudes, excluding specified layers
        all_magnitudes = torch.cat(
            [param.abs().view(-1) for name, param in model.named_parameters() if not any(skip in name for skip in SKIP_LAYERS)]
        )
        k = int((1 - sparsity) * len(all_magnitudes))
        k = max(k, 1)

        # Find indices of top-k magnitudes
        _, global_idx = torch.topk(all_magnitudes, k, largest=True, sorted=False)

        # Map global indices back to individual parameter masks
        current_index = 0
        for name, param in model.named_parameters():
            if any(skip in name for skip in SKIP_LAYERS):
                magnitude_mask[name] = torch.ones_like(param)
            else:
                param_size = param.numel()
                mask_indices = global_idx[(global_idx >= current_index) & (global_idx < current_index + param_size)] - current_index
                magnitude_mask[name].view(-1)[mask_indices] = 1
                current_index += param_size

    elif mask_type == "layer":
        # Layer-wise magnitude-based masking
        for name, param in model.named_parameters():
            if any(skip in name for skip in SKIP_LAYERS):
                magnitude_mask[name] = torch.ones_like(param)
            else:
                data = param.abs().view(-1)
                k = int((1 - sparsity) * len(data))
                k = max(k, 1)

                # Select top-k within the layer
                _, idx = torch.topk(data, k, largest=True, sorted=False)
                magnitude_mask[name].view(-1)[idx] = 1

    return magnitude_mask

def grad_mean(
    model: nn.Module, 
    device: torch.device, 
    maskloader: torch.utils.data.DataLoader, 
    criterion: nn.Module, 
    optimizer: optim.Optimizer, 
    edge_case: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Calculate the mean gradient magnitude for model parameters.

    Args:
        model (nn.Module): The model whose gradients are calculated.
        device (torch.device): The device to perform computations on.
        maskloader (torch.utils.data.DataLoader): DataLoader for the dataset.
        criterion (nn.Module): Loss function to calculate gradients.
        optimizer (torch.optim.Optimizer): Optimizer for zeroing gradients.
        edge_case (bool): Whether to use an alternative calculation method.

    Returns:
        Dict[str, torch.Tensor]: Mean gradient magnitudes for each parameter.
    """
    grad_mean = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
    model.eval()

    for data in tqdm(maskloader, desc="Grad_Mean_Mask", leave=False):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                if edge_case:
                    grad_mean[name] += param.grad.data.detach() * inputs.size(0)
                else:
                    grad_mean[name] += param.grad.data.detach()

    for name in grad_mean:
        if edge_case:
            grad_mean[name] /= len(maskloader.dataset)
        else:
            grad_mean[name] /= len(maskloader)
        
        grad_mean[name] = grad_mean[name].abs()

    return grad_mean

def fisher_diag(
    model: nn.Module, 
    device: torch.device, 
    maskloader: torch.utils.data.DataLoader, 
    criterion: nn.Module, 
    optimizer: optim.Optimizer, 
    edge_case: bool = False
) -> Dict[str, torch.Tensor]:

    """
    Calculate Fisher Information Matrix Diagonal for model parameters.

    Args:
        model (nn.Module): The model to calculate Fisher Information for.
        device (torch.device): The device to perform computations on.
        maskloader (DataLoader): DataLoader for the dataset.
        criterion (nn.Module): Loss function to calculate gradients.
        optimizer (torch.optim.Optimizer): Optimizer for zeroing gradients.
        edge_case (bool): Whether to use an alternative calculation method.

    Returns:
        Dict[str, torch.Tensor]: Fisher Information Matrix Diagonal for each parameter.
    """

    efim_diag = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
    model.eval()

    for data in tqdm(maskloader, desc="Fisher_Mask", leave=False):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                if edge_case:
                    efim_diag[name] += param.grad.data.detach() * inputs.size(0) # Multiply by batch size to get sum of gradients batch
                else:
                    efim_diag[name] += torch.square(param.grad.data.detach())

    for name in efim_diag:
        if edge_case:
            efim_diag[name] /= len(maskloader.dataset)  # Equivalent to averaging with dataset size
            efim_diag[name] = torch.square(efim_diag[name])
        else:
            efim_diag[name] /= len(maskloader)

    return efim_diag

def snip_sensitivity(
    model: nn.Module, 
    device: torch.device, 
    maskloader: torch.utils.data.DataLoader, 
    criterion: nn.Module
) -> Dict[str, torch.Tensor]:
    """
    Calculate the SNIP sensitivity for model parameters. Based on https://github.com/mi-lad/snip/blob/master/snip.py#L18 

    Args:
        model (nn.Module): The model whose parameters' sensitivities are calculated.
        device (torch.device): The device to perform computations on.
        maskloader (torch.utils.data.DataLoader): DataLoader for the dataset.
        criterion (nn.Module): Loss function to calculate gradients.

    Returns:
        Dict[str, torch.Tensor]: Normalized sensitivity scores for each parameter.
    """
    sensitivity_scores = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
    model.eval()
    model.zero_grad()

    for data in tqdm(maskloader, desc="SNIP_Mask", leave=False):
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

    for name, param in model.named_parameters():
        if param.grad is not None:
            sensitivity_scores[name] = torch.abs(param * param.grad)

    # Normalize sensitivity scores
    normalize_factor = abs(sum([torch.sum(score).item() for score in sensitivity_scores.values()]))
    for name in sensitivity_scores:
        sensitivity_scores[name] /= normalize_factor

    return sensitivity_scores

def grasp_sensitivity(
    model: nn.Module, 
    device: torch.device, 
    maskloader: torch.utils.data.DataLoader, 
    criterion: nn.Module
) -> Dict[str, torch.Tensor]:
    """
    Calculate the GRASP sensitivity for model parameters. Based on https://github.com/alecwangcq/GraSP/blob/master/pruner/GraSP.py#L49

    Args:
        model (nn.Module): The model whose parameters' sensitivities are calculated.
        device (torch.device): The device to perform computations on.
        maskloader (torch.utils.data.DataLoader): DataLoader for the dataset.
        criterion (nn.Module): Loss function to calculate gradients.

    Returns:
        Dict[str, torch.Tensor]: Sensitivity scores for each parameter.
    """
    # Initialize sensitivity scores
    sensitivity_scores = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
    model.eval()  # Set model to evaluation mode
    model.zero_grad()  # Zero out gradients

    for data in tqdm(maskloader, desc="GRASP_Mask_First", leave=False):
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward() # Gradients are accumulated

    grad_w = {name: param.grad.clone() for name, param in model.named_parameters()}

    model.zero_grad()  # Zero out gradients for second pass

    for data in tqdm(maskloader, desc="GRASP_Mask_Second", leave=False):
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        grad_f = torch.autograd.grad(loss, model.parameters(), create_graph=True)

        # Compute Hessian-vector product
        z = 0
        for (name, grad_w_i), grad_f_i in zip(grad_w.items(), grad_f):
            z += torch.sum(grad_f_i * grad_w_i)
        z.backward()

    # Compute grasp sensitivity scores
    for name, param in model.named_parameters():
        sensitivity_scores[name] = -param * param.grad # Theory: -theta * Hg

    # Normalize sensitivity scores
    total_score_sum = sum([torch.sum(score).item() for score in sensitivity_scores.values()])
    norm_factor = abs(total_score_sum)
    for name in sensitivity_scores:
        sensitivity_scores[name] /= norm_factor

    return sensitivity_scores

def fd_taylor(
    model: nn.Module, 
    device: torch.device, 
    maskloader: torch.utils.data.DataLoader, 
    criterion: nn.Module, 
    optimizer: optim.Optimizer,
    edge_case: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Compute saliency with 1st and 2nd degree terms of Taylor expansion, utilizing the Fisher Diagonal.

    Args:
        model (nn.Module): The model to compute OBD saliency for.
        device (torch.device): The device to perform computations on.
        maskloader (DataLoader): DataLoader for the dataset.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for zeroing gradients.

    Returns:
        Dict[str, torch.Tensor]: Fisher Diagonal OBD plus 1st Taylor term saliency for each parameter.
    """
    sensitivity_scores = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
    grad_dict = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
    fisher_diag = {name: torch.zeros_like(param) for name, param in model.named_parameters()}

    model.eval()
    for data in tqdm(maskloader, desc="FD_Taylor", leave=True):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                if edge_case:
                    grad_dict[name] += param.grad.data.detach() * inputs.size(0)
                    fisher_diag[name] += param.grad.data.detach() * inputs.size(0)
                else:
                    grad_dict[name] += param.grad.data.detach()
                    fisher_diag[name] += torch.square(param.grad.data.detach())

    for name in fisher_diag:
        if edge_case:
            grad_dict[name] /= len(maskloader.dataset)
            fisher_diag[name] /= len(maskloader.dataset)
        else:
            grad_dict[name] /= len(maskloader)
            fisher_diag[name] /= len(maskloader)

    # get number of samples in first batch of the dataloader
    first_batch = next(iter(maskloader))  # Get first batch
    batch_size = first_batch[0].size(0)
    for name, param in model.named_parameters():
        sensitivity_scores[name] = (param * grad_dict[name]) + (torch.square(param) * (batch_size * fisher_diag[name]) / 2)

        sensitivity_scores[name] = sensitivity_scores[name].abs()

    return sensitivity_scores

def fd_obs_taylor(
    model: nn.Module, 
    device: torch.device, 
    maskloader: torch.utils.data.DataLoader, 
    criterion: nn.Module, 
    optimizer: optim.Optimizer,
    edge_case: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Compute saliency with 1st and 2nd degree terms of Taylor expansion, utilizing the Fisher Diagonal.

    Args:
        model (nn.Module): The model to compute OBD saliency for.
        device (torch.device): The device to perform computations on.
        maskloader (DataLoader): DataLoader for the dataset.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for zeroing gradients.

    Returns:
        Dict[str, torch.Tensor]: Fisher Diagonal OBD plus 1st Taylor term saliency for each parameter.
    """
    sensitivity_scores = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
    grad_dict = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
    fisher_diag = {name: torch.zeros_like(param) for name, param in model.named_parameters()}

    model.eval()
    for data in tqdm(maskloader, desc="FD_OBS_Taylor", leave=True):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                if edge_case:
                    grad_dict[name] += param.grad.data.detach() * inputs.size(0)
                    fisher_diag[name] += param.grad.data.detach() * inputs.size(0)
                else:
                    grad_dict[name] += param.grad.data.detach()
                    fisher_diag[name] += torch.square(param.grad.data.detach())

    for name in fisher_diag:
        if edge_case:
            grad_dict[name] /= len(maskloader.dataset)
            fisher_diag[name] /= len(maskloader.dataset)
        else:
            grad_dict[name] /= len(maskloader)
            fisher_diag[name] /= len(maskloader)

    # get number of samples in first batch of the dataloader
    first_batch = next(iter(maskloader))  # Get first batch
    batch_size = first_batch[0].size(0)

    for name, param in model.named_parameters():

        fisher_diag_inv = 1 / fisher_diag[name]

        sensitivity_scores[name] = (
            (batch_size/(2*fisher_diag_inv)) * torch.square(param - ((1/batch_size) * fisher_diag_inv * grad_dict[name]))
        )

    return sensitivity_scores