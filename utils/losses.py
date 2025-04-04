import torch
import torch.nn as nn
from utils.image_utils import compute_gram_matrix 

def tv_loss_fn(img, regularization='L1'):
    """
    Compute total variation loss to encourage spatial smoothness.
    
    Args:
        img (torch.Tensor): Input image tensor of shape [1, 3, H, W]
        regularization (str): Regularization method, either 'L1' or 'L2'
    
    Returns:
        torch.Tensor: Total variation loss
    """
    batch_size, channels, height, width = img.size()
    
    # Calculate differences in x direction (horizontally)
    diff_x = img[:, :, :, 1:] - img[:, :, :, :-1]
    
    # Calculate differences in y direction (vertically)
    diff_y = img[:, :, 1:, :] - img[:, :, :-1, :]
    
    if regularization == "L1":
        # L1 Regularization (Sum of absolute differences)
        tv_loss = torch.sum(torch.abs(diff_x)) + torch.sum(torch.abs(diff_y))
    elif regularization == "L2":
        # L2 Regularization 
        tv_loss = torch.sum(diff_x.pow(2)) + torch.sum(diff_y.pow(2))
    else:
        raise ValueError(f"Unsupported regularization method: {regularization}")
    
    # Normalize by the number of elements
    tv_loss = tv_loss / (batch_size * channels * height * width)
    
    return tv_loss

def style_loss_fn(current_features, target_gram):
    """Compute style loss between current features and target gram matrix."""
    gram = compute_gram_matrix(current_features)
    return nn.functional.mse_loss(gram, target_gram)