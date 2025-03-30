import torch

def content_loss(content_image , genrated_image):
    return 0.5 * torch.sum((content_image - genrated_image) ** 2)