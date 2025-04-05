import torch
import torch.nn as nn
from torchvision.models import vgg16, vgg19, VGG16_Weights, VGG19_Weights
from collections import OrderedDict

class VggFeatureExtractor(nn.Module):
    """
    Optimized version of VGG feature extractor that can extract features from multiple layers
    in a single forward pass.
    
    Args:
        target_layers (list or int): Layer indices from which to extract features.
                                    Can be a single int or a list of ints.
        device (str): Device on which to load the model ('cuda' or 'cpu').
        model_type (str): The type of VGG model to use: "vgg16" or "vgg19".
    """
    def __init__(self, target_layers=[1, 6, 11, 20, 29], device="cuda", model_type="vgg16"):
        super(VggFeatureExtractor, self).__init__()
        
        self.device = device
        
        # Convert single layer to list for uniform processing
        if isinstance(target_layers, int):
            target_layers = [target_layers]
        self.target_layers = sorted(target_layers)
        
        # Load pretrained model based on model_type argument
        if model_type.lower() == "vgg16":
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_FEATURES).features.eval()
        elif model_type.lower() == "vgg19":
            vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.eval()
        else:
            raise ValueError("Unsupported model type. Choose 'vgg16' or 'vgg19'.")
        
        # Create a sequential model that includes all required layers
        max_layer = max(target_layers)
        
        # Create an ordered dictionary to store the layers we need
        self.features = nn.Sequential()
        
        # Add all layers up to the maximum target layer
        for i in range(max_layer + 1):
            self.features.add_module(f"layer_{i}", vgg[i])
        
        # Move model to device and set to evaluation mode
        self.features = self.features.to(device).eval()
        
        # Freeze all parameters to save memory and computation
        for param in self.features.parameters():
            param.requires_grad = False
            
        # Store normalization parameters for ImageNet - explicitly move to device
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(device))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(device))
        
        # Move the entire model to the specified device
        self.to(device)
    
    def normalize(self, x):
        """Normalize the input tensor according to ImageNet stats"""
        # Ensure x is on the same device as mean and std
        if x.device != self.mean.device:
            x = x.to(self.mean.device)
        return (x - self.mean) / self.std
    
    def forward(self, x):
        """
        Forward pass that extracts features from all target layers in a single pass.
        
        Args:
            x (torch.Tensor): Input image tensor
            
        Returns:
            dict or torch.Tensor: If multiple target layers, returns a dictionary mapping
                                 layer indices to their feature maps.
                                 If single target layer, returns just that feature map.
        """
        # Ensure input is on the right device before normalizing
        if x.device != self.device:
            x = x.to(self.device)
            
        # Normalize input
        x = self.normalize(x)
        
        features = {}
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.target_layers:
                features[i] = x
        
        # If only one target layer was requested, return just that tensor
        # instead of a dictionary to maintain backward compatibility
        if len(self.target_layers) == 1:
            return features[self.target_layers[0]]
        
        return features