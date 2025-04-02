import torch
import torch.nn as nn
from torchvision.models import vgg16, vgg19, VGG16_Weights, VGG19_Weights

class VggFeatureExtractor(nn.Module):
    """
    Extracts features from specific layers in VGG16 or VGG19.
    
    Args:
        target_layer (int): Index of the layer from which to extract features.
        device (str): Device on which to load the model ('cuda' or 'cpu').
        model_type (str): The type of VGG model to use: "vgg16" or "vgg19".
    """
    def __init__(self, target_layer=10, device="cuda", model_type="vgg16"):
        super(VggFeatureExtractor, self).__init__()
        
        # Load pretrained model based on model_type argument
        if model_type.lower() == "vgg16":
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_FEATURES).features.eval()
        elif model_type.lower() == "vgg19":
            vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.eval()
        else:
            raise ValueError("Unsupported model type. Choose 'vgg16' or 'vgg19'.")
        
        # Create a sequential model that runs only up to the selected layer
        self.model = nn.Sequential(*list(vgg.children())[:target_layer+1])
        
        # Move model to device and set to evaluation mode
        self.model = self.model.to(device).eval()
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Store normalization parameters for ImageNet
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(device)
    
    def normalize(self, x):
        """Normalize the input tensor according to ImageNet stats"""
        return (x - self.mean) / self.std
    
    def forward(self, x):
        # Ensure input is normalized before feeding into the network
        x = self.normalize(x)
        return self.model(x)
