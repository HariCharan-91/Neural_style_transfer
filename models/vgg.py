import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights

class VggFeatureExtractor(nn.Module):
    """
    Extracts features from specific layers in VGG16.
    """
    def __init__(self, target_layer=10, device="cuda"):
        super(VggFeatureExtractor, self).__init__()
        # Load pretrained VGG16 model
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_FEATURES).features.eval()
        # Create a sequential model that runs only up to the selected layer
        self.model = nn.Sequential(*list(vgg.children())[:target_layer+1])
        # Move model to device and set to evaluation mode
        self.model = self.model.to(device).eval()
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Store normalization parameters
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(device)
        # self.mean = torch.tensor([0.48235, 0.45882, 0.40784]).view(-1, 1, 1).to(device)
        # self.std = torch.tensor([0.00392156862745098, 0.00392156862745098, 0.00392156862745098]).view(-1, 1, 1).to(device)
    
    def normalize(self, x):
        """Normalize the input tensor according to ImageNet stats"""
        return (x - self.mean) / self.std
    
    def forward(self, x):
        # Ensure input is normalized
        x = self.normalize(x)
        return self.model(x)