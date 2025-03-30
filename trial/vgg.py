import torch
from torchvision.models import vgg16 , VGG16_Weights
import torch.nn as nn


# class VggFeatureExtractor(nn.Module):
#     def __init__(self, target_layer = 1 , device = "cuda"):
#         super().__init__()
#         self.target_layer = target_layer
#         self.feature_maps = None
#         self.device = device
#         self.model = vgg16(weights="VGG16_Weights.IMAGENET1K_FEATURES").features.eval().to(self.device)
#         self.preprocess = VGG16_Weights.IMAGENET1K_V1.transforms()

#         # Validate layer index
#         if target_layer >= len(self.model):
#             raise ValueError(f"VGG19 features only have {len(self.model)-1} layers")

#         # Register Hook
#         self.model[target_layer].register_forward_hook(self._save_features)


#     def _save_features(self , module , input , output):
#         self.feature_maps = output

#     def forward(self, x):
#         x = self.preprocess(x)
#         if x.dim() == 3: # ( C , H , w )
#             x = x.unsqueeze(0) # ( B , C , H , W)
#         x = x.to(self.device)
#         _ = self.model(x)
#         return self.feature_maps

class VggFeatureExtractor(nn.Module):
    """
    Extracts features from specific layers in VGG19.
    """
    def __init__(self, target_layer = 10 , device = "cuda"):
        super(VggFeatureExtractor, self).__init__()
        # Load pretrained VGG19 model
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_FEATURES).features.eval()
        # Create a sequential model that runs only up to the selected layer
        self.model = nn.Sequential(*list(vgg.children())[:target_layer+1])
        # Move model to device and set to evaluation mode
        self.model = self.model.to(device).eval()
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        return self.model(x)
    