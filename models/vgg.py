import torch
from torchvision.models import vgg19 , VGG19_Weights
import torch.nn as nn


class VggFeatureExtractor(nn.Module):
    def __init__(self, target_layer = 19 , device = "cpu"):
        super().__init__()
        self.target_layer = target_layer
        self.feature_maps = None
        self.device = device
        self.model = vgg19(weights="VGG19_Weights.IMAGENET1K_V1").features.eval().to(self.device)
        self.preprocess = VGG19_Weights.IMAGENET1K_V1.transforms()

        # Validate layer index
        if target_layer >= len(self.model):
            raise ValueError(f"VGG19 features only have {len(self.model)-1} layers")

        # Register Hook
        self.model[target_layer].register_forward_hook(self._save_features)


    def _save_features(self , module , input , output):
        self.feature_maps = output

    def forward(self, x):
        x = self.preprocess(x)
        if x.dim() == 3: # ( C , H , w )
            x = x.unsqueeze(0) # ( B , C , H , W)
        x = x.to(self.device)
        _ = self.model(x)
        return self.feature_maps
    