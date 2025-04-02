import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from utils.image_utils import preprocess_image
from models.vgg import VggFeatureExtractor

class NeuralStyleTransfer:
    def __init__(self, content_image_path, style_image_path, 
                 content_layer=10, style_layers=None, device="cuda"):
        self.device = device
        
        # Initialize content and style images
        self.content_img = preprocess_image(content_image_path, device=device)
        self.style_img = preprocess_image(style_image_path, device=device)
        
        # Set up layers
        self.content_layer = content_layer
        self.style_layers = style_layers if style_layers else [1, 6, 11, 20]
        
        # Initialize feature extractors
        self.content_extractor = VggFeatureExtractor(target_layer=content_layer, device=device)
        self.style_extractors = {
            layer: VggFeatureExtractor(target_layer=layer, device=device)
            for layer in self.style_layers
        }
        
        # Get target features
        with torch.no_grad():
            self.content_target = self.content_extractor(self.content_img)
            self.style_targets = self._get_style_targets(self.style_img)
        
        # Style layer weights
        self.style_weights = {layer: 1.0/len(self.style_layers) for layer in self.style_layers}
        
    def _gram_matrix(self, x):
        batch_size, channels, height, width = x.size()
        features = x.view(batch_size, channels, height * width)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram.div(channels * height * width)
    
    def _get_style_targets(self, img):
        style_features = {}
        for layer in self.style_layers:
            features = self.style_extractors[layer](img)
            style_features[layer] = self._gram_matrix(features)
        return style_features
    
    def _total_variation_loss(self, img):
        tv_h = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
        tv_w = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
        return tv_h + tv_w
    
    def transfer_style(self, iterations=500, output_path="style_transfer",
                       content_weight=1, style_weight=1e6, tv_weight=1e-3,
                       learning_rate=0.1, optimizer_type="adam", 
                       init_image="content", noise_scale=0.1):
        
        os.makedirs(output_path, exist_ok=True)
        
        # Initialize generated image
        if init_image == "content":
            generated = self.content_img.clone() + noise_scale * torch.randn_like(self.content_img)
        elif init_image == "style":
            generated = self.style_img.clone() + noise_scale * torch.randn_like(self.style_img)
        else:  # random
            generated = torch.rand_like(self.content_img, device=self.device)
            
        generated.requires_grad_(True)
        
        # Set up optimizer
        if optimizer_type.lower() == "adam":
            optimizer = optim.Adam([generated], lr=learning_rate)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        elif optimizer_type.lower() == "lbfgs":
            optimizer = optim.LBFGS([generated], lr=learning_rate, max_iter=20)
        else:
            raise ValueError("Invalid optimizer type")
        
        loss_history = []
        best_loss = float('inf')
        best_image = None
        
        # Save original images
        self._save_image(self.content_img, os.path.join(output_path, "original_content.jpg"))
        self._save_image(self.style_img, os.path.join(output_path, "original_style.jpg"))
        
        for i in range(iterations):
            def closure():
                optimizer.zero_grad()
                
                # Content loss
                gen_content_feat = self.content_extractor(generated)
                content_loss = content_weight * nn.functional.mse_loss(gen_content_feat, self.content_target)
                
                # Style loss
                style_loss = 0
                gen_style_feats = self._get_style_targets(generated)
                for layer in self.style_layers:
                    style_loss += self.style_weights[layer] * nn.functional.mse_loss(
                        gen_style_feats[layer], self.style_targets[layer]
                    )
                style_loss *= style_weight
                
                # Total variation loss
                tv_loss = tv_weight * self._total_variation_loss(generated)
                
                total_loss = content_loss + style_loss + tv_loss
                total_loss.backward()
                
                return total_loss
            
            loss = optimizer.step(closure)
            current_loss = loss.item() if isinstance(loss, torch.Tensor) else loss
            
            # Update learning rate for Adam
            if optimizer_type == "adam":
                scheduler.step()
            
            # Clamp pixel values
            with torch.no_grad():
                generated.data.clamp_(0, 1)
            
            # Track best result
            if current_loss < best_loss:
                best_loss = current_loss
                best_image = generated.clone()
            
            loss_history.append(current_loss)
            
            # Save progress
            if (i+1) % 50 == 0 or i == 0:
                print(f"Iteration {i+1}/{iterations} - Loss: {current_loss:.2f}")
                self._save_image(generated, os.path.join(output_path, f"progress_{i+1}.jpg"))
        
        # Save final results
        self._save_image(best_image, os.path.join(output_path, "final_result.jpg"))
        self._save_loss(loss_history, output_path)
        
        return best_image, loss_history
    
    def _save_image(self, tensor, path):
        img = tensor.squeeze(0).cpu().detach().clamp(0, 1)
        transforms.ToPILImage()(img).save(path)
    
    def _save_loss(self, loss_history, output_path):
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history)
        plt.title("Style Transfer Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.savefig(os.path.join(output_path, "loss_plot.png"))
        plt.close()