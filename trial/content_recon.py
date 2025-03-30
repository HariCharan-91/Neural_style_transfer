import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from utils.image_utils import preprocess_image
from models.vgg import VggFeatureExtractor

class ContentReconstructor:
    def __init__(self, image_path=None, target_layer=10, device="cuda"):
        self.device = device
        # Process image if path provided
        if image_path:
            self.img = preprocess_image(image_path, device=device)
        else:
            self.img = None
        self.extractor = VggFeatureExtractor(target_layer=target_layer, device=device)
    
    def set_image(self, img_tensor):
        """Set preprocessed image tensor"""
        self.img = img_tensor
    
    def reconstruct(self, iterations=300, output_path="output", 
                    learning_rate=0.05, use_content_init=True):
        """
        Reconstruct content from VGG features
        
        Args:
            iterations: Number of optimization iterations
            output_path: Directory to save outputs
            learning_rate: Learning rate for optimization
            use_content_init: Whether to initialize with low-noise version of content
        """
        # Check that image is set
        if self.img is None:
            raise ValueError("No image set for reconstruction")
            
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Get target features
        with torch.no_grad():
            content_features = self.extractor(self.img)
        
        # Initialize image (noise + content or pure noise)
        if use_content_init:
            # Initialize with content + small noise (better convergence)
            generated = self.img.clone() + 0.1 * torch.randn_like(self.img, device=self.device)
            generated.requires_grad_(True)
        else:
            # Initialize with pure noise
            generated = torch.rand_like(self.img, requires_grad=True, device=self.device)
        
        # Setup optimizer (Adam tends to work better than LBFGS for this)
        optimizer = optim.Adam([generated], lr=learning_rate)
        
        # Track loss history
        loss_history = []

        # Save original content image
        if use_content_init:
            self._save_image(self.img, os.path.join(output_path, "original.jpg"))

        # Training loop
        for i in range(iterations):
            # Forward pass to get features
            gen_features = self.extractor(generated)
            
            # Calculate MSE loss
            loss = nn.functional.mse_loss(gen_features, content_features)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Record loss
            current_loss = loss.item()
            loss_history.append(current_loss)
            
            # Print progress
            if (i+1) % 10 == 0:
                print(f"Iteration {i+1}/{iterations} - Loss: {current_loss:.4f}")
            
            # Clamp pixel values to valid range [0,1]
            with torch.no_grad():
                generated.data.clamp_(0, 1)
            
            # Save intermediate results
            if (i+1) % 50 == 0 or i == 0:
                self._save_image(generated, os.path.join(output_path, f"iter_{i+1}.jpg"))

        # Save final results
        self._save_image(generated, os.path.join(output_path, "reconstruction.jpg"))
        self._save_loss(loss_history, output_path)
        
        return generated, loss_history

    def _save_image(self, tensor, path):
        """Convert tensor to PIL image and save"""
        img = tensor.squeeze(0).cpu().detach().clamp(0, 1)
        transforms.ToPILImage()(img).save(path)
    
    def _save_loss(self, loss_history, output_path):
        """Save loss values and plot"""
        # Save as text file
        np.savetxt(os.path.join(output_path, "loss_values.txt"), loss_history)
        
        # Create loss plot
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history)
        plt.title("Content Reconstruction Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.savefig(os.path.join(output_path, "loss_plot.png"))
        plt.close()