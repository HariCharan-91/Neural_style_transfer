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
                    learning_rate=0.05, use_content_init=True, 
                    optimizer_type="adam", lbfgs_max_iter=20):
        """
        Reconstruct content from VGG features
        
        Args:
            iterations: Number of optimization iterations
            output_path: Directory to save outputs
            learning_rate: Learning rate for optimization
            use_content_init: Whether to initialize with low-noise version of content
            optimizer_type: Type of optimizer to use ('adam' or 'lbfgs')
            lbfgs_max_iter: Max iterations per step for LBFGS optimizer
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
        
        # Setup optimizer based on type
        optimizer_type = optimizer_type.lower()
        if optimizer_type == "adam":
            optimizer = optim.Adam([generated], lr=learning_rate)
        elif optimizer_type == "lbfgs":
            optimizer = optim.LBFGS([generated], lr=learning_rate, max_iter=lbfgs_max_iter)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}. Use 'adam' or 'lbfgs'")
        
        # Track loss history
        loss_history = []

        # Save original content image
        if use_content_init:
            self._save_image(self.img, os.path.join(output_path, "original.jpg"))

        # Training loop
        for i in range(iterations):
            # Different optimization step depending on optimizer type
            if optimizer_type == "adam":
                # Forward pass
                gen_features = self.extractor(generated)
                
                # Calculate MSE loss
                loss = nn.functional.mse_loss(gen_features, content_features)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Record loss
                current_loss = loss.item()
                
            else:  # LBFGS
                # Define closure for LBFGS
                def closure():
                    optimizer.zero_grad()
                    gen_features = self.extractor(generated)
                    loss = nn.functional.mse_loss(gen_features, content_features)
                    loss.backward()
                    return loss
                
                # Run optimization step
                loss = optimizer.step(closure)
                current_loss = loss.item()
            
            # Add loss to history
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