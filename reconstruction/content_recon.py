import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from utils.image_utils import preprocess_image
from models.vgg import VggFeatureExtractor

# Define total variation loss to encourage spatial smoothness in the generated image
def total_variation_loss(img):
    tv_h = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
    return tv_h + tv_w

class ContentReconstructor:
    def __init__(self, image_path=None, target_layer=10, device="cuda" , model_type = "vgg16"):
        self.device = device
        if model_type == 'vgg19':
            self.size = 512
        else:
            self.size = 224
        if image_path:
            # Resize (224, 224) , PIL ---> Tensor
            self.img = preprocess_image(image_path, device=device , size=self.size)
        else:
            self.img = None
        # Create a model that extracts features up to the target layer.
        self.extractor = VggFeatureExtractor(target_layer=target_layer, device=device,model_type= model_type)
    
    def set_image(self, img_tensor):
        """Set preprocessed image tensor"""
        self.img = img_tensor
    
    def reconstruct(self, iterations=300, output_path="output", 
                    learning_rate=0.05, use_content_init=True, 
                    optimizer_type="adam", lbfgs_max_iter=20,
                    tv_weight=0.001, noise_scale=0.1):
        """
        Reconstruct content from VGG features with additional optimizations.
        
        Args:
            iterations: Number of optimization iterations
            output_path: Directory to save outputs
            learning_rate: Learning rate for optimization
            use_content_init: Whether to initialize with content+noise or pure noise
            optimizer_type: Type of optimizer to use ('adam' or 'lbfgs')
            lbfgs_max_iter: Max iterations per step for LBFGS optimizer
            tv_weight: Weight for the total variation loss term
            noise_scale: Scale for the noise added in content initialization
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
            generated = self.img.clone() + noise_scale * torch.randn_like(self.img, device=self.device)
            generated.requires_grad_(True)
        else:
            # Initialize with pure noise
            generated = torch.rand_like(self.img, requires_grad=True, device=self.device)
        
        # Setup optimizer based on type
        optimizer_type = optimizer_type.lower()
        if optimizer_type == "adam":
            optimizer = optim.Adam([generated], lr=learning_rate)
            # Set up a scheduler to reduce the learning rate over time
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        elif optimizer_type == "lbfgs":
            optimizer = optim.LBFGS([generated], lr=learning_rate, max_iter=lbfgs_max_iter)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}. Use 'adam' or 'lbfgs'")
        
        # Track loss history
        loss_history = []

        # save original image 
        self._save_image(self.img, os.path.join(output_path, "original.jpg"))

        # Training loop
        for i in range(iterations):
            if optimizer_type == "adam":
                # Forward pass: extract features from the generated image
                gen_features = self.extractor(generated)
                # Compute content loss and add TV loss for smoothness
                loss = nn.functional.mse_loss(gen_features, content_features) + tv_weight * total_variation_loss(generated)
                
                # Backward pass and update parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()  # update learning rate
                
                current_loss = loss.item()
            else:  # LBFGS branch
                # Define closure for LBFGS optimizer
                def closure():
                    optimizer.zero_grad()
                    # Clamp values inside the closure to ensure valid pixel range
                    with torch.no_grad():
                        generated.clamp_(0, 1)
                    gen_features = self.extractor(generated)
                    loss = nn.functional.mse_loss(gen_features, content_features) + tv_weight * total_variation_loss(generated)
                    loss.backward()
                    return loss
                
                loss = optimizer.step(closure)
                current_loss = loss.item()
            
            # Record loss
            loss_history.append(current_loss)
            
            # Print progress every 10 iterations
            if (i+1) % 10 == 0:
                print(f"Iteration {i+1}/{iterations} - Loss: {current_loss:.4f}")
            
            # Clamp pixel values to valid range [0, 1]
            with torch.no_grad():
                generated.data.clamp_(0, 1)
            
            # Optionally, save intermediate results (currently commented out)
            if (i+1) % 50 == 0 or i == 0:
                self._save_image(generated, os.path.join(output_path, f"iter_{i+1}.jpg"))

        # Save final reconstruction and loss plot
        self._save_image(generated, os.path.join(output_path, "reconstruction.jpg"))
        self._save_loss(loss_history, output_path)
        
        return generated, loss_history

    def _save_image(self, tensor, path):
        """Convert tensor to PIL image and save"""
        img = tensor.squeeze(0).cpu().detach().clamp(0, 1)
        transforms.ToPILImage()(img).save(path)
    
    def _save_loss(self, loss_history, output_path):
        """Plot and save loss values"""
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history)
        plt.title("Content Reconstruction Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.savefig(os.path.join(output_path, "loss_plot.png"))
        plt.close()