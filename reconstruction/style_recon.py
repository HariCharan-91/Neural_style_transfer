import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import os
# Import the optimized VGG feature extractor
from models.vgg_fast import VggFeatureExtractor
from utils.image_utils import preprocess_image, save_image, compute_gram_matrix
from visualization.visualize import plot_loss, show_progress
from utils.losses import tv_loss_fn, style_loss_fn

class StyleReconstructor:
    """
    Reconstructs style from an original image using VGG features.
    Optimized to use a single VGG model for all feature extraction.
    
    Args:
        style_image_path (str): Path to the style image.
        style_layers (list): List of layer indices to extract style features from.
        device (str): Device to use ('cuda' or 'cpu').
        model_type (str): Type of VGG model ('vgg16' or 'vgg19').
        image_size (int): Size of the output image.
    """
    def __init__(self, style_image_path, style_layers=[1, 6, 11, 20, 29], 
                 device="cuda", model_type="vgg16", image_size=224):
        self.device = device
        self.style_layers = style_layers
        self.image_size = image_size
        
        # Load and preprocess the style image
        self.style_image = preprocess_image(style_image_path, size=image_size, device=device)


        
        # Initialize a single feature extractor for all style layers
        self.extractor = VggFeatureExtractor(
            target_layers=style_layers, 
            device=device, 
            model_type=model_type
        )
        
        # Compute style features from the original image
        with torch.no_grad():
            style_features_dict = self.extractor(self.style_image)
            
            # Handle both single layer and multi-layer cases
            if isinstance(style_features_dict, dict):
                self.style_features = {
                    layer: compute_gram_matrix(features) 
                    for layer, features in style_features_dict.items()
                }
            else:
                # Handle case where only one layer was specified and a tensor is returned
                self.style_features = {
                    style_layers[0]: compute_gram_matrix(style_features_dict)
                }
     
    def _initialize_image(self, init_method="noise", noise_factor=0.1):
        """Initialize the image to be optimized."""
        print(f"Initializing image with method: {init_method}")
        if init_method == "noise":
            # Pure Gaussian noise
            img = torch.randn(1, 3, self.image_size, self.image_size).to(self.device)
        elif init_method == "style_with_noise":
            # Style image with added noise
            img = self.style_image.clone()
            noise = torch.randn_like(img) * noise_factor
            img = img + noise
        else:
            # Default to white noise
            img = torch.randn(1, 3, self.image_size, self.image_size).to(self.device)
            
        # Ensure values are clamped to valid range
        img = torch.clamp(img, 0, 1)
        return img.requires_grad_(True)
    
    def reconstruct(self, optimizer_type="lbfgs", init_method="noise", 
                   num_steps=300, style_weight=1e6, tv_weight=1e2, 
                   lr=0.01, noise_factor=0.1, print_freq=10, 
                   show_images=True, output_path="output/style"):
        """
        Reconstruct style from the original image.
        
        Args:
            optimizer_type (str): Type of optimizer to use ('lbfgs' or 'adam').
            init_method (str): Method to initialize the image ('noise' or 'style_with_noise').
            num_steps (int): Number of optimization steps.
            style_weight (float): Weight for style loss.
            tv_weight (float): Weight for total variation loss.
            lr (float): Learning rate for optimizer.
            noise_factor (float): Factor for noise when using 'style_with_noise'.
            print_freq (int): Frequency to print progress to terminal.
            show_images (bool): Whether to display images during optimization.
            output_path (str): Directory to save output images.
            
        Returns:
            torch.Tensor: Reconstructed image tensor.
        """
        os.makedirs(output_path, exist_ok=True)
        
        # Initialize image
        generated = self._initialize_image(init_method, noise_factor)
        
        save_image(self.style_image , path=os.path.join(output_path, "original.jpg"))
        # Setup optimizer
        print(f"Setting up {optimizer_type} optimizer with learning rate {lr}...")
        if optimizer_type.lower() == "lbfgs":
            optimizer = optim.LBFGS([generated], lr=lr)
        elif optimizer_type.lower() == "adam":
            optimizer = optim.Adam([generated], lr=lr)
        else:
            raise ValueError("Unsupported optimizer. Choose 'lbfgs' or 'adam'.")
        
        # For storing loss history
        style_loss_history = []
        tv_loss_history = []
        total_loss_history = []
        
        # For displaying progress
        if num_steps >= 10:
            display_steps = [int(num_steps * i / 10) for i in range(11)]
        else:
            display_steps = []
        
        # Start timer
        start_time = time.time()
        last_print_time = start_time
        
        # Function to perform a single optimization step
        def closure():
            optimizer.zero_grad()
            
            # Get all feature maps in one forward pass
            current_features = self.extractor(generated)
            
            # Compute style loss for each layer
            total_style_loss = 0
            
            # Handle both single layer and multi-layer cases
            if len(self.style_layers) == 1 and not isinstance(current_features, dict):
                # Single layer case
                current_gram = compute_gram_matrix(current_features)
                layer = self.style_layers[0]
                layer_style_loss = style_loss_fn(current_gram, self.style_features[layer])
                total_style_loss += layer_style_loss
            else:
                # Multi-layer case
                for layer in self.style_layers:
                    current_gram = compute_gram_matrix(current_features[layer])
                    layer_style_loss = style_loss_fn(current_gram, self.style_features[layer])
                    total_style_loss += layer_style_loss
            
            # Compute total variation loss
            variation_loss = tv_loss_fn(generated)
            
            # Combine losses with their weights
            weighted_style_loss = style_weight * total_style_loss
            weighted_tv_loss = tv_weight * variation_loss
            total_loss = weighted_style_loss + weighted_tv_loss
            
            # Record losses
            style_loss_history.append(weighted_style_loss.item())
            tv_loss_history.append(weighted_tv_loss.item())
            total_loss_history.append(total_loss.item())
            
            # Compute gradients
            total_loss.backward()
            
            # Clamp the image values after gradient step if using Adam
            if optimizer_type.lower() == "adam":
                with torch.no_grad():
                    generated.clamp_(0, 1)
            
            return total_loss
        
        # Optimization loop
        print(f"\nStarting style reconstruction with {optimizer_type} optimizer...")
        print(f"Total steps: {num_steps}, Style weight: {style_weight}, TV weight: {tv_weight}")
        print("-" * 80)
        print(f"{'Step':>8} | {'Style Loss':>12} | {'TV Loss':>12} | {'Total Loss':>12} | {'Time (s)':>9}")
        print("-" * 80)
        
        current_step = 0
        
        if optimizer_type.lower() == "lbfgs":
            # L-BFGS optimizer
            while current_step < num_steps:
                loss = optimizer.step(closure)
                
                # Manually clamp values for L-BFGS
                with torch.no_grad():
                    generated.clamp_(0, 1)
                
                # Display progress in terminal
                if current_step % print_freq == 0 or current_step == num_steps - 1:
                    current_time = time.time()
                    elapsed = current_time - last_print_time
                    last_print_time = current_time
                    
                    style_loss = style_loss_history[-1] if style_loss_history else 0
                    tv_loss = tv_loss_history[-1] if tv_loss_history else 0
                    total_loss = total_loss_history[-1] if total_loss_history else 0
                    
                    print(f"{current_step:>8} | {style_loss:>12.4e} | {tv_loss:>12.4e} | {total_loss:>12.4e} | {elapsed:>9.2f}")
                
                # Display image if requested
                if current_step in display_steps:
                    save_image(generated, os.path.join(output_path, f"iter_{current_step}.jpg"))
                    if show_images :
                        show_progress(generated.detach().clone(), current_step, num_steps)
                        
                current_step += 1
        else:
            # Adam optimizer
            for step in range(num_steps):
                loss = closure()
                optimizer.step()
                
                # Print progress
                if step % print_freq == 0 or step == num_steps - 1:
                    current_time = time.time()
                    elapsed = current_time - last_print_time
                    last_print_time = current_time
                    
                    style_loss = style_loss_history[-1]
                    tv_loss = tv_loss_history[-1]
                    total_loss = total_loss_history[-1]
                    
                    print(f"{step:>8} | {style_loss:>12.4e} | {tv_loss:>12.4e} | {total_loss:>12.4e} | {elapsed:>9.2f}")
                
                # Display image if requested
                if show_images and step in display_steps:
                    save_image(generated, os.path.join(output_path, f"iter_{step}.jpg"))
        
        # Final timing
        total_time = time.time() - start_time
        print("-" * 80)
        print(f"Optimization completed in {total_time:.2f} seconds")
        
        # Display final result
        if show_images:
            save_image(generated, os.path.join(output_path, f"Final_style.jpg"))
        
        # Plot loss history
        plot_loss(style_loss_history=style_loss_history, tv_loss_history=tv_loss_history, total_loss_history=total_loss_history)
        
        return generated.detach()