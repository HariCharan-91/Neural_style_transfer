import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import torchvision.transforms as transforms
import os
from models.vgg_fast import VggFeatureExtractor
from utils.image_utils import preprocess_image, save_image, compute_gram_matrix
from visualization.visualize import plot_loss, show_progress
from utils.losses import tv_loss_fn, style_loss_fn

class NeuralStyleTransfer:
    """
    Neural Style Transfer using VGG features.
    
    Combines content from content_image_path with style from style_image_path.
    
    Args:
        content_image_path (str): Path to the content image.
        style_image_path (str): Path to the style image.
        content_layer (int): Layer index to extract content features from.
        style_layers (list): List of layer indices to extract style features from.
        device (str): Device to use ('cuda' or 'cpu').
        model_type (str): Type of VGG model ('vgg16' or 'vgg19').
        image_size (int): Size of the output image.
    """
    def __init__(self, content_image_path, style_image_path, 
                 content_layer= 11, style_layers=[1, 6, 11, 20, 29], 
                 device="cuda", model_type="vgg16", image_size= 224):
        self.device = device
        self.image_size = image_size
        self.style_layers = style_layers
        self.content_layer = content_layer
        # Load and preprocess the content image
        print(f"Loading content image from {content_image_path}...")
        self.content_image = preprocess_image(content_image_path, size=image_size, device=device)
        
        # Load and preprocess the style image
        print(f"Loading style image from {style_image_path}...")
        self.style_image = preprocess_image(style_image_path, size=image_size, device=device)
        
        # Initialize content feature extractor
        print(f"Initializing content feature extractor (VGG {model_type}, layer {content_layer})...")
        self.feature_extractor = VggFeatureExtractor(
            target_layers=[content_layer] + style_layers, device=device, model_type=model_type
        )

        # Initialize style feature extractors
        print(f"Initializing style feature extractor (VGG {model_type}, layers {style_layers})...")
        # self.style_extractor = VggFeatureExtractor(
        #     target_layers=style_layers, device=device, model_type=model_type
        # )
        
        # Compute content features from the content image
        print("Computing content features...")
        with torch.no_grad():
            self.content_features = self.feature_extractor(self.content_image)
       
        # Compute style features from the style image
        print("Computing style features...")
        with torch.no_grad():
            self.style_features_dict = self.feature_extractor(self.style_image)

            self.style_features = {
                layer : compute_gram_matrix(features) 
                for layer , features in self.style_features_dict.items()
            }
        
        print("Initialization complete.")
    
    def _initialize_image(self, init_method="content", noise_factor=0.1):
        """Initialize the image to be optimized."""
        print(f"Initializing image with method: {init_method}")
        if init_method == "noise":
            # Pure Gaussian noise
            img = torch.randn(1, 3, self.image_size, self.image_size).to(self.device)
        elif init_method == "content":
            # Content image (recommended for neural style transfer)
            img = self.content_image.clone()
        elif init_method == "content_with_noise":
            # Content image with added noise
            img = self.content_image.clone()
            noise = torch.randn_like(img) * noise_factor
            img = img + noise
        elif init_method == "style":
            # Style image
            img = self.style_image.clone()
        elif init_method == "style_with_noise":
            # Style image with added noise
            img = self.style_image.clone()
            noise = torch.randn_like(img) * noise_factor
            img = img + noise
        else:
            # Default to content image
            img = self.content_image.clone()
            
        # Ensure values are clamped to valid range
        img = torch.clamp(img, 0, 1)
        return img.requires_grad_(True)
    
    def transfer(self, optimizer_type="lbfgs", init_method="content", 
                 num_steps=300, content_weight=1.0, style_weight=1e6, tv_weight=1e-2, 
                 lr=0.01, noise_factor=0.1, print_freq=10, 
                 show_images=True, output_path="output/style_transfer"):
        """
        Perform neural style transfer.
        
        Args:
            optimizer_type (str): Type of optimizer to use ('lbfgs' or 'adam').
            init_method (str): Method to initialize the image ('content', 'content_with_noise', 'style', 'style_with_noise', or 'noise').
            num_steps (int): Number of optimization steps.
            content_weight (float): Weight for content loss.
            style_weight (float): Weight for style loss.
            tv_weight (float): Weight for total variation loss.
            lr (float): Learning rate for optimizer.
            noise_factor (float): Factor for noise when using initialization with noise.
            print_freq (int): Frequency to print progress to terminal.
            show_images (bool): Whether to display images during optimization.
            output_path (str): Path to save output images.
            
        Returns:
            torch.Tensor: Stylized image tensor.
        """
        os.makedirs(output_path, exist_ok=True)
        
        # Save original content and style images
        save_image(self.content_image, os.path.join(output_path, "content_original.jpg"))
        save_image(self.style_image, os.path.join(output_path, "style_original.jpg"))
        
        # Initialize image
        generated = self._initialize_image(init_method, noise_factor)
        
        # Setup optimizer
        print(f"Setting up {optimizer_type} optimizer with learning rate {lr}...")
        if optimizer_type.lower() == "lbfgs":
            optimizer = optim.LBFGS([generated], lr=lr)
        elif optimizer_type.lower() == "adam":
            optimizer = optim.Adam([generated], lr=lr)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=num_steps//3, gamma=0.5)
        else:
            raise ValueError("Unsupported optimizer. Choose 'lbfgs' or 'adam'.")
        
        # For storing loss history
        content_loss_history = []
        style_loss_history = []
        tv_loss_history = []
        total_loss_history = []
        
        # For displaying progress
        if show_images and num_steps >= 10:
            display_steps = [int(num_steps * i / 10) for i in range(11)]
        else:
            display_steps = []
        
        # Start timer
        start_time = time.time()
        last_print_time = start_time
        
        # Function to perform a single optimization step
        def closure():
            optimizer.zero_grad()

            current_features = self.feature_extractor(generated)
            
            # Compute content loss
            
            content_loss = nn.functional.mse_loss(current_features[self.content_layer], self.content_features[self.content_layer])
            
            # Compute style loss for each layer
            total_style_loss = 0 

            for layer in self.style_layers:
                current_gram = compute_gram_matrix(current_features[layer])
                layer_style_loss = style_loss_fn(current_gram, self.style_features[layer])
                total_style_loss += layer_style_loss 
                
           
            # Compute total variation loss for smoothness
            variation_loss = tv_loss_fn(generated)
            
            # Combine losses with their weights
            weighted_content_loss = content_weight * content_loss
            weighted_style_loss = style_weight * total_style_loss
            weighted_tv_loss = tv_weight * variation_loss
            total_loss = weighted_content_loss + weighted_style_loss + weighted_tv_loss
            
            # Record losses
            content_loss_history.append(weighted_content_loss.item())
            style_loss_history.append(weighted_style_loss.item())
            tv_loss_history.append(weighted_tv_loss.item())
            total_loss_history.append(total_loss.item())
            
            # Compute gradients
            total_loss.backward()
            
            return total_loss
        
        # Optimization loop
        print(f"\nStarting neural style transfer with {optimizer_type} optimizer...")
        print(f"Total steps: {num_steps}, Content weight: {content_weight}, Style weight: {style_weight}, TV weight: {tv_weight}")
        print("-" * 100)
        print(f"{'Step':>8} | {'Content Loss':>12} | {'Style Loss':>12} | {'TV Loss':>12} | {'Total Loss':>12} | {'Time (s)':>9}")
        print("-" * 100)
        
        if optimizer_type.lower() == "lbfgs":
            # L-BFGS optimizer
            for step in range(num_steps):
                optimizer.step(closure)
                
                # Manually clamp values for L-BFGS
                with torch.no_grad():
                    generated.clamp_(0, 1)
                
                # Display progress in terminal
                if step % print_freq == 0 or step == num_steps - 1:
                    current_time = time.time()
                    elapsed = current_time - last_print_time
                    last_print_time = current_time
                    
                    content_loss = content_loss_history[-1] if content_loss_history else 0
                    style_loss = style_loss_history[-1] if style_loss_history else 0
                    tv_loss = tv_loss_history[-1] if tv_loss_history else 0
                    total_loss = total_loss_history[-1] if total_loss_history else 0
                    
                    print(f"{step:>8} | {content_loss:>12.4e} | {style_loss:>12.4e} | {tv_loss:>12.4e} | {total_loss:>12.4e} | {elapsed:>9.2f}")
                
                # Display image if requested
                if show_images and step in display_steps:
                    save_image(generated, os.path.join(output_path, f"iter_{step}.jpg"))
                    if show_images:
                        show_progress(generated.detach().clone(), step, num_steps)
        else:
            # Adam optimizer
            for step in range(num_steps):
                loss = closure()
                optimizer.step()
                scheduler.step()
                
                # Clamp values after step
                with torch.no_grad():
                    generated.clamp_(0, 1)
                
                # Print progress
                if step % print_freq == 0 or step == num_steps - 1:
                    current_time = time.time()
                    elapsed = current_time - last_print_time
                    last_print_time = current_time
                    
                    content_loss = content_loss_history[-1]
                    style_loss = style_loss_history[-1]
                    tv_loss = tv_loss_history[-1]
                    total_loss = total_loss_history[-1]
                    
                    print(f"{step:>8} | {content_loss:>12.4e} | {style_loss:>12.4e} | {tv_loss:>12.4e} | {total_loss:>12.4e} | {elapsed:>9.2f}")
                
                # Display image if requested
                if show_images and step in display_steps:
                    save_image(generated, os.path.join(output_path, f"iter_{step}.jpg"))
                    if show_images:
                        show_progress(generated.detach().clone(), step, num_steps)
        
        # Final timing
        total_time = time.time() - start_time
        print("-" * 100)
        print(f"Style transfer completed in {total_time:.2f} seconds")
        
        # Display final result
        save_image(generated, os.path.join(output_path, f"final_stylized.jpg"))
        if show_images:
            show_progress(generated.detach().clone(), num_steps, num_steps, final=True)
        
        # Plot loss history
        plot_loss(
            content_loss_history=content_loss_history,
            style_loss_history=style_loss_history,
            tv_loss_history=tv_loss_history,
            total_loss_history=total_loss_history,
            output_path=os.path.join(output_path, "loss_plot.png")
        )
        
        return generated.detach()