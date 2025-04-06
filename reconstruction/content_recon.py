import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from utils.image_utils import preprocess_image , save_image
from visualization.visualize import plot_loss
import utils.losses as loss_fn
from models.vgg_fast import VggFeatureExtractor


class ContentReconstructor:
    def __init__(self, image_path=None, target_layer=10, device="cuda", model_type = "vgg16"):
        self.device = device
        
        self.size = 224 if model_type == 'vgg16' else 512

        if image_path:
            # Resize (224, 224) , PIL ---> Tensor
            self.img = preprocess_image(image_path, device=device, size=self.size)
        else:
            self.img = None
        # Create a model that extracts features up to the target layer.
        self.extractor = VggFeatureExtractor(target_layers=target_layer, device=device, model_type=model_type)
    
    # def set_image(self, img_tensor):
    #     """Set preprocessed image tensor"""
    #     self.img = img_tensor
    
    def reconstruct(self, iterations=300, output_path="output/content", 
                    learning_rate=0.05, use_content_init=True, 
                    optimizer_type="adam", lbfgs_max_iter=20,
                    content_weight=1.0, tv_weight=0.001, noise_scale=0.1):
        """
        Reconstruct content from VGG features with additional optimizations.
        
        Args:
            iterations: Number of optimization iterations
            output_path: Directory to save outputs
            learning_rate: Learning rate for optimization
            use_content_init: Whether to initialize with content+noise or pure noise
            optimizer_type: Type of optimizer to use ('adam' or 'lbfgs')
            lbfgs_max_iter: Max iterations per step for LBFGS optimizer
            content_weight: Weight for the content loss term
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
        
        # Track loss history and individual loss components
        loss_history = []
        content_loss_history = []
        tv_loss_history = []

        if iterations >= 10:
            display_steps = [int(iterations * i / 10) for i in range(11)]
     
        # Save original image 
        save_image(self.img, os.path.join(output_path, "original.jpg"))
        
        # Print header for reconstruction process
        print(f"\nStarting content reconstruction with {optimizer_type} optimizer...")
        print(f"Total steps: {iterations}, Content weight: {content_weight}, TV weight: {tv_weight}")
        print("-" * 85)
        print(f"{'Step':>8} | {'Content Loss':>12} | {'TV Loss':>12} | {'Total Loss':>12} | {'Time (s)':>9}")
        print("-" * 85)
        
        # Training loop
        start_time = time.time()
        for i in range(iterations):
            step_start_time = time.time()
            
            if optimizer_type == "adam":
                # Forward pass: extract features from the generated image
                gen_features = self.extractor(generated)
                
                # Compute content loss
                content_loss = nn.functional.mse_loss(gen_features, content_features)
                
                # Compute TV loss for smoothness
                tv_loss_val = loss_fn.tv_loss_fn(generated)
                
                # Total loss with weights
                loss = content_weight * content_loss + tv_weight * tv_loss_val
                
                # Backward pass and update parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()  # update learning rate
                
                # Get individual loss components
                current_content_loss = content_loss.item()
                current_tv_loss = tv_loss_val.item()
                current_total_loss = loss.item()
                
            else:  # LBFGS branch
                # Define closure for LBFGS optimizer
                def closure():
                    optimizer.zero_grad()
                    # Clamp values inside the closure to ensure valid pixel range
                    with torch.no_grad():
                        generated.clamp_(0, 1)
                    gen_features = self.extractor(generated)
                    content_loss = nn.functional.mse_loss(gen_features, content_features)
                    tv_loss_val = loss_fn.tv_loss_fn(generated)
                    loss = content_weight * content_loss + tv_weight * tv_loss_val
                    
                    # Store these for printing outside closure
                    nonlocal current_content_loss, current_tv_loss, current_total_loss
                    current_content_loss = content_loss.item()
                    current_tv_loss = tv_loss_val.item()
                    current_total_loss = loss.item()
                    
                    loss.backward()
                    return loss
                
                # Initialize variables that will be set inside closure
                current_content_loss = 0.0
                current_tv_loss = 0.0
                current_total_loss = 0.0
                
                # Run optimizer step with closure
                optimizer.step(closure)
            
            # Calculate step time
            step_time = time.time() - step_start_time
            
            # Record losses
            loss_history.append(current_total_loss)
            content_loss_history.append(current_content_loss)
            tv_loss_history.append(current_tv_loss)
            
            # Print detailed progress information
            if (i+1) % 10 == 0 or i == 0:
                print(f"{i+1:8d} | {current_content_loss:12.4f} | {current_tv_loss:12.4f} | {current_total_loss:12.4f} | {step_time * 10 :9.2f}")
            
            # Clamp pixel values to valid range [0, 1]
            with torch.no_grad():
                generated.data.clamp_(0, 1)
            
            # Save intermediate results
            if i in display_steps:
                save_image(generated, os.path.join(output_path, f"iter_{i+1}.jpg"))

        # Print final summary
        total_time = time.time() - start_time
        print("-" * 85)
        print(f"Content reconstruction completed in {total_time:.2f} seconds")
        
        # Save final reconstruction and loss plot
        save_image(generated, os.path.join(output_path, "reconstruction.jpg"))
        plot_loss(total_loss_history = tv_loss_history ,  content_loss_history=content_loss_history , tv_loss_history= loss_history,output_path="loss_graph")
        
        return generated, loss_history
