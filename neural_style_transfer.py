import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import os
import gc
from models.vgg_fast import VggFeatureExtractor
from utils.image_utils import preprocess_image, save_image, compute_gram_matrix
from visualization.visualize import plot_loss, show_progress
from utils.losses import tv_loss_fn, style_loss_fn

class NeuralStyleTransfer:
    """Neural Style Transfer using VGG features with memory optimization."""
    
    def __init__(self, content_image_path, style_image_path, 
                 content_layer=11, style_layers=[1, 6, 11, 20, 29], 
                 device="cuda", model_type="vgg16", image_size=224):
        self.device = device
        self.image_size = image_size
        self.style_layers = style_layers
        self.content_layer = content_layer
        
        # Load and preprocess images
        self.content_image = preprocess_image(content_image_path, size=image_size, device=device)
        self.style_image = preprocess_image(style_image_path, size=image_size, device=device)
        
        # Initialize feature extractor
        self.feature_extractor = VggFeatureExtractor(
            target_layers=[content_layer] + style_layers, device=device, model_type=model_type
        )
        
        # Compute content features
        with torch.no_grad():
            self.content_features = {}
            all_features = self.feature_extractor(self.content_image)
            self.content_features[self.content_layer] = all_features[self.content_layer].detach().clone()
            del all_features
            torch.cuda.empty_cache() if device == "cuda" else gc.collect()
       
        # Compute style features
        with torch.no_grad():
            style_features_dict = self.feature_extractor(self.style_image)
            self.style_features = {}
            for layer in self.style_layers:
                if layer in style_features_dict:
                    self.style_features[layer] = compute_gram_matrix(style_features_dict[layer])
            del style_features_dict
            torch.cuda.empty_cache() if device == "cuda" else gc.collect()
    
    def _initialize_image(self, init_method="content", noise_factor=0.1):
        """Initialize the image to be optimized."""
        if init_method == "noise":
            img = torch.randn(1, 3, self.image_size, self.image_size).to(self.device)
        elif init_method == "content":
            img = self.content_image.clone()
        elif init_method == "content_with_noise":
            img = self.content_image.clone()
            noise = torch.randn_like(img) * noise_factor
            img = img + noise
            del noise
        elif init_method == "style":
            img = self.style_image.clone()
        elif init_method == "style_with_noise":
            img = self.style_image.clone()
            noise = torch.randn_like(img) * noise_factor
            img = img + noise
            del noise
        else:
            img = self.content_image.clone()
            
        img = torch.clamp(img, 0, 1)
        return img.requires_grad_(True)
    
    def transfer(self, optimizer_type="lbfgs", init_method="content", 
                 num_steps=300, content_weight=1.0, style_weight=1e6, tv_weight=1e-2, 
                 lr=0.01, noise_factor=0.1, print_freq=10, 
                 show_images=True, output_path="output/style_transfer",
                 memory_efficient=True, checkpoint_freq=50):
        """Perform neural style transfer with memory optimization."""
        
        os.makedirs(output_path, exist_ok=True)
        
        # Save original images
        save_image(self.content_image, os.path.join(output_path, "content_original.jpg"))
        save_image(self.style_image, os.path.join(output_path, "style_original.jpg"))
        
        # Initialize image
        generated = self._initialize_image(init_method, noise_factor)
        
        # Setup optimizer
        if optimizer_type.lower() == "lbfgs":
            optimizer = optim.LBFGS([generated], lr=lr, max_iter=20, max_eval=40)
        elif optimizer_type.lower() == "adam":
            optimizer = optim.Adam([generated], lr=lr)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=num_steps//3, gamma=0.5)
        else:
            raise ValueError("Unsupported optimizer. Choose 'lbfgs' or 'adam'.")
        
        # For tracking loss history
        content_loss_history = []
        style_loss_history = []
        tv_loss_history = []
        total_loss_history = []
        
        # For displaying progress
        if show_images and num_steps >= 10:
            display_steps = [int(num_steps * i / 10) for i in range(11)]
        else:
            display_steps = []
            
        # Checkpoint setup
        checkpoint_steps = [i for i in range(0, num_steps, checkpoint_freq)] if checkpoint_freq > 0 else []
        if num_steps - 1 not in checkpoint_steps:
            checkpoint_steps.append(num_steps - 1)
        
        # Start timer
        start_time = time.time()
        last_print_time = start_time
        
        # Function to perform a single optimization step
        def closure():
            optimizer.zero_grad()
            
            if memory_efficient:
                # Process content and style separately to save memory
                with torch.set_grad_enabled(True):
                    all_features = self.feature_extractor(generated)
                    content_layer_features = all_features[self.content_layer]
                    style_layer_features = {layer: all_features[layer] for layer in self.style_layers if layer in all_features}
                    del all_features
                    torch.cuda.empty_cache() if self.device == "cuda" else None
                
                # Compute content loss
                content_loss = nn.functional.mse_loss(
                    content_layer_features, 
                    self.content_features[self.content_layer]
                )
                
                # Compute style loss for each layer
                total_style_loss = 0
                for layer in self.style_layers:
                    if layer in style_layer_features:
                        current_gram = compute_gram_matrix(style_layer_features[layer])
                        layer_style_loss = style_loss_fn(current_gram, self.style_features[layer])
                        total_style_loss += layer_style_loss
                        del current_gram
                
                del content_layer_features
                del style_layer_features
                torch.cuda.empty_cache() if self.device == "cuda" else None
            else:
                # Original implementation (less memory efficient)
                current_features = self.feature_extractor(generated)
                
                # Compute content loss
                content_loss = nn.functional.mse_loss(
                    current_features[self.content_layer], 
                    self.content_features[self.content_layer]
                )
                
                # Compute style loss
                total_style_loss = 0
                for layer in self.style_layers:
                    if layer in current_features:
                        current_gram = compute_gram_matrix(current_features[layer])
                        layer_style_loss = style_loss_fn(current_gram, self.style_features[layer])
                        total_style_loss += layer_style_loss
                
                del current_features
            
            # Compute total variation loss
            variation_loss = tv_loss_fn(generated)
            
            # Combine losses
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
            
            # Clean up
            del content_loss, total_style_loss, variation_loss
            del weighted_content_loss, weighted_style_loss, weighted_tv_loss
            torch.cuda.empty_cache() if self.device == "cuda" else gc.collect()
            
            return total_loss
        
        # Optimization loop
        print(f"Starting neural style transfer with {optimizer_type} optimizer...")
        print(f"Steps: {num_steps}, Content weight: {content_weight}, Style weight: {style_weight}, TV weight: {tv_weight}")
        print("-" * 80)
        print(f"{'Step':>8} | {'Content Loss':>12} | {'Style Loss':>12} | {'TV Loss':>12} | {'Total Loss':>12} | {'Time':>8}")
        print("-" * 80)
        
        if optimizer_type.lower() == "lbfgs":
            # L-BFGS optimizer
            for step in range(num_steps):
                optimizer.step(closure)
                
                # Clamp values
                with torch.no_grad():
                    generated.clamp_(0, 1)
                
                # Display progress
                if step % print_freq == 0 or step == num_steps - 1:
                    current_time = time.time()
                    elapsed = current_time - last_print_time
                    last_print_time = current_time
                    
                    c_loss = content_loss_history[-1] if content_loss_history else 0
                    s_loss = style_loss_history[-1] if style_loss_history else 0
                    t_loss = tv_loss_history[-1] if tv_loss_history else 0
                    total_loss = total_loss_history[-1] if total_loss_history else 0
                    
                    print(f"{step:>8} | {c_loss:>12.4e} | {s_loss:>12.4e} | {t_loss:>12.4e} | {total_loss:>12.4e} | {elapsed:>8.2f}")
                
                # Display image
                if show_images and step in display_steps:
                    save_image(generated, os.path.join(output_path, f"iter_{step}.jpg"))
                    if show_images:
                        show_progress(generated.detach().clone(), step, num_steps)
                
                # Save checkpoints
                if step in checkpoint_steps:
                    checkpoint_path = os.path.join(output_path, f"checkpoint_{step}.pt")
                    torch.save({
                        'step': step,
                        'generated': generated.detach().cpu(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'content_loss': content_loss_history,
                        'style_loss': style_loss_history,
                        'tv_loss': tv_loss_history,
                        'total_loss': total_loss_history
                    }, checkpoint_path)
                
                # Clear GPU cache
                if step % 10 == 0 and self.device == "cuda":
                    torch.cuda.empty_cache()
        else:
            # Adam optimizer
            for step in range(num_steps):
                loss = closure()
                optimizer.step()
                scheduler.step()
                
                # Clamp values
                with torch.no_grad():
                    generated.clamp_(0, 1)
                
                # Print progress
                if step % print_freq == 0 or step == num_steps - 1:
                    current_time = time.time()
                    elapsed = current_time - last_print_time
                    last_print_time = current_time
                    
                    c_loss = content_loss_history[-1]
                    s_loss = style_loss_history[-1]
                    t_loss = tv_loss_history[-1]
                    total_loss = total_loss_history[-1]
                    
                    print(f"{step:>8} | {c_loss:>12.4e} | {s_loss:>12.4e} | {t_loss:>12.4e} | {total_loss:>12.4e} | {elapsed:>8.2f}")
                
                # Display image
                if show_images and step in display_steps:
                    save_image(generated, os.path.join(output_path, f"iter_{step}.jpg"))
                    if show_images:
                        show_progress(generated.detach().clone(), step, num_steps)
                
                # Save checkpoints
                if step in checkpoint_steps:
                    checkpoint_path = os.path.join(output_path, f"checkpoint_{step}.pt")
                    torch.save({
                        'step': step,
                        'generated': generated.detach().cpu(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'content_loss': content_loss_history,
                        'style_loss': style_loss_history,
                        'tv_loss': tv_loss_history,
                        'total_loss': total_loss_history
                    }, checkpoint_path)
                
                # Clear GPU cache
                if step % 10 == 0 and self.device == "cuda":
                    torch.cuda.empty_cache()
        
        # Final timing
        total_time = time.time() - start_time
        print(f"Style transfer completed in {total_time:.2f} seconds")
        
        # Save final result
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
        
        # Clear GPU memory
        if self.device == "cuda":
            torch.cuda.empty_cache()
        else:
            gc.collect()
            
        return generated.detach()
    
    def __del__(self):
        """Destructor to free memory."""
        # Clear references to large objects
        for attr in ['content_features', 'style_features', 'content_image', 'style_image', 'feature_extractor']:
            if hasattr(self, attr):
                delattr(self, attr)
            
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()