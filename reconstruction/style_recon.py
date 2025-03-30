import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from utils.image_utils import preprocess_image
from models.vgg import VggFeatureExtractor

class StyleReconstructor:
    def __init__(self, style_image_path=None, target_layers=None, device="cuda"):
        """
        Initialize style reconstructor with a style image and target layers.
        
        Args:
            style_image_path: Path to the style image
            target_layers: List of VGG layer indices to extract style features from
                          Default is [1, 6, 11, 20] (typical style layers in VGG16)
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device
        
        # Set default style layers if not provided
        if target_layers is None:
            self.target_layers = [1, 6, 11, 20]  # Common style layers in VGG16
        else:
            self.target_layers = target_layers
        
        # Process style image if path provided
        if style_image_path:
            self.style_img = preprocess_image(style_image_path, device=device)
        else:
            self.style_img = None
            
        # Create feature extractors for each target layer
        self.extractors = {
            layer: VggFeatureExtractor(target_layer=layer, device=device)
            for layer in self.target_layers
        }
        
        # Style weights for each layer (deeper layers get more weight)
        self.style_weights = {}
        total_layers = len(self.target_layers)
        for i, layer in enumerate(self.target_layers):
            # Increasing weight for deeper layers
            self.style_weights[layer] = 0.5 + (i / total_layers)
            
    def set_style_image(self, img_tensor):
        """Set preprocessed style image tensor"""
        self.style_img = img_tensor
        
    def set_style_weights(self, weight_dict):
        """Set custom weights for style layers"""
        self.style_weights = weight_dict
        
    def _gram_matrix(self, x):
        """
        Calculate Gram matrix for style representation
        
        Args:
            x: Feature maps tensor of shape [batch, channels, height, width]
            
        Returns:
            Gram matrix of shape [batch, channels, channels]
        """
        batch_size, channels, height, width = x.size()
        features = x.view(batch_size, channels, height * width)
        gram = torch.bmm(features, features.transpose(1, 2))
        # Normalize by the number of elements
        return gram.div(channels * height * width)
        
    def extract_style_features(self, image):
        """
        Extract style features (Gram matrices) from all target layers
        
        Args:
            image: Input image tensor
            
        Returns:
            Dictionary of style features (Gram matrices) for each layer
        """
        style_features = {}
        for layer, extractor in self.extractors.items():
            features = extractor(image)
            style_features[layer] = self._gram_matrix(features)
        return style_features
        
    def reconstruct(self, iterations=500, output_path="style_output", 
                    learning_rate=0.01, img_size=256, 
                    optimizer_type="adam", lbfgs_max_iter=20, 
                    use_style_init=False, init_noise_scale=0.1):
        """
        Reconstruct an image that matches the style of the input style image
        
        Args:
            iterations: Number of optimization iterations
            output_path: Directory to save outputs
            learning_rate: Learning rate for optimization
            img_size: Size of the output image
            optimizer_type: Type of optimizer to use ('adam' or 'lbfgs')
            lbfgs_max_iter: Max iterations per step for LBFGS optimizer
            use_style_init: Whether to initialize with style image + noise
            init_noise_scale: Scale of noise for initialization
        """
        # Check that style image is set
        if self.style_img is None:
            raise ValueError("No style image set for reconstruction")
            
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Extract target style features
        with torch.no_grad():
            target_style_features = self.extract_style_features(self.style_img)
        
        # Initialize image
        if use_style_init:
            # Initialize with style image + noise
            generated = self.style_img.clone() + init_noise_scale * torch.randn_like(self.style_img, device=self.device)
            generated.requires_grad_(True)
        else:
            # Initialize with noise
            generated = torch.rand(1, 3, img_size, img_size, requires_grad=True, device=self.device)
        
        # Setup optimizer
        optimizer_type = optimizer_type.lower()
        if optimizer_type == "adam":
            optimizer = optim.Adam([generated], lr=learning_rate)
        elif optimizer_type == "lbfgs":
            optimizer = optim.LBFGS([generated], lr=learning_rate, max_iter=lbfgs_max_iter)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}. Use 'adam' or 'lbfgs'")
        
        # Track loss history
        loss_history = []
        
        # Save original style image
        self._save_image(self.style_img, os.path.join(output_path, "original_style.jpg"))
        
        # Training loop
        for i in range(iterations):
            # Different optimization step depending on optimizer type
            if optimizer_type == "adam":
                # Calculate style loss
                total_loss = self._calculate_style_loss(generated, target_style_features)
                
                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                # Record loss
                current_loss = total_loss.item()
                
            else:  # LBFGS
                # Define closure for LBFGS
                def closure():
                    optimizer.zero_grad()
                    total_loss = self._calculate_style_loss(generated, target_style_features)
                    total_loss.backward()
                    return total_loss
                
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
                self._save_image(generated, os.path.join(output_path, f"style_iter_{i+1}.jpg"))
        
        # Save final results
        self._save_image(generated, os.path.join(output_path, "style_reconstruction.jpg"))
        self._save_loss(loss_history, output_path, prefix="style_")
        
        return generated, loss_history
    
    def _calculate_style_loss(self, image, target_style_features):
        """Calculate weighted style loss across all target layers"""
        total_loss = 0
        current_style_features = self.extract_style_features(image)
        
        for layer in self.target_layers:
            weight = self.style_weights[layer]
            current_gram = current_style_features[layer]
            target_gram = target_style_features[layer]
            
            # MSE between Gram matrices
            layer_loss = nn.functional.mse_loss(current_gram, target_gram)
            total_loss += weight * layer_loss
            
        return total_loss
        
    def _save_image(self, tensor, path):
        """Convert tensor to PIL image and save"""
        img = tensor.squeeze(0).cpu().detach().clamp(0, 1)
        transforms.ToPILImage()(img).save(path)
    
    def _save_loss(self, loss_history, output_path, prefix=""):
        """Save loss values and plot"""
        # Save as text file
        np.savetxt(os.path.join(output_path, f"{prefix}loss_values.txt"), loss_history)
        
        # Create loss plot
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history)
        plt.title("Style Reconstruction Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.savefig(os.path.join(output_path, f"{prefix}loss_plot.png"))
        plt.close()