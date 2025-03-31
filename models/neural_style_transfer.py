import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from utils.image_utils import preprocess_image
from models.vgg import VggFeatureExtractor
# from reconstruction.style_recon import StyleReconstructor

class NeuralStyleTransfer:
    def __init__(self, style_image_path=None, content_image_path=None, 
                 style_layers=None, content_layers=None, device="cuda"):
        """
        Initialize neural style transfer with style and content images and layers.
        
        Args:
            style_image_path: Path to the style image
            content_image_path: Path to the content image
            style_layers: List of VGG layer indices to extract style features from
                         Default is [1, 6, 11, 20] (typical style layers in VGG16)
            content_layers: List of VGG layer indices to extract content features from
                           Default is [11] (typical content layer in VGG16)
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device
        
        # Set default style layers if not provided
        if style_layers is None:
            self.style_layers = [1, 6, 11, 20]  # Common style layers in VGG16
        else:
            self.style_layers = style_layers
            
        # Set default content layers if not provided
        if content_layers is None:
            self.content_layers = [11]  # Common content layer in VGG16
        else:
            self.content_layers = content_layers
        
        # Process images if paths provided
        self.style_img = None
        self.content_img = None
        
        if style_image_path:
            self.style_img = preprocess_image(style_image_path, device=device)
            
        if content_image_path:
            self.content_img = preprocess_image(content_image_path, device=device)
            
        # Create feature extractors for each layer
        self.style_extractors = {
            layer: VggFeatureExtractor(target_layer=layer, device=device)
            for layer in self.style_layers
        }
        
        self.content_extractors = {
            layer: VggFeatureExtractor(target_layer=layer, device=device)
            for layer in self.content_layers
        }
        
        # Style weights for each layer (deeper layers get more weight)
        self.style_weights = {}
        total_style_layers = len(self.style_layers)
        for i, layer in enumerate(self.style_layers):
            # Increasing weight for deeper layers
            self.style_weights[layer] = 0.5 + (i / total_style_layers)
            
        # Content weights for each layer
        self.content_weights = {}
        for layer in self.content_layers:
            self.content_weights[layer] = 1.0
            
        # Overall weighting between style and content
        self.style_weight = 1e6  # Default style weight
        self.content_weight = 1.0  # Default content weight
            
    def set_style_image(self, img_tensor):
        """Set preprocessed style image tensor"""
        self.style_img = img_tensor
        
    def set_content_image(self, img_tensor):
        """Set preprocessed content image tensor"""
        self.content_img = img_tensor
        
    def set_style_weights(self, weight_dict):
        """Set custom weights for style layers"""
        self.style_weights = weight_dict
        
    def set_content_weights(self, weight_dict):
        """Set custom weights for content layers"""
        self.content_weights = weight_dict
        
    def set_overall_weights(self, style_weight, content_weight):
        """Set overall weights for style vs content"""
        self.style_weight = style_weight
        self.content_weight = content_weight
        
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
        Extract style features (Gram matrices) from all style layers
        
        Args:
            image: Input image tensor
            
        Returns:
            Dictionary of style features (Gram matrices) for each layer
        """
        style_features = {}
        for layer, extractor in self.style_extractors.items():
            features = extractor(image)
            style_features[layer] = self._gram_matrix(features)
        return style_features
        
    def extract_content_features(self, image):
        """
        Extract content features from all content layers
        
        Args:
            image: Input image tensor
            
        Returns:
            Dictionary of content features for each layer
        """
        content_features = {}
        for layer, extractor in self.content_extractors.items():
            content_features[layer] = extractor(image)
        return content_features
        
    def _calculate_style_loss(self, image, target_style_features):
        """Calculate weighted style loss across all style layers"""
        total_loss = 0
        current_style_features = self.extract_style_features(image)
        
        for layer in self.style_layers:
            weight = self.style_weights[layer]
            current_gram = current_style_features[layer]
            target_gram = target_style_features[layer]
            
            # MSE between Gram matrices
            layer_loss = nn.functional.mse_loss(current_gram, target_gram)
            total_loss += weight * layer_loss
            
        return total_loss
        
    def _calculate_content_loss(self, image, target_content_features):
        """Calculate weighted content loss across all content layers"""
        total_loss = 0
        current_content_features = self.extract_content_features(image)
        
        for layer in self.content_layers:
            weight = self.content_weights[layer]
            current_features = current_content_features[layer]
            target_features = target_content_features[layer]
            
            # MSE between feature maps
            layer_loss = nn.functional.mse_loss(current_features, target_features)
            total_loss += weight * layer_loss
            
        return total_loss
        
    def _calculate_total_loss(self, image, target_style_features, target_content_features):
        """Calculate combined style and content loss"""
        style_loss = self._calculate_style_loss(image, target_style_features)
        content_loss = self._calculate_content_loss(image, target_content_features)
        
        # Combine losses with overall weights
        total_loss = self.style_weight * style_loss + self.content_weight * content_loss
        
        return total_loss, style_loss, content_loss
    
    def transfer_style(self, iterations=1000, output_path="style_transfer_output", 
                      learning_rate=0.01, optimizer_type="adam", lbfgs_max_iter=20,
                      use_content_init=True, init_noise_scale=0.1,
                      save_interval=50):
        """
        Apply style transfer to combine style and content images
        
        Args:
            iterations: Number of optimization iterations
            output_path: Directory to save outputs
            learning_rate: Learning rate for optimization
            optimizer_type: Type of optimizer to use ('adam' or 'lbfgs')
            lbfgs_max_iter: Max iterations per step for LBFGS optimizer
            use_content_init: Whether to initialize with content image (True) or noise (False)
            init_noise_scale: Scale of noise for initialization
            save_interval: Interval to save intermediate results
        """
        # Check that images are set
        if self.style_img is None:
            raise ValueError("No style image set for style transfer")
        if self.content_img is None:
            raise ValueError("No content image set for style transfer")
            
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Extract target features
        with torch.no_grad():
            target_style_features = self.extract_style_features(self.style_img)
            target_content_features = self.extract_content_features(self.content_img)
        
        # Initialize generated image
        if use_content_init:
            # Initialize with content image + noise
            generated = self.content_img.clone() + init_noise_scale * torch.randn_like(self.content_img, device=self.device)
        else:
            # Initialize with noise (using content image shape)
            generated = torch.rand_like(self.content_img, device=self.device)
            
        generated.requires_grad_(True)
        
        # Setup optimizer
        optimizer_type = optimizer_type.lower()
        if optimizer_type == "adam":
            optimizer = optim.Adam([generated], lr=learning_rate)
        elif optimizer_type == "lbfgs":
            optimizer = optim.LBFGS([generated], lr=learning_rate, max_iter=lbfgs_max_iter)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}. Use 'adam' or 'lbfgs'")
        
        # Track loss history
        loss_history = {
            'total': [],
            'style': [],
            'content': []
        }
        
        # Save original images
        self._save_image(self.style_img, os.path.join(output_path, "original_style.jpg"))
        self._save_image(self.content_img, os.path.join(output_path, "original_content.jpg"))
        
        # Training loop
        for i in range(iterations):
            # Different optimization step depending on optimizer type
            if optimizer_type == "adam":
                # Calculate losses
                total_loss, style_loss, content_loss = self._calculate_total_loss(
                    generated, target_style_features, target_content_features
                )
                
                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                # Record losses
                loss_history['total'].append(total_loss.item())
                loss_history['style'].append(style_loss.item())
                loss_history['content'].append(content_loss.item())
                
            else:  # LBFGS
                # Define closure for LBFGS
                def closure():
                    optimizer.zero_grad()
                    total_loss, style_loss, content_loss = self._calculate_total_loss(
                        generated, target_style_features, target_content_features
                    )
                    
                    # Record losses
                    loss_history['total'].append(total_loss.item())
                    loss_history['style'].append(style_loss.item())
                    loss_history['content'].append(content_loss.item())
                    
                    total_loss.backward()
                    return total_loss
                
                # Run optimization step
                optimizer.step(closure)
            
            # Print progress
            if (i+1) % 10 == 0:
                print(f"Iteration {i+1}/{iterations} - "
                      f"Total Loss: {loss_history['total'][-1]:.4f}, "
                      f"Style Loss: {loss_history['style'][-1]:.4f}, "
                      f"Content Loss: {loss_history['content'][-1]:.4f}")
            
            # Clamp pixel values to valid range [0,1]
            with torch.no_grad():
                generated.data.clamp_(0, 1)
            
            # Save intermediate results
            if (i+1) % save_interval == 0 or i == 0:
                self._save_image(generated, os.path.join(output_path, f"stylized_iter_{i+1}.jpg"))
        
        # Save final results
        self._save_image(generated, os.path.join(output_path, "final_stylized_image.jpg"))
        self._save_losses(loss_history, output_path)
        
        return generated, loss_history
    
    def _save_image(self, tensor, path):
        """Convert tensor to PIL image and save"""
        img = tensor.squeeze(0).cpu().detach().clamp(0, 1)
        transforms.ToPILImage()(img).save(path)
    
    def _save_losses(self, loss_history, output_path):
        """Save loss values and plot"""
        # Save as text files
        for loss_type, values in loss_history.items():
            np.savetxt(os.path.join(output_path, f"{loss_type}_loss_values.txt"), values)
        
        # Create loss plots
        plt.figure(figsize=(12, 8))
        
        # Plot all losses on the same graph
        plt.subplot(2, 1, 1)
        plt.plot(loss_history['total'], label='Total Loss')
        plt.plot(loss_history['style'], label='Style Loss')
        plt.plot(loss_history['content'], label='Content Loss')
        plt.title("Neural Style Transfer Losses")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        
        # Plot style and content losses separately
        plt.subplot(2, 2, 3)
        plt.plot(loss_history['style'], 'r-')
        plt.title("Style Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        plt.plot(loss_history['content'], 'g-')
        plt.title("Content Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "loss_plots.png"))
        plt.close()