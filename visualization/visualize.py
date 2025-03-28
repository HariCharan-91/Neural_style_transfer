import matplotlib.pyplot as plt

def feature_visualization(feature_maps, layer_name="Layer 19"):
    """
    Visualizes first 16 channels of feature maps from a single layer
    """
    if feature_maps is None:
        raise ValueError("No feature maps detected!")
    
    features = feature_maps.detach().cpu().squeeze(0)
    num_channels = min(16, features.shape[0])  # Show first 16 channels
    
    plt.figure(figsize=(15, 8))
    plt.suptitle(f"Style Feature Maps: {layer_name}", fontsize=14)
    
    for i in range(num_channels):
        plt.subplot(4, 4, i+1)
        plt.imshow(features[i], cmap='viridis')
        plt.axis('off')
        plt.title(f"Ch {i+1}")
    
    plt.tight_layout()
    plt.show()