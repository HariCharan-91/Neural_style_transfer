import torch
from models.neural_style_transfer import NeuralStyleTransfer

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Path to your style and content images
    style_image_path = r"data\style\neural_style_transfer_5_1 (1).jpg"  # Replace with your actual style image path
    content_image_path = r"data\content\wallhaven-9d9g38.jpg"  # Replace with your actual content image path
    
    # Create neural style transfer model with custom layers
    nst = NeuralStyleTransfer(
        style_image_path=style_image_path,
        content_image_path=content_image_path,
        style_layers=[1, 6, 11, 20],  # These are good layers for style in VGG16
        content_layers=[11],  # Layer 11 (relu2_2) is good for content preservation
        device=device
    )
    
    # Optional: Set custom weights for each layer
    # nst.set_style_weights({
    #     1: 0.5,   # Early layer (texture details)
    #     6: 1.0,   # Mid layer
    #     11: 1.5,  # Mid-deep layer
    #     20: 2.0   # Deep layer (larger patterns)
    # })
    
    # Optional: Set custom weights for content layers
    # nst.set_content_weights({
    #     11: 1.0
    # })
    
    # Optional: Set overall balance between style and content
    # Higher style_weight emphasizes artistic style more
    # Higher content_weight preserves more of the original content
    nst.set_overall_weights(style_weight=1e6, content_weight=1.0)
    
    # Run style transfer
    nst.transfer_style(
        iterations=1000,  # More iterations = better results but slower
        output_path="output/style_transfer_results",
        learning_rate=0.01,
        optimizer_type="adam",  # 'adam' is faster, 'lbfgs' gives better quality
        use_content_init=True,  # Initialize with content image (helps preserve structure)
        init_noise_scale=0.1,   # Amount of noise to add to initialization
        save_interval=50        # Save intermediate results every 50 iterations
    )
    
    # Alternative configuration for higher quality (but slower)
    # nst.transfer_style(
    #     iterations=300,
    #     output_path="output/high_quality_results",
    #     learning_rate=1.0,
    #     optimizer_type="lbfgs",
    #     lbfgs_max_iter=20,
    #     use_content_init=True,
    #     save_interval=20
    # )