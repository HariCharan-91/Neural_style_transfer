# import torch
# from reconstruction.content_recon import ContentReconstructor

# if __name__ == "__main__":
#     device = "cuda" if torch.cuda.is_available() else "cpu"
    
#     # Path to your content image
#     image_path = r"data\content\wallhaven-9d9g38.jpg"  # Replace with your actual image path
    
#  # Create reconstructor
#     reconstructor = ContentReconstructor(
#         image_path=image_path,
#         target_layer=10,  # Adjust layer as needed
#         device=device
#     )
    
#     # Run reconstruction with Adam optimizer (default)
#     reconstructor.reconstruct(
#         iterations=300,
#         output_path="output/adam_results",
#         learning_rate=0.05,
#         use_content_init=True,  # Start with content + noise (faster convergence)
#         optimizer_type="adam"    # Use Adam optimizer
#     )
    
#     # Or run with LBFGS optimizer
#     reconstructor.reconstruct(
#         iterations=200,  # LBFGS typically needs fewer iterations
#         output_path="output/lbfgs_results",
#         learning_rate=0.01,  # Lower learning rate for LBFGS
#         use_content_init=True,
#         optimizer_type="lbfgs",  # Use LBFGS optimizer
#         lbfgs_max_iter=20        # Max iterations per optimization step
#     )



import torch
from reconstruction.style_recon import StyleReconstructor

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Path to your style image
    style_image_path = r"data\style\original.jpg"  # Replace with your actual style image path
    
    # Create style reconstructor with custom layers
    # You can customize which layers to use for style extraction
    reconstructor = StyleReconstructor(
        style_image_path=style_image_path,
        target_layers=[1, 6, 11, 20],  # These are good layers for style in VGG16
        device=device
    )
    
    # Optional: Set custom weights for each layer
    # reconstructor.set_style_weights({
    #     1: 0.5,   # Early layer (texture details)
    #     6: 1.0,   # Mid layer
    #     11: 1.5,  # Mid-deep layer
    #     20: 2.0   # Deep layer (larger patterns)
    # })
    
    # Run style reconstruction
    # reconstructor.reconstruct(
    #     iterations=500,
    #     output_path="output/style_results",
    #     learning_rate=0.01,
    #     img_size=256,            # Size of the generated image
    #     optimizer_type="adam",   # Use Adam optimizer
    #     use_style_init=False     # Start from random noise (better for style)
    # )
    
    # Alternatively, use LBFGS optimizer
    reconstructor.reconstruct(
        iterations=200,
        output_path="output/style_lbfgs_results",
        learning_rate=0.005,
        img_size=256,
        optimizer_type="lbfgs",
        lbfgs_max_iter=20
    )