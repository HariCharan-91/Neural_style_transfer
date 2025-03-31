# Content Reconstruction

import torch
from reconstruction.content_recon import ContentReconstructor


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Path to your content image
    image_path = r"data\content\wallhaven-9d9g38.jpg"  # Replace with your actual image path
    
    reconstructor = ContentReconstructor(image_path=image_path, target_layer=10, device="cuda")
    generated_img, loss_hist = reconstructor.reconstruct(
            iterations= 300,
            output_path="output/content_reconstruction/lbfgs_results",
            learning_rate=0.05,
            use_content_init=False,
            optimizer_type="lbfgs",  # Change to "lbfgs" to use LBFGS optimizer
            lbfgs_max_iter=20,
            tv_weight=0.001,
            noise_scale=0.1
        )

# Style Reconstruction

# import torch
# from reconstruction.style_recon import StyleReconstructor

# if __name__ == "__main__":
#     device = "cuda" if torch.cuda.is_available() else "cpu"
    
#     # Path to your style image
#     style_image_path = r"data\style\original.jpg"  # Replace with your actual style image path
    
#     # Create style reconstructor with custom layers
#     # You can customize which layers to use for style extraction
#     reconstructor = StyleReconstructor(
#         style_image_path=style_image_path,
#         target_layers=[1, 6, 11, 20],  # These are good layers for style in VGG16
#         device=device
#     )
    
#     # Optional: Set custom weights for each layer
#     # reconstructor.set_style_weights({
#     #     1: 0.5,   # Early layer (texture details)
#     #     6: 1.0,   # Mid layer
#     #     11: 1.5,  # Mid-deep layer
#     #     20: 2.0   # Deep layer (larger patterns)
#     # })
    
#     # Run style reconstruction
#     # reconstructor.reconstruct(
#     #     iterations=500,
#     #     output_path="output/style_results",
#     #     learning_rate=0.01,
#     #     img_size=256,            # Size of the generated image
#     #     optimizer_type="adam",   # Use Adam optimizer
#     #     use_style_init=False     # Start from random noise (better for style)
#     # )
    
#     # Alternatively, use LBFGS optimizer
#     reconstructor.reconstruct(
#         iterations=200,
#         output_path="output/style_lbfgs_results",
#         learning_rate=0.005,
#         img_size=256,
#         optimizer_type="lbfgs",
#         lbfgs_max_iter=20
#     )


# Feature Extraction


# import torch
# from models.vgg import VggFeatureExtractor
# from visualization.visualize import feature_visualization
# from utils.image_utils import preprocess_image

# if __name__ == "__main__":
    
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     image_path = r"data\style\wallhaven-p9532p.jpg"

#     tensor = preprocess_image(image_path)

#     model = VggFeatureExtractor(1, device=device)

#     features = model(tensor)

#     feature_visualization(features,"Layer 10")


