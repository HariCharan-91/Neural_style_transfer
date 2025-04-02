# Content Reconstruction

# import torch
# from reconstruction.content_recon import ContentReconstructor


# if __name__ == "__main__":
#     device = "cuda" if torch.cuda.is_available() else "cpu"
    
#     # Path to your content image
#     image_path = r"data\content\wallhaven-9d9g38.jpg"  # Replace with your actual image path
    
#     reconstructor = ContentReconstructor(image_path=image_path, target_layer=10, device="cuda")
#     generated_img, loss_hist = reconstructor.reconstruct(
#             iterations= 300,
#             output_path="output/content_reconstruction/lbfgs_results",
#             learning_rate=0.05,
#             use_content_init=False,
#             optimizer_type="lbfgs",  # Change to "lbfgs" to use LBFGS optimizer
#             lbfgs_max_iter=20,
#             tv_weight=0.001,
#             noise_scale=0.1
#         )

# Style Reconstruction

# import torch
# from reconstruction.style_recon import StyleReconstructor
# from utils.image_utils import preprocess_image
# from PIL import Image

# if __name__ == "__main__":
#     device = "cuda" if torch.cuda.is_available() else "cpu"
    
#     # Path to your style image
#     style_image_path = r"data\style\ben_giles.jpg"  # Replace with your actual style image path
    
#     # reconstructor = StyleReconstructor(style_image_path=style_image_path, target_layers=[1, 6, 11, 20], device="cuda")
#     # generated_img, loss_hist = reconstructor.reconstruct(
#     #     iterations=100,
#     #     output_path= "output/style_reconstruction/lbfgs_results",
#     #     learning_rate=0.01,
#     #     img_size=256,
#     #     optimizer_type="lbfgs",  # Change to "lbfgs" to use LBFGS optimizer
#     #     lbfgs_max_iter=20,
#     #     use_style_init=False,
#     #     init_noise_scale=0.1,
#     #     tv_weight=0.001  # Adjust TV loss weight as desired
#     # )
    
#     preprocess_image(style_image_path , display=True)


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

import torch 

from models.neural_style import NeuralStyleTransfer


# Example usage
nst = NeuralStyleTransfer(
    content_image_path=r"data\content\wallhaven-9d9g38.jpg",
    style_image_path=r"data\style\wallhaven-p9532p.jpg",
    content_layer=21,  # Typically conv4_2 in VGG
    style_layers=[1, 6, 11, 20],  # Common style layers
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Perform style transfer
result, losses = nst.transfer_style(
    iterations=150,
    output_path="results",
    content_weight=1,
    style_weight=1e5,
    tv_weight=1e-3,
    learning_rate=0.1,
    optimizer_type="lbfgs",
    init_image="content"
)