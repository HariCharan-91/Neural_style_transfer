from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch


def preprocess_image(image_path, size=224, device="cuda", display=False):
    """
    Optimized function to load an image, resize it using high-quality interpolation,
    convert it to a tensor, and optionally display it.
    
    Args:
        image_path (str): Path to the image file.
        size (int): Target size (width and height) for resizing.
        device (str): Device to load the tensor onto ('cuda' or 'cpu').
        display (bool): Whether to display the original and preprocessed images.
        
    Returns:
        img_tensor (torch.Tensor): The preprocessed image tensor.
    """
    try:
        # Load and convert the image to RGB
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

    if display:
        plt.figure()
        plt.imshow(img)
        plt.title("Original Image")
        plt.axis("off")
        plt.show()

    # Define preprocessing with bicubic interpolation and antialiasing for better clarity
    preprocess = transforms.Compose([
        transforms.Resize((size, size), interpolation=Image.BICUBIC, antialias=True),
        transforms.ToTensor()  # Scales pixel values to [0, 1]
    ])

    # Apply preprocessing and add a batch dimension, then move to the specified device
    img_tensor = preprocess(img).unsqueeze(0).to(device)

    if display:
        # Convert tensor back to a PIL image for display purposes
        img_after = transforms.ToPILImage()(img_tensor.squeeze(0).cpu())
        plt.figure()
        plt.imshow(img_after)
        plt.title("Preprocessed Image")
        plt.axis("off")
        plt.show()
    return img_tensor




def save_image(tensor, path):
    """Convert tensor to PIL image and save"""
    img = tensor.squeeze(0).cpu().detach().clamp(0, 1)
    transforms.ToPILImage()(img).save(path)

def compute_gram_matrix(features):
    """Compute the Gram matrix from features."""
    _, channels, height, width = features.size()
    features_reshaped = features.view(channels, height * width)
    gram = torch.mm(features_reshaped, features_reshaped.t())
    # Normalize by the size of the feature map
    return gram.div(channels * height * width)