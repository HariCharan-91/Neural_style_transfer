from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def preprocess_image(image_path, size=224, device="cuda", display=False):
    """
    Load an image and convert to tensor without normalization.
    
    Args:
        image_path: Path to the image file.
        size: The size to which the image will be resized (size x size).
        device: The device on which to load the tensor ('cuda' or 'cpu').
        display: If True, displays the image before and after preprocessing.
    
    Returns:
        img_tensor: The preprocessed image tensor.
    """
    try:
        # Load the image and convert to RGB
        img = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return None

    # Optionally display the original image
    if display:
        plt.figure()
        plt.imshow(img)
        plt.title("Original Image")
        plt.axis("off")
        plt.show()

    # Define the preprocessing steps
    preprocess = transforms.Compose([
        transforms.Resize((size, size), interpolation=Image.BICUBIC),
        transforms.ToTensor()  # Scales pixel values to [0, 1]
    ])

    # Apply preprocessing and add a batch dimension
    img_tensor = preprocess(img).unsqueeze(0).to(device)

    # Optionally display the preprocessed image (converted back to PIL)
    if display:
        # Remove the batch dimension and move to CPU
        processed_img = img_tensor.squeeze(0).cpu()
        # Convert tensor to PIL image
        img_after = transforms.ToPILImage()(processed_img)
        plt.figure()
        plt.imshow(img_after)
        plt.title("Preprocessed Image")
        plt.axis("off")
        plt.show()

    return img_tensor
