from PIL import Image
import torchvision.transforms as transforms

def preprocess_image(image_path, size=224, device="cuda"):
    """
    Load an image and convert to tensor without normalization
    """
    try:
        img = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return None

    preprocess = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()  # Scales to [0, 1]
    ])

    img_tensor = preprocess(img).unsqueeze(0).to(device)
    return img_tensor