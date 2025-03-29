import torch
import PIL.Image as Image
from models.vgg import VggFeatureExtractor
from visualization.visualize import feature_visualization

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_path = r"data\content\wallhaven-9d9g38.jpg"
    try:
        img = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return
    model = VggFeatureExtractor(target_layer= 1 ,device = device)

    with torch.no_grad():
        features = model(img)

    feature_visualization(features)

if __name__ == "__main__":
    main()