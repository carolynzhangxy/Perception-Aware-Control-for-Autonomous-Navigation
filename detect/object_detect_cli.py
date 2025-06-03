import argparse
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
try:
    from .object_detector import ObjectDetector
except:
    from object_detector import ObjectDetector
def load_model(model_path: str) -> ObjectDetector:
    model = ObjectDetector(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'),weights_only=False))
    model.eval()
    return model

def preprocess_image(image_path: str) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)
def preprocess_image_array(image: torch.Tensor) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def detect_object(model: ObjectDetector, image: torch.Tensor) -> tuple:
    with torch.no_grad():
        output = model(image)
    return tuple(output[0].tolist())

def visualize_result(image_path: str, coordinates: tuple):
    image = Image.open(image_path)
    plt.imshow(image)
    plt.plot(coordinates[0], coordinates[1], 'ro', markersize=10)
    plt.title(f"Detected coordinates: {coordinates}")
    plt.axis('off')
    plt.show()

def main(image_path: str, model_path: str):
    model = load_model(model_path)
    image = preprocess_image(image_path)
    coordinates = detect_object(model, image)
    
    print(f"Detected object coordinates: {coordinates}")
    visualize_result(image_path, coordinates)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect object in image using trained model")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument("model_path", type=str, help="Path to the trained model file")
    
    args = parser.parse_args()
    
    main(args.image_path, args.model_path)