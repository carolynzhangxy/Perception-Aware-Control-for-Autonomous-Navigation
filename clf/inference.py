import torch
from torchvision import transforms
from PIL import Image
from .clf import ImageClassifier
# from clf import ImageClassifier
import argparse
from typing import Literal

def load_classes(classes_file):
    with open(classes_file, 'r') as f:
        return [line.strip() for line in f.readlines()]

def load_model(model_path, num_classes, device='cpu'):
    model = ImageClassifier(num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def predict(model, image_tensor, device, classes):
    with torch.no_grad():
        outputs = model(image_tensor.to(device))
        _, predicted = torch.max(outputs, 1)
        return classes[predicted.item()]
    
class Predector:
    def __init__(self,name: Literal["car","new"] ):
        # classes_file = f"/home/perception/Obstacle_Avoidance_2024/clf/{name}_classes.txt"
        # model_path = f"/home/perception/Obstacle_Avoidance_2024/clf/{name}_model.pth"
        classes_file = f"C:/Users/Perception/Desktop/Obstacle_Avoidance/Obstacle_Avoidance_2024/clf/{name}_classes.txt"
        model_path = f"C:/Users/Perception/Desktop/Obstacle_Avoidance/Obstacle_Avoidance_2024/clf/{name}_model.pth"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        classes = load_classes(classes_file)
        model = load_model(model_path, len(classes), device=device)
        self.model = model
        self.classes = classes
        
    def predict(self,image_path):
        image_tensor = preprocess_image(image_path)
        prediction = predict(self.model, image_tensor, 'cuda', self.classes)
        return prediction
    
def main():
    parser = argparse.ArgumentParser(description='Image classification inference')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('model_path', type=str, help='Path to the trained model')
    parser.add_argument('classes_file', type=str, help='Path to the classes file')
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = load_classes(args.classes_file)
    model = load_model(args.model_path, len(classes), device=device)
    image_tensor = preprocess_image(args.image_path)
    prediction = predict(model, image_tensor, device, classes)

    print(f"The image is classified as: {prediction}")

if __name__ == "__main__":
    main()