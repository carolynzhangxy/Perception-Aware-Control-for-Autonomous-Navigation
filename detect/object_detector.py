import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import argparse
import os
from typing import Tuple, List
import numpy as np

class ObjectDetector(nn.Module):
    def __init__(self, input_shape: Tuple[int, int, int] = (3, 224, 224), num_classes: int = 2):  # 修改这里
        super(ObjectDetector, self).__init__()
        
        self.features = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(input_shape[0], 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第二个卷积块
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第三个卷积块
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 计算展平后的特征维度
        self.feature_size = self._get_conv_output(input_shape)
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.feature_size, num_classes)
        )
    
    def _get_conv_output(self, shape: Tuple[int, int, int]) -> int:
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output_feat = self.features(input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class CustomDataset(Dataset):
    def __init__(self, excel_path: str, transform=None):
        self.data = pd.read_excel(excel_path)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.data.iloc[idx]['Path']
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        coordinates = eval(self.data.iloc[idx]['Agent Coordinate'])
        coordinates = torch.tensor(coordinates, dtype=torch.float32)
        
        return image, coordinates

def train(model: nn.Module, train_loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: torch.device):
    model.train()
    running_loss = 0.0
    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss / len(train_loader)

def main(excel_path: str, num_epochs: int = 10, batch_size: int = 32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = CustomDataset(excel_path, transform=transform)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = ObjectDetector(num_classes=2).to(device)  # 修改这里
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    
    for epoch in range(num_epochs):
        avg_loss = train(model, train_loader, criterion, optimizer, device)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    # 保存模型
    torch.save(model.state_dict(), 'object_detector_model.pth')
    print("Training completed. Model saved as 'object_detector_model.pth'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train object detector model")
    parser.add_argument("excel_path", type=str, help="Path to the Excel file containing image paths and coordinates")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    
    args = parser.parse_args()
    
    main(args.excel_path, args.epochs, args.batch_size)
