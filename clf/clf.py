import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse

# 

class ImageClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super(ImageClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)  # 0 - 10
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def evaluate(
    model: nn.Module, test_loader: torch.utils.data.DataLoader, device: torch.device
) -> Tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy, correct


def main(args):
    device = "cpu"
    # if torch.backends.mps.is_available():
    #     device = "mps"
    # elif torch.cuda.is_available():
    #     device = "cuda"

    print(f"Using device: {device}")

    # 数据预处理
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # 加载自定义数据集
    full_dataset = datasets.ImageFolder(root=args.data_dir, transform=transform)

    # 分割数据集
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    num_classes = len(full_dataset.classes)
    model = ImageClassifier(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        accuracy, correct = evaluate(model, test_loader, device)
        print(
            f"Epoch {epoch+1}/{args.epochs}, Loss: {train_loss:.4f}, Accuracy: {accuracy:.4f}"
        )

    # 保存模型
    torch.save(model.state_dict(), args.output)
    print(f"Model saved as {args.output}")

    # 保存类别信息
    with open(args.classes_file, "w") as f:
        for class_name in full_dataset.classes:
            f.write(f"{class_name}\n")
    print(f"Classes saved to {args.classes_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train image classifier on custom dataset"
    )
    parser.add_argument("data_dir", type=str, help="Path to the data directory")
    parser.add_argument(
        "--output",
        type=str,
        default="custom_model.pth",
        help="Path to save the trained model",
    )
    parser.add_argument(
        "--classes_file",
        type=str,
        default="classes.txt",
        help="Path to save the classes file",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train"
    )
    args = parser.parse_args()

    main(args)
