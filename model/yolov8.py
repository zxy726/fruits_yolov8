import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import random
import cv2
import numpy as np


# 1. 数据集加载器
class FruitDataset(Dataset):
    def __init__(self, root_dir, transform=None, label_dir=None):
        self.root_dir = root_dir
        self.transform = transform
        self.label_dir = label_dir  # 存放标签（边界框和类别）文件的路径
        self.classes = sorted(os.listdir(os.path.join(root_dir, 'train')))
        self.image_paths = []
        self.labels = []

        # 构建图片路径和标签
        for idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, 'train', class_name)
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    label_path = os.path.join(self.label_dir, class_name,
                                              img_name.replace('.jpg', '.txt'))  # 假设标签文件为txt
                    if os.path.exists(label_path):  # 确保每个图像都有对应的标签文件
                        self.image_paths.append(img_path)
                        self.labels.append(label_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_path = self.labels[idx]

        # 打开图像
        img = Image.open(img_path).convert('RGB')

        # 加载标签（边界框 + 类别）
        with open(label_path, 'r') as file:
            label_data = file.readlines()

        boxes = []
        labels = []
        for line in label_data:
            parts = line.strip().split()
            class_id = int(parts[0])  # 类别ID
            bbox = list(map(float, parts[1:]))  # 边界框 [x_min, y_min, x_max, y_max]
            boxes.append(bbox)
            labels.append(class_id)

        # 转换为张量形式
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        if self.transform:
            img = self.transform(img)

        return img, boxes, labels


# 2. 图像预处理（数据增强和标准化）
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomVerticalFlip(),  # 随机垂直翻转
    transforms.RandomRotation(30),  # 随机旋转，最大旋转30度
    transforms.Resize((416, 416)),  # 调整图片大小
    transforms.ToTensor(),  # 转换为Tensor并归一化到[0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])

# 数据集路径和标签路径
train_dataset = FruitDataset(
    root_dir='D:/A Learning courses/fruit_web/fruit_data',
    transform=transform,
    label_dir='D:/A Learning courses/fruit_web/fruit_labels'
)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)


# 3. 定义YOLOv8模型
class YOLOv8(nn.Module):
    def __init__(self, num_classes):
        super(YOLOv8, self).__init__()

        # 采用简化的卷积骨干网络
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Detection head
        self.detect_head = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, num_classes * 5, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.detect_head(x)
        return x


# 4. 定义损失函数
def yolo_loss(pred, target, num_classes):
    batch_size = pred.size(0)
    grid_size = pred.size(2)
    cell_size = 1 / grid_size
    pred = pred.view(batch_size, num_classes + 5, grid_size, grid_size)

    # 计算损失
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()

    loss = 0
    for i in range(batch_size):
        for j in range(grid_size):
            for k in range(grid_size):
                # 预测框
                pred_box = pred[i, :, j, k]
                target_box = target[i, :, j, k]

                # 计算边界框损失
                box_loss = mse_loss(pred_box[:4], target_box[:4])

                # 分类损失
                class_loss = bce_loss(pred_box[4:], target_box[4:])

                loss += box_loss + class_loss

    return loss


# 5. 训练过程
def train(model, train_loader, num_epochs=10, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, boxes, labels in train_loader:
            images = images.to(device)
            boxes = boxes.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = yolo_loss(outputs, boxes, num_classes=len(train_loader.dataset.classes))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}")


# 6. 保存模型
def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)


# 7. 初始化模型并开始训练
num_classes = len(train_loader.dataset.classes)
model = YOLOv8(num_classes)
train(model, train_loader, num_epochs=10, lr=1e-3)
save_model(model, 'fruit81_best.pt')

print("Training complete. Model saved as fruit81_best.pt.")
