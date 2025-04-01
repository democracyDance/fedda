#数据加载器
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
from torch.utils.data import Subset
import random
'''
def get_loader(data_dir, domain, batch_size):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    path = os.path.join(data_dir, domain)
    dataset = ImageFolder(root=path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return loader'''

# utils/data_loader.py

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os

def get_loader(data_dir, domain, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # ✅可选，建议加
    ])
    path = os.path.join(data_dir, domain)
    dataset = ImageFolder(root=path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)  # ✅ num_workers=0
    return loader

def split_noniid(dataset, num_clients, ratio=0.6):
    class_indices = {i: [] for i in range(len(dataset.classes))}
    for idx, (_, label) in enumerate(dataset.samples):
        class_indices[label].append(idx)

    client_indices = [[] for _ in range(num_clients)]

    # 每个类别 → 随机分给客户端
    for c, indices in class_indices.items():
        random.shuffle(indices)
        split = int(len(indices) * ratio)
        for i in range(num_clients):
            part = indices[i * split // num_clients : (i + 1) * split // num_clients]
            client_indices[i].extend(part)

        # 剩余数据均匀分配 (保证 IID 部分)
        remaining = indices[split:]
        for i, idx in enumerate(remaining):
            client_indices[i % num_clients].append(idx)

    return client_indices


