# 18_test_unified_benchmark.py
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

# ================= 配置 =================
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = './vgg16_baseline.pth'

# 测试目标文件夹
TEST_DIRS = {
    "Clean (Baseline)":      './data/gtsrb/GTSRB/Training',
    "Compound Distorted":    './data/processed/Compound',
    "Unified Restored":      './data/restored/Compound'
}
# =======================================

def evaluate_model(model, data_dir, name):
    if not os.path.exists(data_dir):
        print(f"跳过 {name}: 路径不存在 {data_dir}")
        return None

    # VGG 标准预处理
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集
    dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    correct = 0; total = 0
    model.eval()
    
    print(f"正在测试: {name} (共 {len(dataset)} 张)...")
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct / total
    print(f"-> {name} 准确率: {acc*100:.2f}%")
    return acc

def main():
    # 1. 加载 VGG
    print("加载 VGG16 裁判模型...")
    model = models.vgg16(weights='DEFAULT')
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 43)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)

    # 2. 依次测试
    results = {}
    print("\n=== 开始最终 Benchmark ===")
    for name, path in TEST_DIRS.items():
        acc = evaluate_model(model, path, name)
        if acc is not None: results[name] = acc

    # 3. 打印报表
    print("\n" + "="*45)
    print("FINAL UNIFIED MODEL REPORT")
    print("="*45)
    print(f"{'Dataset Condition':<25} | {'Accuracy':<10}")
    print("-" * 45)
    for name, acc in results.items():
        print(f"{name:<25} | {acc*100:.2f}%")
    print("="*45)

if __name__ == '__main__':
    main()