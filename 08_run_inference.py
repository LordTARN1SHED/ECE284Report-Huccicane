# 08_run_inference.py
import torch
import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
from torchvision import transforms
from PIL import Image

# 引入你的模型定义 (复用 07 的类，或者直接复制过来)
# 这里假设你把 SimpleUNet 放在同一个文件里，或者直接粘贴在这里
import torch.nn as nn

# ==========================================
# 必须和 07_train_restoration.py 里的模型定义一模一样
class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())
        self.pool2 = nn.MaxPool2d(2, 2)
        self.bottleneck = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.Conv2d(256, 256, 3, padding=1), nn.ReLU())
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.final = nn.Conv2d(64, 3, 1)
        
    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        b = self.bottleneck(p2)
        d2 = self.up2(b)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.dec1(d1)
        out = self.final(d1)
        return out
# ==========================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLEAN_DIR = Path('./data/gtsrb/GTSRB/Training')

# 任务配置
TASKS = ['Noise', 'Blur', 'Fog'] 

def process_task(task_name):
    print(f"\n=== 开始处理任务: {task_name} ===")
    
    # 1. 路径设置
    distorted_dir = Path(f'./data/processed/{task_name}')
    restored_dir = Path(f'./data/restored/{task_name}') # 输出目录
    model_path = f'./restoration_{task_name.lower()}.pth'
    
    if not os.path.exists(model_path):
        print(f"警告: 找不到模型 {model_path}，跳过该任务。")
        return

    # 2. 加载模型
    model = SimpleUNet().to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 3. 准备数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # 统计指标
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0
    
    # 4. 遍历所有文件
    files = list(distorted_dir.glob('*/*.ppm')) + list(distorted_dir.glob('*/*.png'))
    
    for file_path in tqdm(files):
        # 读取坏图
        img_pil = Image.open(file_path).convert('RGB')
        input_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)
        
        # 推理 (Inference)
        with torch.no_grad():
            output_tensor = model(input_tensor)
            
        # 后处理：转回 0-255 的图片格式
        output_tensor = torch.clamp(output_tensor, 0, 1)
        output_img = output_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
        output_img = (output_img * 255).astype(np.uint8)
        # RGB -> BGR for OpenCV saving
        output_bgr = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
        
        # 保存图片 (保持原有目录结构，这对于ImageFolder很重要)
        rel_path = file_path.relative_to(distorted_dir)
        save_path = restored_dir / rel_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 注意：这里我们统一存为 png 以避免压缩损耗
        save_path = save_path.with_suffix('.png')
        cv2.imwrite(str(save_path), output_bgr)
        
        # --- 计算 PSNR/SSIM (为了 Proposal 中的 Image Quality Evaluation) ---
        # 必须加载对应的原图做对比
        clean_path = CLEAN_DIR / rel_path
        # 原图可能是ppm，兼容一下
        if not clean_path.exists(): clean_path = clean_path.with_suffix('.ppm')
        
        if clean_path.exists():
            clean_img = cv2.imread(str(clean_path))
            clean_img = cv2.resize(clean_img, (224, 224)) # 确保尺寸一致
            
            # 计算指标
            # PSNR
            psnr_val = psnr_metric(clean_img, output_bgr, data_range=255)
            # SSIM (需要灰度或者多通道设置)
            ssim_val = ssim_metric(clean_img, output_bgr, data_range=255, channel_axis=2)
            
            total_psnr += psnr_val
            total_ssim += ssim_val
            count += 1
            
    # 打印该任务的平均指标
    if count > 0:
        print(f"任务 [{task_name}] 完成。")
        print(f"平均 PSNR: {total_psnr / count:.2f} dB")
        print(f"平均 SSIM: {total_ssim / count:.4f}")
    else:
        print("未处理任何图片。")

if __name__ == '__main__':
    for task in TASKS:
        process_task(task)