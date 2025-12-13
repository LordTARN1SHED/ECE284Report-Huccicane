# 16_gen_compound_data.py
import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import random

# ================= 配置 =================
SRC_DIR = Path('./data/gtsrb/GTSRB/Training')
DST_DIR = Path('./data/processed/Compound')
# =======================================

def apply_compound_distortion(image):
    """
    叠加顺序: Blur -> Fog -> Noise
    这是最符合物理逻辑且难度最高的顺序
    """
    img = image.astype(np.float32) / 255.0
    
    # 1. Blur (需要转uint8处理)
    temp_img = (img * 255).astype(np.uint8)
    degree = 10; angle = 45
    M = cv2.getRotationMatrix2D((degree/2, degree/2), angle, 1)
    k = np.diag(np.ones(degree)); k = cv2.warpAffine(k, M, (degree, degree)) / degree
    temp_img = cv2.filter2D(temp_img, -1, k)
    img = temp_img.astype(np.float32) / 255.0
    
    # 2. Fog
    intensity = 0.5; A = 0.9; t = 1.0 - intensity
    img = img * t + A * (1 - t)
    
    # 3. Noise
    noise = np.random.normal(0, 0.02 ** 0.5, img.shape)
    img = img + noise
    
    return np.clip(img * 255, 0, 255).astype(np.uint8)

def process():
    img_paths = list(SRC_DIR.glob('*/*.ppm'))
    print(f"开始生成全量混合畸变数据 (Blur+Fog+Noise)... 总数: {len(img_paths)}")

    for img_path in tqdm(img_paths):
        img = cv2.imread(str(img_path))
        if img is None: continue

        # 生成坏图
        bad_img = apply_compound_distortion(img)

        # 保持目录结构
        relative_path = img_path.relative_to(SRC_DIR)
        save_path = DST_DIR / relative_path
        
        # 兼容: 统一存为png，防止压缩损失
        save_path = save_path.with_suffix('.png')
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), bad_img)

    print(f"生成完成！请检查: {DST_DIR}")

if __name__ == '__main__':
    process()