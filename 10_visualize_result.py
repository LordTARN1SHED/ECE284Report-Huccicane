# 09_visualize_result.py
import matplotlib.pyplot as plt
import cv2
import os
import random
from pathlib import Path

# 配置
TASKS = ['Noise', 'Blur', 'Fog']
CLEAN_DIR = Path('./data/gtsrb/GTSRB/Training')

def show_comparison():
    # 随机选一张图 (比如找个特定文件夹)
    # 随便找个存在的路径
    all_files = list(CLEAN_DIR.glob('*/*.ppm'))
    if not all_files: return
    
    # 随机抽一张
    target_file = random.choice(all_files)
    rel_path = target_file.relative_to(CLEAN_DIR)
    
    print(f"正在可视化文件: {rel_path}")
    
    plt.figure(figsize=(15, 10))
    
    # 第一列：原图
    clean_img = cv2.imread(str(target_file))
    clean_img = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)
    
    plt.subplot(3, 3, 2)
    plt.title("Original Clean Image")
    plt.imshow(clean_img)
    plt.axis('off')
    
    # 遍历三个任务
    for idx, task in enumerate(TASKS):
        # 坏图路径
        bad_path = Path(f'./data/processed/{task}') / rel_path
        # 某些生成脚本可能存为png
        if not bad_path.exists(): bad_path = bad_path.with_suffix('.png')
        
        # 修复图路径
        restored_path = Path(f'./data/restored/{task}') / rel_path
        restored_path = restored_path.with_suffix('.png') # 修复脚本统一存png
        
        if bad_path.exists():
            bad_img = cv2.imread(str(bad_path))
            bad_img = cv2.cvtColor(bad_img, cv2.COLOR_BGR2RGB)
            
            plt.subplot(3, 3, 4 + idx) # 第二行
            plt.title(f"{task} (Distorted)")
            plt.imshow(bad_img)
            plt.axis('off')

        if restored_path.exists():
            res_img = cv2.imread(str(restored_path))
            res_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
            
            plt.subplot(3, 3, 7 + idx) # 第三行
            plt.title(f"{task} (Restored)")
            plt.imshow(res_img)
            plt.axis('off')
            
    plt.tight_layout()
    plt.savefig('result_visualization.png')
    print("对比图已保存为 result_visualization.png")
    plt.show()

if __name__ == '__main__':
    show_comparison()