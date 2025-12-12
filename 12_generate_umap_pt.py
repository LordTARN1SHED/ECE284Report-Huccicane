# 12_generate_umap_pt.py
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import random
from pathlib import Path
from tqdm import tqdm

# ================= 配置区域 =================
# 采样数量：每种模式取多少张图？(建议 50-100，太多图看不清，太少没代表性)
SAMPLES_PER_MODE = 100 

# 基础路径
CLEAN_DIR = Path('./data/gtsrb/GTSRB/Training')
# 定义 7 种模式及其路径
DATA_MODES = {
    'Clean':            CLEAN_DIR,
    'Noise (Bad)':      Path('./data/processed/Noise'),
    'Noise (Restored)': Path('./data/restored/Noise'),
    'Blur (Bad)':       Path('./data/processed/Blur'),
    'Blur (Restored)':  Path('./data/restored/Blur'),
    'Fog (Bad)':        Path('./data/processed/Fog'),
    'Fog (Restored)':   Path('./data/restored/Fog')
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_PT_PATH = 'umap_embeddings.pt'
OUTPUT_IMG_PATH = 'umap_visualization.png'

# ===========================================

def get_vgg_features(model, img_tensor):
    """
    提取 VGG features 模块的最终输出
    Output Shape: [1, 512, 7, 7]
    """
    with torch.no_grad():
        features = model.features(img_tensor)
    return features

def process_features_for_umap(feature_tensor):
    """
    关键步骤：维度变换
    Input:  [1, 512, 7, 7]
    Output: [1, 512] (numpy array)
    策略：Global Average Pooling (GAP)
    """
    # 1. 在 7x7 的空间维度上求平均 (dim=2, 3)
    # 结果变成 [1, 512]
    pooled = torch.mean(feature_tensor, dim=[2, 3])
    
    # 2. 转 Numpy
    return pooled.cpu().numpy()

def main():
    print("1. 加载 VGG16 模型...")
    # 我们只需要特征提取器，不需要分类器
    vgg = models.vgg16(weights='DEFAULT').to(DEVICE)
    vgg.eval()

    # 预处理 (必须和训练时一致)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 准备容器
    all_features = [] # 存放 X: (N_total, 512)
    all_labels = []   # 存放 y: (N_total,) -> 对应的模式名称

    # 2. 遍历 7 种模式提取特征
    # 为了保证对比公平，我们应该尽量取同名的图片
    # 先获取所有 Clean 图片的文件名列表
    all_files = list(CLEAN_DIR.glob('*/*.ppm'))
    if len(all_files) > SAMPLES_PER_MODE:
        selected_files = random.sample(all_files, SAMPLES_PER_MODE)
    else:
        selected_files = all_files

    print(f"每种模式采样 {len(selected_files)} 张图片，共 7 种模式...")

    for mode_name, dir_path in DATA_MODES.items():
        print(f"正在处理: {mode_name} ...")
        
        valid_count = 0
        for clean_path in tqdm(selected_files):
            # 构建相对路径以找到对应的 Bad/Restored 文件
            rel_path = clean_path.relative_to(CLEAN_DIR)
            
            # 确定当前模式下的文件路径
            if mode_name == 'Clean':
                target_path = clean_path
            else:
                target_path = dir_path / rel_path
                # 兼容 png
                if not target_path.exists():
                    target_path = target_path.with_suffix('.png')
            
            if not target_path.exists():
                continue

            # 读取与预处理
            try:
                img = Image.open(target_path).convert('RGB')
                tensor = transform(img).unsqueeze(0).to(DEVICE) # [1, 3, 224, 224]
                
                # 提取特征 [1, 512, 7, 7]
                raw_feat = get_vgg_features(vgg, tensor)
                
                # 压扁成 [1, 512]
                flat_feat = process_features_for_umap(raw_feat)
                
                all_features.append(flat_feat)
                all_labels.append(mode_name)
                valid_count += 1
            except Exception as e:
                print(f"Error processing {target_path}: {e}")

    # 3. 运行 UMAP
    print("\n正在运行 UMAP 降维 (这可能需要几秒钟)...")
    # Stack 成一个大矩阵: (Total_Samples, 512)
    X = np.vstack(all_features)
    
    # UMAP 配置
    reducer = umap.UMAP(
        n_neighbors=15,    # 邻居数量，越大越关注全局结构
        min_dist=0.1,      # 点之间的最小距离，越小聚类越紧密
        n_components=2,    # 降维到 2D
        metric='cosine',   # 使用余弦相似度 (在高维空间通常比欧氏距离好)
        random_state=42    # 固定随机种子
    )
    
    embedding = reducer.fit_transform(X) # Output: (Total_Samples, 2)
    
    print(f"降维完成! 输入形状: {X.shape}, 输出形状: {embedding.shape}")

    # 4. 保存为 .pt 文件 (应你要求)
    print(f"正在保存数据到 {OUTPUT_PT_PATH} ...")
    data_to_save = {
        'embeddings': torch.tensor(embedding), # UMAP 坐标 (x, y)
        'labels': all_labels,                  # 对应的类别名称列表
        'original_features': torch.tensor(X)   # 原始 512 维向量 (备份用)
    }
    torch.save(data_to_save, OUTPUT_PT_PATH)

    # 5. 画图可视化
    print(f"正在生成可视化图表 {OUTPUT_IMG_PATH} ...")
    plt.figure(figsize=(12, 10))
    
    # 使用 Seaborn 绘图，因为它处理字符串标签很方便
    sns.scatterplot(
        x=embedding[:, 0], 
        y=embedding[:, 1], 
        hue=all_labels, 
        palette="tab10", # 颜色盘
        s=60,            # 点的大小
        alpha=0.7        # 透明度
    )
    
    plt.title('UMAP Projection of VGG16 Features (Layer: features.30)', fontsize=15)
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) # 图例放外面
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG_PATH)
    plt.show()

    print("全部完成！")

if __name__ == '__main__':
    main()