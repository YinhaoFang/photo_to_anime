"""
preprocess.py  —  Anime Style Version
-------------------------------------
功能：
1. 统一 data/raw/photos 和 data/raw/anime 下的图片大小为 512x512
2. 为照片生成 Canny 边缘图（ControlNet 训练或推理条件）
3. 输出至 data/processed/photos, data/processed/anime, data/processed/edges
"""

import os
import cv2
from PIL import Image
from tqdm import tqdm

# ======== 基本路径配置 ========
RAW_PHOTO_DIR = "data/raw/photos"
RAW_ANIME_DIR = "data/raw/anime"
PROC_PHOTO_DIR = "data/processed/photos"
PROC_ANIME_DIR = "data/processed/anime"
PROC_EDGE_DIR = "data/processed/edges"
IMAGE_SIZE = 512  # 输出统一尺寸

# ======== 工具函数 ========
def ensure_dir(path: str):
    """确保路径存在"""
    if not os.path.exists(path):
        os.makedirs(path)

def resize_image(img_path: str, output_path: str, size=512, sharp=False):
    """使用 Pillow resize 并保存"""
    img = Image.open(img_path).convert("RGB")
    # 动漫图像锐度高，用 BICUBIC；照片可用 LANCZOS
    resample_method = Image.BICUBIC if sharp else Image.LANCZOS
    img = img.resize((size, size), resample_method)
    img.save(output_path)

def generate_edge(img_path: str, output_path: str, size=512):
    """生成 Canny 边缘图"""
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"[WARN] 无法读取图像：{img_path}")
        return
    img = cv2.resize(img, (size, size))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 阈值调整适合动漫线条
    edges = cv2.Canny(gray, 50, 150)
    cv2.imwrite(output_path, edges)

# ======== 主函数 ========
def main():
    print("=== [1/4] 创建输出目录 ===")
    for d in [PROC_PHOTO_DIR, PROC_ANIME_DIR, PROC_EDGE_DIR]:
        ensure_dir(d)

    print("=== [2/4] 处理照片 (photos) ===")
    photos = sorted([f for f in os.listdir(RAW_PHOTO_DIR)
                     if f.lower().endswith((".jpg", ".png", ".jpeg"))])
    for fname in tqdm(photos, desc="处理照片"):
        src = os.path.join(RAW_PHOTO_DIR, fname)
        dst_photo = os.path.join(PROC_PHOTO_DIR, fname)
        dst_edge = os.path.join(PROC_EDGE_DIR, fname)
        resize_image(src, dst_photo, IMAGE_SIZE, sharp=False)
        generate_edge(src, dst_edge, IMAGE_SIZE)

    print("=== [3/4] 处理动漫图片 (anime) ===")
    animes = sorted([f for f in os.listdir(RAW_ANIME_DIR)
                     if f.lower().endswith((".jpg", ".png", ".jpeg"))])
    for fname in tqdm(animes, desc="处理动漫图片"):
        src = os.path.join(RAW_ANIME_DIR, fname)
        dst_anime = os.path.join(PROC_ANIME_DIR, fname)
        resize_image(src, dst_anime, IMAGE_SIZE, sharp=True)

    print("=== [4/4] 汇总结果 ===")
    print(f"✅ 共处理照片 {len(photos)} 张，动漫图片 {len(animes)} 张。")
    print(f"输出路径：{PROC_PHOTO_DIR}, {PROC_ANIME_DIR}, {PROC_EDGE_DIR}")

if __name__ == "__main__":
    main()
