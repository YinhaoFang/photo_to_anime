import os
import cv2
import numpy as np
import pandas as pd
import json
from PIL import Image, ImageFilter
import glob
import argparse
from pathlib import Path

class ImagePreprocessor:
    def __init__(self, target_size=(768, 1024)):
        """
        初始化预处理器
        
        Args:
            target_size: 目标尺寸 (width, height)，默认为768x1024 (3:4比例)
        """
        self.target_size = target_size
        self.width, self.height = target_size
        
    def create_output_dirs(self, output_dir):
        """创建输出目录结构"""
        dirs = {
            'images': os.path.join(output_dir, 'images'),
            'edges': os.path.join(output_dir, 'edges'),
            'poses': os.path.join(output_dir, 'poses'),
            'masks': os.path.join(output_dir, 'masks')
        }
        
        for dir_path in dirs.values():
            os.makedirs(dir_path, exist_ok=True)
            
        return dirs
    
    def crop_and_resize(self, image_path):
        """
        裁剪和调整图片尺寸到目标比例
        
        Args:
            image_path: 输入图片路径
            
        Returns:
            处理后的PIL图像对象
        """
        try:
            # 使用PIL打开图片
            img = Image.open(image_path)
            original_width, original_height = img.size
            
            # 计算目标宽高比 (3:4)
            target_ratio = self.width / self.height
            current_ratio = original_width / original_height
            
            if current_ratio > target_ratio:
                # 图片太宽，需要裁剪宽度
                new_width = int(original_height * target_ratio)
                left = (original_width - new_width) // 2
                top = 0
                right = left + new_width
                bottom = original_height
            else:
                # 图片太高，需要裁剪高度
                new_height = int(original_width / target_ratio)
                left = 0
                top = (original_height - new_height) // 2
                right = original_width
                bottom = top + new_height
            
            # 裁剪图片
            cropped_img = img.crop((left, top, right, bottom))
            
            # 调整到目标尺寸
            resized_img = cropped_img.resize(self.target_size, Image.Resampling.LANCZOS)
            
            return resized_img
            
        except Exception as e:
            print(f"裁剪图片 {image_path} 时出错: {e}")
            return None
    
    def generate_edges(self, image):
        """
        生成边缘图
        
        Args:
            image: PIL图像对象
            
        Returns:
            边缘图PIL图像对象
        """
        try:
            # 转换为OpenCV格式
            img_cv = np.array(image)
            
            # 转换为灰度图
            gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
            
            # 使用Canny边缘检测
            edges = cv2.Canny(gray, 50, 150)
            
            # 转换为PIL图像
            edges_pil = Image.fromarray(edges)
            
            return edges_pil
            
        except Exception as e:
            print(f"生成边缘图时出错: {e}")
            return None
    
    def generate_poses(self, image):
        """
        使用 MediaPipe Pose 生成姿态估计图
        """
        try:
            import mediapipe as mp
            import cv2

            mp_pose = mp.solutions.pose
            mp_drawing = mp.solutions.drawing_utils

            # 转换为 OpenCV 格式（BGR）
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # 使用 MediaPipe 进行姿态检测
            with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
                results = pose.process(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

            # 创建一张空白黑图
            pose_img = np.zeros_like(img_cv)

            # 如果检测到人体关键点，绘制骨架
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    pose_img,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                )

            # 转回 PIL 图像
            return Image.fromarray(cv2.cvtColor(pose_img, cv2.COLOR_BGR2RGB))

        except Exception as e:
            print(f"生成姿态图时出错: {e}")
            return None

    
    def generate_masks(self, image):
        """
        使用 rembg 生成前景掩码图（白色=人物，黑色=背景）
        """
        try:
            from rembg import remove
            import io

            # 将 PIL 图像保存为字节流
            img_bytes = io.BytesIO()
            image.save(img_bytes, format='PNG')
            img_bytes = img_bytes.getvalue()

            # 使用 rembg 去除背景
            result_bytes = remove(img_bytes)

            # 将结果转换回 PIL 图像
            result_image = Image.open(io.BytesIO(result_bytes)).convert('RGBA')

            # 提取 Alpha 通道作为掩码
            mask = result_image.getchannel('A').point(lambda p: 255 if p > 0 else 0).convert('L')

            return mask

        except Exception as e:
            print(f"生成掩码图时出错: {e}")
            return None

    
    def process_single_image(self, image_path, output_dirs, index):
        """
        处理单张图片
        
        Args:
            image_path: 输入图片路径
            output_dirs: 输出目录字典
            index: 图片索引
            
        Returns:
            处理结果字典或None（如果处理失败）
        """
        # 生成输出文件名
        filename = f"{index:06d}"
        
        # 裁剪和调整尺寸
        processed_img = self.crop_and_resize(image_path)
        if processed_img is None:
            return None
        
        # 保存处理后的图片
        img_output_path = os.path.join(output_dirs['images'], f"{filename}.png")
        processed_img.save(img_output_path, 'PNG')
        
        # 生成边缘图
        edges_img = self.generate_edges(processed_img)
        if edges_img:
            edges_output_path = os.path.join(output_dirs['edges'], f"{filename}.png")
            edges_img.save(edges_output_path, 'PNG')
        
        # 生成姿态图
        poses_img = self.generate_poses(processed_img)
        if poses_img:
            poses_output_path = os.path.join(output_dirs['poses'], f"{filename}.png")
            poses_img.save(poses_output_path, 'PNG')
        
        # 生成掩码图
        masks_img = self.generate_masks(processed_img)
        if masks_img:
            masks_output_path = os.path.join(output_dirs['masks'], f"{filename}.png")
            masks_img.save(masks_output_path, 'PNG')
        
        return {
            'index': index,
            'original_path': image_path,
            'image_path': img_output_path,
            'edges_path': os.path.join(output_dirs['edges'], f"{filename}.png") if edges_img else None,
            'poses_path': os.path.join(output_dirs['poses'], f"{filename}.png") if poses_img else None,
            'masks_path': os.path.join(output_dirs['masks'], f"{filename}.png") if masks_img else None,
            'width': self.width,
            'height': self.height
        }
    
    def process_folder(self, input_folder, output_folder):
        """
        处理整个文件夹
        
        Args:
            input_folder: 输入文件夹路径
            output_folder: 输出文件夹路径
            
        Returns:
            处理结果列表
        """
        # 创建输出目录
        output_dirs = self.create_output_dirs(output_folder)
        
        # 获取所有图片文件
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        image_files = []
        for extension in image_extensions:
            image_files.extend(glob.glob(os.path.join(input_folder, extension)))
        
        if not image_files:
            print("在指定文件夹中未找到图片文件！")
            return []
        
        print(f"找到 {len(image_files)} 张图片")
        
        # 处理所有图片
        results = []
        success_count = 0
        
        for i, image_path in enumerate(image_files):
            print(f"处理图片 {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
            
            result = self.process_single_image(image_path, output_dirs, i)
            if result:
                results.append(result)
                success_count += 1
                print(f"  ✓ 成功处理")
            else:
                print(f"  ✗ 处理失败")
        
        print(f"\n处理完成！成功处理 {success_count}/{len(image_files)} 张图片")
        
        return results
    
    def generate_metadata(self, results, output_folder, format='both'):
        """
        生成元数据文件
        
        Args:
            results: 处理结果列表
            output_folder: 输出文件夹路径
            format: 输出格式 ('csv', 'json', 'both')
        """
        if not results:
            print("没有处理结果，无法生成元数据")
            return
        
        # 转换为DataFrame
        df = pd.DataFrame(results)
        
        if format in ['csv', 'both']:
            csv_path = os.path.join(output_folder, 'metadata.csv')
            df.to_csv(csv_path, index=False)
            print(f"生成CSV元数据: {csv_path}")
        
        if format in ['json', 'both']:
            json_path = os.path.join(output_folder, 'metadata.json')
            
            # 转换为JSON格式
            json_data = {
                'total_images': len(results),
                'target_size': {'width': self.width, 'height': self.height},
                'images': results
            }
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            print(f"生成JSON元数据: {json_path}")
        
        # 生成简单的训练列表文件
        train_list_path = os.path.join(output_folder, 'train_list.txt')
        with open(train_list_path, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(f"{result['image_path']}\n")
        
        print(f"生成训练列表: {train_list_path}")

def main():
    print("=== 图片批量预处理工具 ===")
    input_folder = input("请输入输入图片文件夹路径: ").strip()
    output_folder = input("请输入输出文件夹路径（留空则为输入文件夹下的 preprocessed）: ").strip()

    if not output_folder:
        output_folder = os.path.join(input_folder, 'preprocessed')

    # 固定目标比例为 3:4（768x1024）
    target_size = (768, 1024)
    metadata_format = 'both'

    # 创建预处理器
    preprocessor = ImagePreprocessor(target_size=target_size)

    # 处理图片
    results = preprocessor.process_folder(input_folder, output_folder)

    # 生成元数据
    preprocessor.generate_metadata(results, output_folder, metadata_format)

    print(f"\n预处理完成！输出目录: {output_folder}")
    print("目录结构:")
    print(f"  {output_folder}/")
    print("  ├── images/     # 处理后的图片")
    print("  ├── edges/      # 边缘图")
    print("  ├── poses/      # 姿态图")
    print("  ├── masks/      # 掩码图")
    print("  ├── metadata.csv    # CSV元数据")
    print("  ├── metadata.json   # JSON元数据")
    print("  └── train_list.txt  # 训练列表")


if __name__ == "__main__":
    # 检查依赖
    try:
        import cv2
        import pandas as pd
        from PIL import Image
    except ImportError as e:
        print(f"缺少依赖库: {e}")
        print("请安装所需依赖: pip install opencv-python pandas Pillow")
        exit(1)
    
    main()
