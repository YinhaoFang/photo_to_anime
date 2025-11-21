#姿态一致性评估
import mediapipe as mp
import numpy as np
from scipy.spatial.distance import cosine

class PoseConsistencyEvaluator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
    
    def extract_keypoints(self, image):
        """提取人体关键点"""
        results = self.pose.process(image)
        if not results.pose_landmarks:
            return None
        
        keypoints = []
        for landmark in results.pose_landmarks.landmark:
            keypoints.append([landmark.x, landmark.y, landmark.visibility])
        
        return np.array(keypoints)
    
    def calculate_pose_similarity(self, original_kps, generated_kps):
        """计算姿态相似度"""
        if original_kps is None or generated_kps is None:
            return 0.0
        
        # 只使用可见的关键点
        visible_mask = (original_kps[:, 2] > 0.5) & (generated_kps[:, 2] > 0.5)
        if np.sum(visible_mask) < 5:  # 至少需要5个关键点
            return 0.0
        
        visible_original = original_kps[visible_mask, :2]
        visible_generated = generated_kps[visible_mask, :2]
        
        # 计算欧氏距离
        distances = np.linalg.norm(visible_original - visible_generated, axis=1)
        similarity = 1.0 / (1.0 + np.mean(distances))
        
        return similarity

# 图像质量与风格评估
import torch
import clip
from PIL import Image
import lpips

class StyleQualityEvaluator:
    def __init__(self, device='cuda'):
        self.device = device
        self.lpips_model = lpips.LPIPS(net='alex').to(device)
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
        
        # 国风风格关键词
        self.guofeng_keywords = [
            "Chinese style", "traditional Chinese painting", 
            "ink wash painting", "ancient Chinese costume",
            "hanfu", "classical elegance", "oriental aesthetic"
        ]
    
    def calculate_lpips(self, img1, img2):
        """计算感知相似度"""
        # 将图像转换为LPIPS需要的格式
        tensor1 = lpips.im2tensor(img1).to(self.device)
        tensor2 = lpips.im2tensor(img2).to(self.device)
        
        with torch.no_grad():
            distance = self.lpips_model(tensor1, tensor2)
        
        return distance.item()
    
    def calculate_clip_score(self, image, prompt):
        """计算CLIP文本-图像匹配度"""
        image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        text_input = clip.tokenize([prompt]).to(self.device)
        
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
            text_features = self.clip_model.encode_text(text_input)
            
            # 计算余弦相似度
            similarity = torch.cosine_similarity(image_features, text_features)
        
        return similarity.item()
    
    def calculate_style_fidelity(self, image):
        """计算国风风格符合度"""
        scores = []
        for keyword in self.guofeng_keywords:
            score = self.calculate_clip_score(image, keyword)
            scores.append(score)
        
        return np.mean(scores)
#美学评分
import tensorflow as tf

class AestheticEvaluator:
    def __init__(self):
        # 使用预训练的美学评分模型
        self.model = self.load_aesthetic_model()
    
    def load_aesthetic_model(self):
        """加载美学评分模型"""
        # 可以使用NIMA、AVA等预训练模型
        # 这里使用简化版本，实际项目中需要加载完整模型
        pass
    
    def predict_aesthetic_score(self, image):
        """预测美学评分"""
        # 实现美学评分预测
        # 返回0-10的评分
        processed_image = self.preprocess_image(image)
        score = self.model.predict(processed_image)
        return float(score[0])
    
    # evaluate.py
#完整代码
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import yaml
from datetime import datetime

class StyleEvaluationPipeline:
    def __init__(self, config_path=None):
        self.config = self.load_config(config_path)
        self.pose_evaluator = PoseConsistencyEvaluator()
        self.style_evaluator = StyleQualityEvaluator()
        self.aesthetic_evaluator = AestheticEvaluator()
        
        self.results = {}
    
    def load_config(self, config_path):
        """加载评估配置"""
        default_config = {
            "metrics": {
                "pose_consistency": True,
                "lpips": True,
                "clip_score": True,
                "style_fidelity": True,
                "aesthetic_score": True,
                "inference_time": True
            },
            "visualization": {
                "generate_charts": True,
                "output_format": "png",
                "dpi": 300
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f)
                default_config.update(user_config)
        
        return default_config
    
    def evaluate_experiment(self, experiment_dir):
        """评估单个实验"""
        exp_path = Path(experiment_dir)
        metadata_path = exp_path / "metadata.csv"
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata.csv not found in {experiment_dir}")
        
        # 读取元数据
        metadata = pd.read_csv(metadata_path)
        results = []
        
        for idx, row in metadata.iterrows():
            result = self.evaluate_single_pair(row, exp_path)
            results.append(result)
        
        # 转换为DataFrame
        results_df = pd.DataFrame(results)
        
        # 计算汇总统计
        summary = self.calculate_summary(results_df)
        
        return results_df, summary
    
    def evaluate_single_pair(self, row, exp_path):
        """评估单对图像"""
        original_img_path = exp_path / row['original_image_path']
        generated_img_path = exp_path / row['generated_image_path']
        
        # 加载图像
        original_img = self.load_image(original_img_path)
        generated_img = self.load_image(generated_img_path)
        
        result = {
            'original_image': str(original_img_path),
            'generated_image': str(generated_img_path),
            'prompt': row.get('prompt', ''),
            'seed': row.get('seed', ''),
        }
        
        # 计算各项指标
        if self.config['metrics']['pose_consistency']:
            pose_score = self.pose_evaluator.calculate_pose_similarity(
                original_img, generated_img
            )
            result['pose_consistency'] = pose_score
        
        if self.config['metrics']['lpips']:
            lpips_score = self.style_evaluator.calculate_lpips(
                original_img, generated_img
            )
            result['lpips'] = lpips_score
        
        if self.config['metrics']['clip_score'] and 'prompt' in row:
            clip_score = self.style_evaluator.calculate_clip_score(
                generated_img, row['prompt']
            )
            result['clip_score'] = clip_score
        
        if self.config['metrics']['style_fidelity']:
            style_score = self.style_evaluator.calculate_style_fidelity(generated_img)
            result['style_fidelity'] = style_score
        
        if self.config['metrics']['aesthetic_score']:
            aesthetic_score = self.aesthetic_evaluator.predict_aesthetic_score(generated_img)
            result['aesthetic_score'] = aesthetic_score
        
        if self.config['metrics']['inference_time'] and 'inference_time' in row:
            result['inference_time'] = row['inference_time']
        
        return result
    
    def calculate_summary(self, results_df):
        """计算汇总统计"""
        summary = {}
        numeric_columns = results_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            summary[f'{col}_mean'] = results_df[col].mean()
            summary[f'{col}_std'] = results_df[col].std()
            summary[f'{col}_min'] = results_df[col].min()
            summary[f'{col}_max'] = results_df[col].max()
            summary[f'{col}_median'] = results_df[col].median()
        
        return summary
    
    def generate_visualizations(self, results_df, summary, output_dir):
        """生成可视化图表"""
        if not self.config['visualization']['generate_charts']:
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. 箱型图 - 各指标分布
        self.create_boxplots(results_df, output_path)
        
        # 2. 散点图 - 指标相关性
        self.create_correlation_plot(results_df, output_path)
        
        # 3. 雷达图 - 实验对比
        self.create_radar_chart(summary, output_path)
        
        # 4. 示例对比图
        self.create_example_comparisons(results_df, output_path)
    
    def create_boxplots(self, results_df, output_path):
        """创建箱型图"""
        numeric_cols = results_df.select_dtypes(include=[np.number]).columns
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, col in enumerate(numeric_cols[:6]):  # 最多显示6个指标
            if i < len(axes):
                results_df[col].plot.box(ax=axes[i])
                axes[i].set_title(f'Distribution of {col}')
                axes[i].set_ylabel('Score')
        
        # 隐藏多余的子图
        for i in range(len(numeric_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_path / 'metrics_distribution.png', 
                   dpi=self.config['visualization']['dpi'])
        plt.close()
    
    def create_correlation_plot(self, results_df, output_path):
        """创建相关性热力图"""
        numeric_df = results_df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) > 1:
            plt.figure(figsize=(10, 8))
            correlation_matrix = numeric_df.corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
                       center=0, fmt='.2f')
            plt.title('Metrics Correlation Heatmap')
            plt.tight_layout()
            plt.savefig(output_path / 'correlation_heatmap.png',
                       dpi=self.config['visualization']['dpi'])
            plt.close()
