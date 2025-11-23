import os
import sys
import time
import json
import yaml
import torch
import clip
import lpips
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# -----------------------------------------------------------------------------
# 1. 姿态一致性评估模块 (Pose Consistency)
# -----------------------------------------------------------------------------
class PoseConsistencyEvaluator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        # static_image_mode=True 适合单张图片评估
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )

    def extract_keypoints(self, image_pil):
        """提取人体关键点，返回 (33, 3) 的 numpy 数组 [x, y, visibility]"""
        # MediaPipe 需要 RGB numpy 数组
        image_np = np.array(image_pil.convert('RGB'))
        results = self.pose.process(image_np)
        
        if not results.pose_landmarks:
            return None
        
        keypoints = []
        for landmark in results.pose_landmarks.landmark:
            keypoints.append([landmark.x, landmark.y, landmark.visibility])
        
        return np.array(keypoints)

    def calculate_pose_metrics(self, original_img, generated_img):
        """
        计算姿态一致性指标
        返回: {'pose_mse': float, 'pose_similarity': float}
        """
        kps1 = self.extract_keypoints(original_img)
        kps2 = self.extract_keypoints(generated_img)
        
        # 如果任意一张图检测不到姿态，返回 None 或 默认最差值
        if kps1 is None or kps2 is None:
            return {'pose_mse': np.nan, 'pose_similarity': 0.0}
        
        # 筛选出在两张图中可见度都大于阈值的点
        threshold = 0.5
        mask = (kps1[:, 2] > threshold) & (kps2[:, 2] > threshold)
        
        # 如果有效匹配点太少（例如少于5个），认为姿态不可比
        if np.sum(mask) < 5:
            return {'pose_mse': np.nan, 'pose_similarity': 0.0}
        
        valid_kps1 = kps1[mask, :2] # 取 x, y
        valid_kps2 = kps2[mask, :2]
        
        # 1. MSE (Mean Squared Error) - 越低越好
        # MediaPipe 输出已经是归一化坐标 (0-1)，直接计算 MSE
        mse = np.mean(np.sum((valid_kps1 - valid_kps2)**2, axis=1))
        
        # 2. Cosine Similarity (基于向量方向，可选) - 越高越好
        # 这里简单使用基于距离的相似度公式
        distances = np.linalg.norm(valid_kps1 - valid_kps2, axis=1)
        similarity = 1.0 / (1.0 + np.mean(distances) * 10) # *10 是为了拉伸分布
        
        return {'pose_mse': mse, 'pose_similarity': similarity}

# -----------------------------------------------------------------------------
# 2. 图像质量与风格评估模块 (Style & Quality)
# -----------------------------------------------------------------------------
class StyleQualityEvaluator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"Loading models on {self.device}...")
        
        # LPIPS (感知相似度) - 越低越好 (需要两张图)
        self.lpips_model = lpips.LPIPS(net='alex').to(device)
        
        # CLIP (文本-图像匹配度) - 越高越好
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
        
        # 国风风格关键词 (用于 Style Fidelity)
        self.guofeng_keywords = [
            "Chinese traditional painting style", 
            "ink wash painting", 
            "elegant oriental aesthetic",
            "wuxia style"
        ]

    def calculate_lpips(self, img1_pil, img2_pil):
        """计算感知差异 (Perceptual Distance)"""
        # LPIPS 需要 [-1, 1] 的 tensor
        try:
            img1_tensor = lpips.im2tensor(lpips.load_image(img1_pil)).to(self.device)
            img2_tensor = lpips.im2tensor(lpips.load_image(img2_pil)).to(self.device)
            
            with torch.no_grad():
                dist = self.lpips_model(img1_tensor, img2_tensor)
            return dist.item()
        except Exception as e:
            print(f"LPIPS calculation failed: {e}")
            return np.nan

    def calculate_clip_score(self, image_pil, prompt):
        """计算 CLIP Score (图像与 Prompt 的余弦相似度)"""
        # 截断 Prompt 以防过长
        text_token = clip.tokenize([prompt[:77]]).to(self.device)
        image_tensor = self.clip_preprocess(image_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_tensor)
            text_features = self.clip_model.encode_text(text_token)
            
            # 归一化
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            similarity = (image_features @ text_features.T).item()
            
        return similarity

    def calculate_style_fidelity(self, image_pil):
        """计算图像与预设风格关键词的平均匹配度"""
        scores = []
        for kw in self.guofeng_keywords:
            scores.append(self.calculate_clip_score(image_pil, kw))
        return np.mean(scores)

# -----------------------------------------------------------------------------
# 3. 美学评估模块 (Aesthetic - Optional)
# -----------------------------------------------------------------------------
class AestheticEvaluator:
    def __init__(self, device='cpu'):
        # 这里使用简单的占位符，如果需要真实评分，请加载 LAION-Aesthetic Predictor 等模型
        self.model_loaded = False
        # 示例：如果本地有模型权重，可以加载
        # self.model = load_aesthetic_model("path/to/sac+logos+ava1-l14-linearMSE.pth")
        pass

    def predict_score(self, image_pil):
        if not self.model_loaded:
            # 返回 -1 表示未启用，或者实现一个简单的亮度/对比度启发式算法
            return -1.0 
        # return self.model(image)
        return 5.0

# -----------------------------------------------------------------------------
# 4. 主评估管道 (Main Pipeline)
# -----------------------------------------------------------------------------
class EvaluationPipeline:
    def __init__(self, exp_dir, output_dir="eval"):
        self.exp_dir = Path(exp_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata_path = self.exp_dir / "metadata.csv"
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found at {self.metadata_path}")
            
        # 初始化评估器
        self.pose_eval = PoseConsistencyEvaluator()
        self.style_eval = StyleQualityEvaluator()
        self.aes_eval = AestheticEvaluator()

    def run_evaluation(self):
        """执行所有评估"""
        print(f"Starting evaluation for: {self.exp_dir.name}")
        df = pd.read_csv(self.metadata_path)
        results = []
        
        # 遍历 metadata 中的每一行
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
            # 构建完整路径 (假设 metadata 中存储的是相对路径)
            orig_path = self.exp_dir / row['original_image_path']
            gen_path = self.exp_dir / row['generated_image_path']
            
            if not orig_path.exists() or not gen_path.exists():
                print(f"Warning: Image missing at index {idx}")
                continue
                
            try:
                orig_img = Image.open(orig_path).convert('RGB')
                gen_img = Image.open(gen_path).convert('RGB')
            except Exception as e:
                print(f"Error opening image: {e}")
                continue

            # 1. 基础信息
            res_entry = {
                'filename': gen_path.name,
                'original_image': row['original_image_path'], # 用于分组
                'prompt': row.get('prompt', ''),
                'seed': row.get('seed', 0),
                'inference_time': row.get('inference_time', np.nan)
            }
            
            # 2. 计算 Pose Consistency (MSE)
            pose_metrics = self.pose_eval.calculate_pose_metrics(orig_img, gen_img)
            res_entry.update(pose_metrics)
            
            # 3. 计算 LPIPS (原图 vs 生成图，衡量结构/感知差异)
            # 注意：在风格迁移任务中，LPIPS 不一定越低越好，因为它包含颜色差异
            # 但可以作为结构保留的参考
            res_entry['lpips'] = self.style_eval.calculate_lpips(orig_img, gen_img)
            
            # 4. 计算 CLIP Score (Prompt Match)
            if 'prompt' in row:
                res_entry['clip_score'] = self.style_eval.calculate_clip_score(gen_img, row['prompt'])
            
            # 5. 计算 Style Fidelity (风格一致性)
            res_entry['style_fidelity'] = self.style_eval.calculate_style_fidelity(gen_img)
            
            results.append(res_entry)
            
        # 保存详细结果 CSV
        self.results_df = pd.DataFrame(results)
        csv_path = self.output_dir / f"results_{self.exp_dir.name}.csv"
        self.results_df.to_csv(csv_path, index=False)
        print(f"Detailed results saved to {csv_path}")
        
        return self.results_df

    def generate_report(self):
        """生成统计摘要和 Markdown 报告"""
        if not hasattr(self, 'results_df') or self.results_df.empty:
            print("No results to report.")
            return

        df = self.results_df
        
        # --- 统计逻辑 ---
        # 1. 全局平均值
        numeric_cols = ['pose_mse', 'lpips', 'clip_score', 'style_fidelity', 'inference_time']
        # 过滤掉 NaN 进行统计
        summary_stats = df[numeric_cols].mean().to_dict()
        summary_std = df[numeric_cols].std().to_dict()
        
        # 2. 按原图分组统计 (One-to-Many aggregation)
        # 计算每张原图产生的一组生成图中的 "最佳 CLIP Score" 和 "平均 Pose MSE"
        grouped = df.groupby('original_image')
        group_stats = pd.DataFrame({
            'mean_clip': grouped['clip_score'].mean(),
            'max_clip': grouped['clip_score'].max(),
            'mean_pose_mse': grouped['pose_mse'].mean(),
            'min_pose_mse': grouped['pose_mse'].min(), # MSE 越小越好
            'count': grouped.size()
        })
        
        # --- 可视化 ---
        self._plot_boxplots(df, numeric_cols)
        self._plot_radar_chart(summary_stats)
        self._plot_correlation(df, numeric_cols)
        
        # --- 生成 Markdown ---
        report_path = self.output_dir / f"summary_{self.exp_dir.name}.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# Experiment Report: {self.exp_dir.name}\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            
            f.write("## 1. Executive Summary\n")
            f.write("| Metric | Mean | Std Dev | Interpretation |\n")
            f.write("| :--- | :--- | :--- | :--- |\n")
            f.write(f"| **Pose MSE** | {summary_stats.get('pose_mse', 0):.4f} | {summary_std.get('pose_mse', 0):.4f} | Lower is better |\n")
            f.write(f"| **LPIPS** | {summary_stats.get('lpips', 0):.4f} | {summary_std.get('lpips', 0):.4f} | Lower = closer to source structure |\n")
            f.write(f"| **CLIP Score** | {summary_stats.get('clip_score', 0):.4f} | {summary_std.get('clip_score', 0):.4f} | Higher is better |\n")
            f.write(f"| **Style Fidelity** | {summary_stats.get('style_fidelity', 0):.4f} | {summary_std.get('style_fidelity', 0):.4f} | Match to 'Guofeng' keywords |\n")
            f.write(f"| **Inference Time** | {summary_stats.get('inference_time', 0):.4f}s | - | Per image |\n\n")
            
            f.write("## 2. Data Insights\n")
            f.write(f"- Total Images Evaluated: {len(df)}\n")
            f.write(f"- Unique Source Images: {len(group_stats)}\n")
            f.write(f"- Images with failed Pose Detect: {df['pose_mse'].isna().sum()}\n\n")
            
            f.write("## 3. Visualizations\n")
            f.write("### Metric Distributions\n")
            f.write(f"![Boxplots](charts/boxplots_{self.exp_dir.name}.png)\n")
            f.write("### Metric Correlation\n")
            f.write(f"![Correlation](charts/correlation_{self.exp_dir.name}.png)\n")
            
            f.write("## 4. Best & Worst Examples\n")
            # 找出 CLIP Score 最高和最低的
            best_clip = df.loc[df['clip_score'].idxmax()]
            worst_clip = df.loc[df['clip_score'].idxmin()]
            
            f.write("### Best CLIP Score (Text Alignment)\n")
            f.write(f"- **Score:** {best_clip['clip_score']:.4f}\n")
            f.write(f"- **Prompt:** {best_clip['prompt']}\n")
            f.write(f"- **File:** `{best_clip['filename']}`\n\n")
            
            f.write("### Best Pose Consistency (Lowest MSE)\n")
            # 注意处理全 NaN 的情况
            if df['pose_mse'].notna().any():
                best_pose = df.loc[df['pose_mse'].idxmin()]
                f.write(f"- **MSE:** {best_pose['pose_mse']:.5f}\n")
                f.write(f"- **File:** `{best_pose['filename']}`\n")
            else:
                f.write("No valid pose data detected.\n")

        print(f"Report generated at {report_path}")

    # --- Visualization Helpers ---
    def _plot_boxplots(self, df, cols):
        chart_dir = self.output_dir / "charts"
        chart_dir.mkdir(exist_ok=True)
        
        valid_cols = [c for c in cols if c in df.columns]
        if not valid_cols: return

        plt.figure(figsize=(15, 5))
        # Melt data for seaborn
        melted = df[valid_cols].melt(var_name='Metric', value_name='Score')
        sns.boxplot(data=melted, x='Metric', y='Score')
        plt.title("Distribution of Evaluation Metrics")
        plt.savefig(chart_dir / f"boxplots_{self.exp_dir.name}.png", dpi=150)
        plt.close()

    def _plot_correlation(self, df, cols):
        chart_dir = self.output_dir / "charts"
        valid_cols = [c for c in cols if c in df.columns]
        if len(valid_cols) < 2: return
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[valid_cols].corr(), annot=True, cmap='coolwarm', center=0)
        plt.title("Metric Correlation Heatmap")
        plt.savefig(chart_dir / f"correlation_{self.exp_dir.name}.png", dpi=150)
        plt.close()

    def _plot_radar_chart(self, summary_stats):
        # 简单的雷达图实现
        chart_dir = self.output_dir / "charts"
        
        # 标准化数据以便在雷达图中显示 (这步比较粗糙，实际需归一化)
        labels = ['clip_score', 'style_fidelity', 'pose_similarity'] # 选择 0-1 范围的指标
        values = [summary_stats.get(l, 0) for l in labels]
        
        if not values: return

        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.fill(angles, values, color='blue', alpha=0.25)
        ax.plot(angles, values, color='blue', linewidth=2)
        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        
        plt.title("Average Performance Radar")
        plt.savefig(chart_dir / f"radar_{self.exp_dir.name}.png", dpi=150)
        plt.close()

# -----------------------------------------------------------------------------
# 5. 入口函数
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Painting Evaluation Pipeline")
    parser.add_argument("--exp_dir", type=str, required=True, help="Path to experiment directory containing metadata.csv")
    parser.add_argument("--output_dir", type=str, default="eval_results", help="Directory to save reports")
    
    args = parser.parse_args()
    
    pipeline = EvaluationPipeline(args.exp_dir, args.output_dir)
    
    # 1. 计算指标
    pipeline.run_evaluation()
    
    # 2. 生成可视化报告
    pipeline.generate_report()
