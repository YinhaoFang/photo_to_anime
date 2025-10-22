# photo_to_anime

## 🎯 项目目标
将真实人物照片转换为动漫风格图像，实现现实到二次元的**风格迁移（Style Transfer）**。  
项目基于 **Stable Diffusion + ControlNet + LoRA**，利用现成的预训练模型*Animagine-XL-4.0*实现高质量动漫化生成。

---

## 👥 团队成员
- 组长：czh  
- 模型加载与优化组：fyh、姓名  
- 推理展示与交互组：czh、姓名  
- 风格评估与报告组：姓名、姓名  

---

## 🧩 数据策略
采用 **非成对（unpaired）** 数据策略：  
- 单独收集“动漫风格图像”与“真实照片”两类数据集；  
- 模型不依赖配对样本，而通过 ControlNet 或图像特征约束实现结构保持与风格转换。

---

## 🧠 技术路线

### 🔹 总体流程

输入：现实人物照片
↓
图像预处理（边缘/姿态提取）
↓
Stable Diffusion + ControlNet/LoRA 推理
↓
输出：动漫风格人物图像


### 🔹 技术要点
| 模块 | 方法 / 工具 | 功能 |
|------|---------------|------|
| 基础模型 | Animagine-XL-4.0（Stable Diffusion 系列） | 生成动漫风格图像 |
| 结构保持 | ControlNet | 保持姿态、轮廓、构图一致 |
| 风格微调 | LoRA | 增强特定动漫风格特征 |
| 推理接口 | Diffusers Pipeline + Torch | 实现本地加载与批量生成 |
| 展示交互 | Gradio / Streamlit | 用户界面输入 prompt、查看结果 |
| 性能评估 | SSIM / CLIPScore / 耗时统计 | 分析保真度与风格质量 |

---

## ⚙️ 项目结构

photo_to_anime/
├── data/ # 数据目录（raw / processed）
├── models/ # Animagine 模型文件与权重
├── outputs/ # 生成图像结果
├── logs/ # 日志记录
├── scripts/ # 核心脚本
├── requirements.txt # 依赖库
└── README.md # 项目说明

## 🧰 环境依赖

python >= 3.10
torch >= 2.0
diffusers
transformers
safetensors
gradio
opencv-python