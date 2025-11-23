#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate.py

通用的 Stable Diffusion / XL 本地模型推理脚本，适配项目的 models/ 目录结构。
特性：
- 支持从本地 models/ 目录加载（兼容 diffusers 格式）
- 支持单 prompt 或 prompt 列表文件批量生成
- 支持设置种子、输出尺寸、步数、guidance scale 等常用参数
- 自动在 CUDA/CPU 上选择 dtype（在 GPU 上使用 float16）并开启内存优化
- 可选启用 xformers（若已安装）

后续需要的操作:
请修改load_pipeline函数调用时的参数lora_weight以及control_weight为训练出的数据

用法示例：
    python scripts/generate.py --model-dir /home/huimolrb/photo_to_anime/models/models --prompt "a girl in hanfu, anime style" --out outputs/
    python scripts/generate.py --model-dir /home/huimolrb/photo_to_anime/models/models --prompt "the man in graph stands on the gr" --input-image /home/huimolrb/photo_to_anime/input_photos/6D08C07EB92CD0B6540C5B0819FABABF.png  --out outputs/singing_man.png --device cuda --width 1024 --height 1024 --steps 28 --guidance 7.5

    python scripts/generate.py --model-dir models --prompt "a girl in hanfu, anime style" --out outputs/test.png
    python scripts/generate.py --model-dir /home/huimolrb/photo_to_anime/models/models --prompts-file prompts.txt --num-per-prompt 2 --out outputs/
//--model-dir后的路径建议使用绝对路径(形如/home/usr/photo_to_anime/models，要求model_index.json包含在绝对路径下!

依赖：
    torch, diffusers, transformers, accelerate, safetensors, Pillow

作者：你的项目组
"""

import argparse
import logging
from pathlib import Path
import sys
import time
import os

import torch
from PIL import Image
import numpy as np  
import cv2  

from diffusers import DiffusionPipeline

# 其余代码...



LOG = logging.getLogger("generate")


def setup_logger(logfile: Path | None = None):
    LOG.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    LOG.addHandler(sh)
    if logfile:
        # 确保日志目录存在（FileHandler 不会自动创建父目录）
        logfile.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(logfile), encoding="utf-8")
        fh.setFormatter(fmt)
        LOG.addHandler(fh)

def get_images_from_folder(folder_path: Path):
    """从指定文件夹中提取所有支持的图像文件"""
    valid_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"]
    image_paths = []
    
    for filename in os.listdir(folder_path):
        if any(filename.lower().endswith(ext) for ext in valid_extensions):
            image_paths.append(folder_path / filename)
    
    return image_paths
    
def load_pipeline(model_dir: Path, device: str, torch_dtype: torch.dtype, lora_weight: Path | None = None, controlnet_weight: Path | None = None):
    """尝试从本地模型目录加载 diffusers pipeline。

    model_dir: 包含 model_index.json, unet/, vae/, text_encoder/ 等的目录
    device: 'cuda' 或 'cpu'
    torch_dtype: torch.float16 或 torch.float32
    """
    # 确保用户传入的路径指向存在的本地目录。如果用户从 scripts/ 下运行并传入相对路径 'models'，
    # 这会被解析为 scripts/models，这通常不是项目根的 models 目录。下面尝试几种解析方式。
    LOG.info("尝试解析模型目录: %s", model_dir)

    if not model_dir.exists():
        # 尝试以脚本所在项目根为基准解析（项目根假定为脚本的父目录的父目录）
        possible = Path(__file__).resolve().parent.parent / model_dir
        if possible.exists():
            model_dir = possible
            LOG.info("模型目录在项目根下找到，使用: %s", model_dir)
        else:
            # 最后尝试把传入 path 转为绝对路径
            alt = Path(model_dir).expanduser().resolve()
            if alt.exists():
                model_dir = alt
                LOG.info("模型目录解析为绝对路径: %s", model_dir)
            else:
                LOG.error("无法找到模型目录。请确保 --model-dir 指向存在的本地目录（如项目根下的 models/）。尝试的路径: %s, %s, %s", model_dir, possible, alt)
                raise FileNotFoundError(f"模型目录不存在: {model_dir}")

    LOG.info("从本地加载模型: %s", model_dir)
    try:
        # 优先尝试通过 DiffusionPipeline 自动加载本地 diffusers 格式模型
        pipe = DiffusionPipeline.from_pretrained(
            str(model_dir),
            torch_dtype=torch_dtype,
            local_files_only=True,
        )
    except Exception as e:
        LOG.warning("直接加载失败，尝试更宽松的加载方式：%s", e)
        # 退而求其次：允许网络下载（如果环境允许），以获取缺失的元数据或转换
        try:
            pipe = DiffusionPipeline.from_pretrained(str(model_dir), local_files_only=False)
        except Exception as e2:
            LOG.error("即使允许在线下载也无法加载模型：%s", e2)
            raise
    
    if lora_weight and lora_weight.exists():
        pipe.load_lora_weights(str(lora_weight))  # 加载 LoRA 权重
        LOG.info(f"加载 LoRA 权重: {lora_weight}")
    else:
        LOG.info("未提供 LoRA 权重，跳过加载")

    # 如果 controlnet_weight 存在，则加载 ControlNet 权重
    if controlnet_weight and controlnet_weight.exists():
        pipe.load_controlnet_weights(str(controlnet_weight))  # 加载 ControlNet 权重
        LOG.info(f"加载 ControlNet 权重: {controlnet_weight}")
    else:
        LOG.info("未提供 ControlNet 权重，跳过加载")
    
    # 把 pipeline 放到设备上
    pipe.to(device)

    # 关闭安全检查器（可选），避免某些模型带来阻塞；若需要请注释下面三行
    if hasattr(pipe, "safety_checker"):
        try:
            pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
            LOG.info("已禁用 safety_checker（如需开启，请在脚本中注释相关代码）")
        except Exception:
            pass

    # 内存和速度优化
    try:
        pipe.enable_attention_slicing()
        LOG.info("已启用 attention slicing")
    except Exception:
        pass

    try:
        pipe.enable_vae_slicing()
        LOG.info("已启用 VAE slicing")
    except Exception:
        pass

    # 尝试启用 xformers（如果安装了）以节省显存
    if device.startswith("cuda"):
        try:
            pipe.enable_xformers_memory_efficient_attention()
            LOG.info("已启用 xformers memory efficient attention")
        except Exception:
            LOG.info("xformers 未启用（未安装或不兼容）")

    return pipe


def read_prompts(prompts_arg: str | None, prompts_file: Path | None):
    prompts = []
    if prompts_arg:
        prompts.append(prompts_arg)
    if prompts_file:
        with open(prompts_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                prompts.append(line)
    return prompts

def process_image(input_image: Path, size: int = 512):
    """处理输入的图片，调整大小并生成边缘图（适用于 ControlNet）"""
    image = Image.open(input_image).convert("RGB")
    image = image.resize((size, size), Image.LANCZOS)
    
    # 生成 Canny 边缘图
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_image = Image.fromarray(edges)
    
    return image, edge_image

def main():
    parser = argparse.ArgumentParser(description="Generate images with a local diffusers model")
    parser.add_argument("--model-dir", type=Path, default=Path("models"), help="本地模型目录（含 model_index.json）")
    parser.add_argument("--out", type=Path, required=True, help="输出文件或输出目录")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompt", type=str, help="单条 prompt")
    group.add_argument("--prompts-file", type=Path, help="每行一个 prompt 的文本文件")

    parser.add_argument("--input-image", type=Path, help="输入图像路径（用于图像修改或生成）")
    parser.add_argument("--num-per-prompt", type=int, default=1, help="每个 prompt 生成多少张图")
    parser.add_argument("--seed", type=int, default=None, help="随机种子，默认随机")
    parser.add_argument("--width", type=int, default=1024, help="输出宽度（部分模型可能仅支持 512/1024 等预设）")
    parser.add_argument("--height", type=int, default=1024, help="输出高度")
    parser.add_argument("--steps", type=int, default=28, help="推理步数")
    parser.add_argument("--guidance", type=float, default=7.5, help="classifier-free guidance scale")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", choices=["cuda", "cpu"], help="运行设备")
    parser.add_argument("--log", type=Path, default=Path("logs/generate.log"), help="日志文件")
    parser.add_argument("--batch-size", type=int, default=1, help="管道一次处理多少张（若模型支持）")
    parser.add_argument("--negative-prompt", type=str, default=None, help="negative prompt")

    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True) if args.out.suffix else args.out.mkdir(parents=True, exist_ok=True)
    setup_logger(args.log)

    LOG.info("参数: %s", args)

    # 选择 dtype
    if args.device == "cuda":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # 加载 pipeline
    pipe = load_pipeline(args.model_dir, args.device, torch_dtype, lora_weight=None, controlnet_weight=None)


    # 读取 prompt(s)
    prompts = read_prompts(args.prompt, args.prompts_file)
    if not prompts:
        LOG.error("没有找到任何 prompts")
        return

     # 处理输入图像（如果提供了输入图像）
    input_image = None
    if args.input_image:
        input_image, edge_image = process_image(args.input_image, size=args.width)
        LOG.info("已处理输入图像：%s", args.input_image)

    # 随机种子与 generator
    if args.seed is None:
        args.seed = int(time.time() % (2 ** 31 - 1))

    LOG.info("使用种子: %d", args.seed)

    device_for_gen = "cuda" if args.device == "cuda" else "cpu"
    gen = torch.Generator(device_for_gen)
    gen.manual_seed(args.seed)

    out_is_file = args.out.suffix in [".png", ".jpg", ".jpeg"]

    
    total = 0
    for i, prompt in enumerate(prompts):
        for k in range(args.num_per_prompt):
            LOG.info("生成：prompt %d/%d - %d/%d", i + 1, len(prompts), k + 1, args.num_per_prompt)

            # 使用输入图像（如果有）与 prompt 一起生成图像
            try:
                if input_image:
                    # 如果提供了输入图像，可以在 pipeline 调用中传入它
                    result = pipe(
                        prompt=prompt,
                        negative_prompt=args.negative_prompt,
                        height=args.height,
                        width=args.width,
                        num_inference_steps=args.steps,
                        guidance_scale=args.guidance,
                        generator=gen,
                        image=input_image,  # 传入输入图像
                    )
                else:
                    result = pipe(
                        prompt=prompt,
                        negative_prompt=args.negative_prompt,
                        height=args.height,
                        width=args.width,
                        num_inference_steps=args.steps,
                        guidance_scale=args.guidance,
                        generator=gen,
                    )
            except TypeError:
                result = pipe(prompt, num_inference_steps=args.steps, guidance_scale=args.guidance, generator=gen)

            images = result.images if hasattr(result, "images") else result

            # 输出文件名
            if out_is_file:
                out_path = args.out
            else:
                out_dir = args.out
                out_dir.mkdir(parents=True, exist_ok=True)
                safe_prompt = "".join(c for c in prompt if c.isalnum() or c in " _-[](){}")[:80].strip()
                out_path = out_dir / f"gen_{i+1}_{k+1}_{safe_prompt}_{args.seed}.png"

            if isinstance(images, list):
                for idx, img in enumerate(images):
                    if len(images) == 1:
                        img.save(out_path)
                    else:
                        out_path_idx = out_path.with_name(out_path.stem + f"_{idx}.png")
                        images[idx].save(out_path_idx)
            else:
                if isinstance(images, Image.Image):
                    images.save(out_path)
                else:
                    Image.fromarray(images).save(out_path)

            total += 1

    LOG.info("完成，共生成 %d 张图像", total)


if __name__ == "__main__":
    main()















    