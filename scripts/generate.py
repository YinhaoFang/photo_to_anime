# file: /mnt/data/generate.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用的 Stable Diffusion / XL 本地模型推理脚本。
在原版基础上新增：
- 支持 --input-folder：将文件夹中所有图片拼成一张拼接图，再用于生成。
- 生成张数仍由 --num-per-prompt 控制（每个 prompt 可生成多张）。
"""

import argparse
import logging
from pathlib import Path
import sys
import time
import os
import math

import torch
from PIL import Image
import numpy as np
import cv2

from diffusers import DiffusionPipeline

LOG = logging.getLogger("generate")


def setup_logger(logfile: Path | None = None):
    LOG.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    LOG.addHandler(sh)
    if logfile:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(logfile), encoding="utf-8")
        fh.setFormatter(fmt)
        LOG.addHandler(fh)  # 沿用原日志体系:contentReference[oaicite:7]{index=7}


def get_images_from_folder(folder_path: Path):
    """枚举文件夹内所有支持的图片路径。"""
    valid_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"]
    image_paths = []
    for filename in os.listdir(folder_path):
        if any(filename.lower().endswith(ext) for ext in valid_extensions):
            image_paths.append(folder_path / filename)
    return image_paths  # 已有函数，直接复用:contentReference[oaicite:8]{index=8}


def load_pipeline(
    model_dir: Path,
    device: str,
    torch_dtype: torch.dtype,
    lora_weight: Path | None = None,
    controlnet_weight: Path | None = None,
):
    """从本地加载 diffusers pipeline，并做常规优化。"""
    LOG.info("尝试解析模型目录: %s", model_dir)

    if not model_dir.exists():
        possible = Path(__file__).resolve().parent.parent / model_dir
        if possible.exists():
            model_dir = possible
            LOG.info("模型目录在项目根下找到，使用: %s", model_dir)
        else:
            alt = Path(model_dir).expanduser().resolve()
            if alt.exists():
                model_dir = alt
                LOG.info("模型目录解析为绝对路径: %s", model_dir)
            else:
                LOG.error(
                    "无法找到模型目录，请检查 --model-dir。尝试: %s / %s / %s",
                    model_dir, possible, alt
                )
                raise FileNotFoundError(f"模型目录不存在: {model_dir}")

    LOG.info("从本地加载模型: %s", model_dir)
    try:
        pipe = DiffusionPipeline.from_pretrained(
            str(model_dir), torch_dtype=torch_dtype, local_files_only=True
        )
    except Exception as e:
        LOG.warning("本地直载失败，尝试允许在线下载: %s", e)
        pipe = DiffusionPipeline.from_pretrained(str(model_dir), local_files_only=False)

    if lora_weight and Path(lora_weight).exists():
        pipe.load_lora_weights(str(lora_weight))
        LOG.info("加载 LoRA 权重: %s", lora_weight)
    else:
        LOG.info("未提供 LoRA 权重，跳过")

    if controlnet_weight and Path(controlnet_weight).exists():
        pipe.load_controlnet_weights(str(controlnet_weight))
        LOG.info("加载 ControlNet 权重: %s", controlnet_weight)
    else:
        LOG.info("未提供 ControlNet 权重，跳过")

    pipe.to(device)

    if hasattr(pipe, "safety_checker"):
        try:
            # 仅为避免生成被安全检查阻塞；如需开启请移除此段。
            pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
            LOG.info("已禁用 safety_checker")
        except Exception:
            pass

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

    if device.startswith("cuda"):
        try:
            pipe.enable_xformers_memory_efficient_attention()
            LOG.info("已启用 xformers")
        except Exception:
            LOG.info("xformers 未启用")

    return pipe  # 保持与原脚本一致:contentReference[oaicite:9]{index=9}


def read_prompts(prompts_arg: str | None, prompts_file: Path | None):
    """读取单条或文件多条 prompt。"""
    prompts = []
    if prompts_arg:
        prompts.append(prompts_arg)
    if prompts_file:
        with open(prompts_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    prompts.append(line)
    return prompts  # 保持与原脚本一致:contentReference[oaicite:10]{index=10}


def process_image(input_image: Path, size: int = 512):
    """单图预处理（保留以兼容旧用法）。"""
    image = Image.open(input_image).convert("RGB")
    image = image.resize((size, size), Image.LANCZOS)
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_image = Image.fromarray(edges)
    return image, edge_image  # 原逻辑保留:contentReference[oaicite:11]{index=11}


def compose_collage(
    image_paths: list[Path],
    out_w: int,
    out_h: int,
    pad: int = 8,
    bg_color=(255, 255, 255),
) -> Image.Image:
    """
    将多张图片按近似方形网格拼接为一张，尺寸严格为 (out_w, out_h)。

    为什么这么做：多数 SD/XL 图像条件仅接受单张图，拼接能“利用所有图片信息”且通用。
    """
    if not image_paths:
        raise ValueError("输入文件夹中未发现有效图片")

    n = len(image_paths)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    # 计算每格可用区域（含内边距）
    cell_w = (out_w - pad * (cols + 1)) // cols
    cell_h = (out_h - pad * (rows + 1)) // rows
    if cell_w <= 0 or cell_h <= 0:
        raise ValueError("输出分辨率过小或图片过多，无法排版，请增大 --width/--height 或减少图片数")

    canvas = Image.new("RGB", (out_w, out_h), color=bg_color)
    i = 0
    for r in range(rows):
        for c in range(cols):
            if i >= n:
                break
            p = image_paths[i]
            try:
                im = Image.open(p).convert("RGB")
            except Exception as e:
                LOG.warning("跳过无法读取的图片: %s (%s)", p, e)
                i += 1
                continue
            # 等比缩放填满单元格的短边，避免严重拉伸
            im_ratio = im.width / im.height
            cell_ratio = cell_w / cell_h
            if im_ratio > cell_ratio:
                # 宽限制
                new_w = cell_w
                new_h = int(new_w / im_ratio)
            else:
                # 高限制
                new_h = cell_h
                new_w = int(new_h * im_ratio)
            im = im.resize((max(1, new_w), max(1, new_h)), Image.LANCZOS)
            # 置于单元格中央
            x0 = pad + c * (cell_w + pad) + (cell_w - im.width) // 2
            y0 = pad + r * (cell_h + pad) + (cell_h - im.height) // 2
            canvas.paste(im, (x0, y0))
            i += 1
    return canvas


def main():
    parser = argparse.ArgumentParser(description="Generate images with a local diffusers model")
    parser.add_argument("--model-dir", type=Path, default=Path("models"), help="本地模型目录（含 model_index.json）")
    parser.add_argument("--out", type=Path, required=True, help="输出文件或输出目录")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompt", type=str, help="单条 prompt")
    group.add_argument("--prompts-file", type=Path, help="每行一个 prompt 的文本文件")

    # 新增：从文件夹读取多图并拼接为一张
    parser.add_argument("--input-folder", type=Path, help="输入图像文件夹；会将全部图片拼为一张作为条件图像")
    parser.add_argument("--num-per-prompt", type=int, default=1, help="每个 prompt 生成张数（默认1）")
    parser.add_argument("--seed", type=int, default=None, help="随机种子，默认随机")
    parser.add_argument("--width", type=int, default=1024, help="输出宽度")
    parser.add_argument("--height", type=int, default=1024, help="输出高度")
    parser.add_argument("--steps", type=int, default=28, help="推理步数")
    parser.add_argument("--guidance", type=float, default=7.5, help="classifier-free guidance scale")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", choices=["cuda", "cpu"], help="运行设备")
    parser.add_argument("--log", type=Path, default=Path("logs/generate.log"), help="日志文件")
    parser.add_argument("--batch-size", type=int, default=1, help="管道一次处理多少张（若模型支持）")
    parser.add_argument("--negative-prompt", type=str, default=None, help="negative prompt")

    args = parser.parse_args()

    # 目录准备与日志
    args.out.parent.mkdir(parents=True, exist_ok=True) if args.out.suffix else args.out.mkdir(parents=True, exist_ok=True)
    setup_logger(args.log)
    LOG.info("参数: %s", args)

    # dtype 选择（沿用）
    torch_dtype = torch.float16 if args.device == "cuda" else torch.float32  # :contentReference[oaicite:12]{index=12}

    # 加载 pipeline
    pipe = load_pipeline(args.model_dir, args.device, torch_dtype, lora_weight=None, controlnet_weight=None)  # :contentReference[oaicite:13]{index=13}

    # 读取 prompts
    prompts = read_prompts(args.prompt, args.prompts_file)
    if not prompts:
        LOG.error("没有找到任何 prompts")
        return

    # 从文件夹构造拼接图（核心改动）
    collage_image = None
    if args.input_folder:
        if not args.input_folder.exists():
            raise FileNotFoundError(f"输入文件夹不存在: {args.input_folder}")
        paths = get_images_from_folder(args.input_folder)
        if not paths:
            raise ValueError(f"文件夹中未发现图片: {args.input_folder}")
        LOG.info("发现 %d 张图片，开始拼接为一张 (%dx%d)...", len(paths), args.width, args.height)
        collage_image = compose_collage(paths, args.width, args.height, pad=8)
        LOG.info("拼接完成，尺寸: %s", collage_image.size)

    # 随机种子与 generator（沿用）
    if args.seed is None:
        args.seed = int(time.time() % (2 ** 31 - 1))
    LOG.info("使用种子: %d", args.seed)
    gen = torch.Generator("cuda" if args.device == "cuda" else "cpu").manual_seed(args.seed)

    out_is_file = args.out.suffix in [".png", ".jpg", ".jpeg"]

    total = 0
    for i, prompt in enumerate(prompts):
        for k in range(args.num_per_prompt):
            LOG.info("生成：prompt %d/%d - %d/%d", i + 1, len(prompts), k + 1, args.num_per_prompt)
            try:
                if collage_image is not None:
                    result = pipe(
                        prompt=prompt,
                        negative_prompt=args.negative_prompt,
                        height=args.height,
                        width=args.width,
                        num_inference_steps=args.steps,
                        guidance_scale=args.guidance,
                        generator=gen,
                        image=collage_image,  # 关键：利用整夹拼接图
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
                # 兼容不支持关键字的老接口
                result = pipe(prompt, num_inference_steps=args.steps, guidance_scale=args.guidance, generator=gen)

            images = result.images if hasattr(result, "images") else result

            # 输出命名（沿用）
            if out_is_file:
                out_path = args.out
            else:
                out_dir = args.out
                out_dir.mkdir(parents=True, exist_ok=True)
                safe_prompt = "".join(c for c in prompt if c.isalnum() or c in " _-[](){}")[:80].strip()
                out_path = out_dir / f"gen_{i+1}_{k+1}_{safe_prompt}_{args.seed}.png"

            if isinstance(images, list):
                if len(images) == 1:
                    images[0].save(out_path)
                else:
                    for idx, img in enumerate(images):
                        out_path_idx = out_path.with_name(out_path.stem + f"_{idx}.png")
                        img.save(out_path_idx)
            else:
                if isinstance(images, Image.Image):
                    images.save(out_path)
                else:
                    Image.fromarray(images).save(out_path)

            total += 1

    LOG.info("完成，共生成 %d 张图像", total)


if __name__ == "__main__":
    main()
