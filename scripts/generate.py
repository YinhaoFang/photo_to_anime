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

用法示例：
    python scripts/generate.py --model-dir models --prompt "a girl in hanfu, anime style" --out outputs/test.png
    python scripts/generate.py --model-dir models --prompts-file prompts.txt --num-per-prompt 2 --out outputs/

依赖：
    torch, diffusers, transformers, accelerate, safetensors, Pillow

作者：你的项目组
"""

import argparse
import logging
from pathlib import Path
import sys
import time

import torch
from PIL import Image

from diffusers import DiffusionPipeline


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


def load_pipeline(model_dir: Path, device: str, torch_dtype: torch.dtype):
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


def main():
    parser = argparse.ArgumentParser(description="Generate images with a local diffusers model")
    parser.add_argument("--model-dir", type=Path, default=Path("models"), help="本地模型目录（含 model_index.json）")
    parser.add_argument("--out", type=Path, required=True, help="输出文件或输出目录")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompt", type=str, help="单条 prompt")
    group.add_argument("--prompts-file", type=Path, help="每行一个 prompt 的文本文件")

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
    pipe = load_pipeline(args.model_dir, args.device, torch_dtype)

    # 读取 prompt(s)
    prompts = read_prompts(args.prompt, args.prompts_file)
    if not prompts:
        LOG.error("没有找到任何 prompts")
        return

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

            # 支持部分 pipeline 的不同参数签名（部分版本使用 "generator", 有的使用 "num_images_per_prompt"）
            try:
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
                # 兼容旧签名
                result = pipe(prompt, num_inference_steps=args.steps, guidance_scale=args.guidance, generator=gen)

            images = result.images if hasattr(result, "images") else result

            # 输出文件名
            if out_is_file:
                # 如果用户指定了一个具体文件（单 prompt 单图场景），按顺序覆盖或只保存第一个
                out_path = args.out
            else:
                out_dir = args.out
                out_dir.mkdir(parents=True, exist_ok=True)
                safe_prompt = "".join(c for c in prompt if c.isalnum() or c in " _-[](){}")[:80].strip()
                out_path = out_dir / f"gen_{i+1}_{k+1}_{safe_prompt}_{args.seed}.png"

            # 如果生成多张图片则分别保存
            if isinstance(images, list):
                for idx, img in enumerate(images):
                    if len(images) == 1:
                        img_to_save = img
                        img_to_save.save(out_path)
                        LOG.info("保存: %s", out_path)
                    else:
                        out_path_idx = out_path.with_name(out_path.stem + f"_{idx}.png")
                        images[idx].save(out_path_idx)
                        LOG.info("保存: %s", out_path_idx)
            else:
                # 单张图片对象
                if isinstance(images, Image.Image):
                    images.save(out_path)
                else:
                    # 有些 pipeline 返回 numpy
                    Image.fromarray(images).save(out_path)
                LOG.info("保存: %s", out_path)

            total += 1

    LOG.info("完成，共生成 %d 张图像", total)


if __name__ == "__main__":
    main()