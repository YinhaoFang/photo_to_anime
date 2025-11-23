#!/usr/bin/env python3
# train_lora.py
"""
Train LoRA adapters for Animagine-XL (Stable Diffusion) + ControlNet.

主要功能：
- 在 UNet / text_encoder / (可选) ControlNet 注入 LoRA (LoRALinear)
- 只训练 LoRA 参数，冻结原始模型参数
- 支持 ControlNet 条件（edge/pose maps）
- 保存 LoRA 为 safetensors（轻量），支持 resume/load

Author: 生成脚本（根据项目 README 规范）
"""

import os
import argparse
import json
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from accelerate import Accelerator
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, ControlNetModel, DDPMScheduler
from safetensors.torch import save_file as safetensors_save, load_file as safetensors_load

# ---------------------
# LoRA wrapper
# ---------------------
class LoRALinear(nn.Module):
    """
    Wrap an nn.Linear with LoRA low-rank adapters.
    out = orig(x) + scaling * (x @ A.T @ B.T)
    where A: (r, in_features), B: (out_features, r)
    """
    def __init__(self, orig: nn.Linear, r: int = 4, alpha: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        self.orig = orig
        # keep original weight/bias but freeze them by default
        self.in_features = orig.in_features
        self.out_features = orig.out_features
        self.r = r
        self.alpha = alpha if alpha is not None else r
        self.scaling = float(self.alpha) / max(1, self.r)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        if self.r > 0:
            self.lora_A = nn.Parameter(torch.zeros(self.r, self.in_features))
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.r))
            nn.init.normal_(self.lora_A, std=0.01)
            nn.init.zeros_(self.lora_B)
        else:
            # placeholders for consistent attribute access
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        # freeze original params
        for p in self.orig.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.orig(x)
        if self.r > 0:
            xi = self.dropout(x) if self.dropout is not None else x
            # (B, in) @ A.T -> (B, r); @ B.T -> (B, out)
            lora_out = (xi @ self.lora_A.T) @ self.lora_B.T
            out = out + lora_out * self.scaling
        return out

# ---------------------
# Helpers: patching & state collection
# ---------------------
def replace_linear_with_lora(module: nn.Module, r: int = 4, alpha: Optional[int] = None, dropout: float = 0.0,
                             target_names: Optional[List[str]] = None, prefix: str = "") -> int:
    """
    Recursively replace nn.Linear in module with LoRALinear when name contains any of target_names substrings.
    If target_names is None -> replace all nn.Linear.
    Returns number of replacements.
    """
    replaced = 0
    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            should = target_names is None or any(t in full_name for t in target_names)
            if should:
                setattr(module, name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
                replaced += 1
        else:
            replaced += replace_linear_with_lora(child, r=r, alpha=alpha, dropout=dropout,
                                                 target_names=target_names, prefix=full_name)
    return replaced

def collect_lora_state_dict(root_module: nn.Module, module_key_prefix: str) -> Dict[str, torch.Tensor]:
    """
    Collect LoRA params (A/B and scaling) from modules and return a dict of tensors ready to save.
    Keys are prefixed with module_key_prefix to indicate component (e.g., 'unet.', 'text_encoder.')
    """
    out = {}
    for name, m in root_module.named_modules():
        if isinstance(m, LoRALinear) and getattr(m, "r", 0) > 0:
            keybase = f"{module_key_prefix}{name.replace('.', '_')}"
            out[f"{keybase}.lora_A"] = m.lora_A.detach().cpu()
            out[f"{keybase}.lora_B"] = m.lora_B.detach().cpu()
            out[f"{keybase}.scaling"] = torch.tensor(m.scaling)
            out[f"{keybase}.r"] = torch.tensor(m.r, dtype=torch.int32)
    return out

def load_lora_into_module(root_module: nn.Module, state: Dict[str, torch.Tensor], module_key_prefix: str):
    """
    Load LoRA tensors from state dict into LoRALinear modules of root_module.
    module_key_prefix must match keys used in collect_lora_state_dict.
    """
    for name, m in root_module.named_modules():
        if isinstance(m, LoRALinear) and getattr(m, "r", 0) > 0:
            keybase = f"{module_key_prefix}{name.replace('.', '_')}"
            a_key = f"{keybase}.lora_A"
            b_key = f"{keybase}.lora_B"
            s_key = f"{keybase}.scaling"
            if a_key in state and b_key in state:
                a = state[a_key].to(m.lora_A.device)
                b = state[b_key].to(m.lora_B.device)
                if a.shape == m.lora_A.shape and b.shape == m.lora_B.shape:
                    m.lora_A.data.copy_(a)
                    m.lora_B.data.copy_(b)
                else:
                    # shape mismatch: try transpose-compatible or skip
                    print(f"[WARN] shape mismatch for {keybase}, skipping load")
                if s_key in state:
                    m.scaling = float(state[s_key].item())

# ---------------------
# Dataset
# ---------------------
class PhotoControlDataset(Dataset):
    """
    Dataset supports:
      - images_dir: directory with processed photos (512x512)
      - control_dir (optional): directory with control maps (edges/pose) aligned by filename
      - captions_file (optional): CSV or JSONL mapping filename->caption
    If captions_file missing, uses default prompt template or filename stem as caption.
    """
    def __init__(self, images_dir: str, control_dir: Optional[str] = None, captions_file: Optional[str] = None,
                 tokenizer: Optional[CLIPTokenizer] = None, resolution: int = 512, prompt_template: Optional[str] = None,
                 flip_prob: float = 0.0):
        super().__init__()
        self.images_dir = images_dir
        self.control_dir = control_dir
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.flip_prob = flip_prob
        self.prompt_template = prompt_template

        # load entries
        self.samples = []  # list of (image_path, control_path_or_None, caption)
        captions_map = {}
        if captions_file and os.path.exists(captions_file):
            if captions_file.endswith(".jsonl") or captions_file.endswith(".json"):
                with open(captions_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        obj = json.loads(line)
                        fn = obj.get("filename") or obj.get("file") or obj.get("image")
                        caption = obj.get("caption") or obj.get("text") or ""
                        if fn:
                            captions_map[fn] = caption
            else:
                # simple CSV: filename,caption
                import csv
                with open(captions_file, newline='', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if len(row) >= 2:
                            captions_map[row[0]] = row[1]

        for fn in os.listdir(images_dir):
            if not fn.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                continue
            img_path = os.path.join(images_dir, fn)
            control_path = None
            if control_dir:
                candidate = os.path.join(control_dir, fn)
                if os.path.exists(candidate):
                    control_path = candidate
            caption = captions_map.get(fn, None)
            if caption is None:
                stem = os.path.splitext(fn)[0]
                if self.prompt_template:
                    caption = self.prompt_template.replace("{filename}", stem)
                else:
                    # default generic caption emphasizing profile/photo->anime conversion
                    caption = "a high quality Chinese traditional anime portrait, intricate hanfu, delicate brush style"
            self.samples.append((img_path, control_path, caption))

        # transforms
        self.image_transform = transforms.Compose([
            transforms.Resize((self.resolution, self.resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
        ])
        self.control_transform = transforms.Compose([
            transforms.Resize((self.resolution, self.resolution)),
            transforms.ToTensor(),
            # control maps typically in [0,1], but diffusers expects [-1,1] latents later; we keep as [0,1]
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, control_path, caption = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if torch.rand(1).item() < self.flip_prob:
            image = transforms.functional.hflip(image)
        pixel_values = self.image_transform(image)

        control_image = None
        if control_path:
            ci = Image.open(control_path).convert("RGB")
            if torch.rand(1).item() < self.flip_prob:
                ci = transforms.functional.hflip(ci)
            control_image = self.control_transform(ci)

        input_ids = None
        if self.tokenizer is not None:
            toks = self.tokenizer(caption, truncation=True, padding="max_length", max_length=self.tokenizer.model_max_length, return_tensors="pt")
            input_ids = toks.input_ids[0]
        else:
            input_ids = caption

        return {"pixel_values": pixel_values, "control_image": control_image, "input_ids": input_ids, "filename": os.path.basename(img_path)}

# ---------------------
# Training core
# ---------------------
def train(args):
    accelerator = Accelerator(mixed_precision=args.mixed_precision if args.mixed_precision != "no" else None)
    device = accelerator.device

    print("Device:", device)

    # Load tokenizer and models
    print("Loading models...")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", use_fast=True)
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet").to(device)

    controlnet = None
    if args.controlnet_model:
        controlnet = ControlNetModel.from_pretrained(args.controlnet_model).to(device)
        # By default freeze controlnet base params; we optionally inject LoRA into controlnet if requested
        for p in controlnet.parameters():
            p.requires_grad = False

    # Freeze base model params (we only train LoRA adapters)
    for p in vae.parameters(): p.requires_grad = False
    for p in unet.parameters(): p.requires_grad = False
    for p in text_encoder.parameters(): p.requires_grad = False

    # Inject LoRA into text_encoder / unet / optional controlnet
    print("Injecting LoRA adapters...")
    te_count = replace_linear_with_lora(text_encoder, r=args.rank, alpha=args.alpha, dropout=args.lora_dropout, target_names=args.te_target_names)
    unet_count = replace_linear_with_lora(unet, r=args.rank, alpha=args.alpha, dropout=args.lora_dropout, target_names=args.unet_target_names)
    cn_count = 0
    if controlnet and args.train_controlnet_lora:
        cn_count = replace_linear_with_lora(controlnet, r=args.rank, alpha=args.alpha, dropout=args.lora_dropout, target_names=args.cn_target_names)

    print(f"LoRA injected: text_encoder {te_count}, unet {unet_count}, controlnet {cn_count}")

    # Optionally load existing LoRA safetensors (resume)
    if args.lora_checkpoint and os.path.exists(args.lora_checkpoint):
        print("Loading LoRA checkpoint:", args.lora_checkpoint)
        state = safetensors_load(args.lora_checkpoint)
        # Expect keys with prefixes: 'text_encoder.', 'unet.', 'controlnet.' as saved by this script
        load_lora_into_module(text_encoder, state, "text_encoder.")
        load_lora_into_module(unet, state, "unet.")
        if controlnet:
            load_lora_into_module(controlnet, state, "controlnet.")

    # Collect LoRA parameters
    def gather_lora_params(module):
        ps = []
        for m in module.modules():
            if isinstance(m, LoRALinear) and getattr(m, "r", 0) > 0:
                ps.append(m.lora_A)
                ps.append(m.lora_B)
        return ps

    trainable_params = []
    trainable_params += gather_lora_params(text_encoder)
    trainable_params += gather_lora_params(unet)
    if controlnet and args.train_controlnet_lora:
        trainable_params += gather_lora_params(controlnet)

    if len(trainable_params) == 0:
        raise RuntimeError("No LoRA parameters found to train. Check injection or target names.")

    # Optimizer
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    # Dataset / Dataloader
    dataset = PhotoControlDataset(
        images_dir=args.train_data_dir,
        control_dir=args.control_data_dir,
        captions_file=args.captions_file,
        tokenizer=tokenizer,
        resolution=args.resolution,
        prompt_template=args.prompt_template,
        flip_prob=args.flip_prob
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    # Scheduler for diffusion noise (simple default)
    if os.path.exists(os.path.join(args.pretrained_model_name_or_path, "scheduler")):
        noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    else:
        noise_scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

    # Prepare with accelerator
    text_encoder, unet, optimizer, dataloader = accelerator.prepare(text_encoder, unet, optimizer, dataloader)
    if controlnet and args.train_controlnet_lora:
        # ensure controlnet is on device for potential LoRA params (LoRA inside controlnet still requires module on device)
        controlnet.to(accelerator.device)

    vae.eval()  # VAE not trained
    noise_scheduler.to(accelerator.device)

    # Training loop
    global_step = 0
    text_encoder.train()
    unet.train()
    if controlnet and args.train_controlnet_lora:
        controlnet.train()
    else:
        if controlnet:
            controlnet.eval()

    scaler = None  # accelerate handles mixed precision

    print("Starting training...")
    for epoch in range(args.num_train_epochs):
        loop = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}")
        for step, batch in loop:
            pixel_values = batch["pixel_values"].to(accelerator.device)
            input_ids = batch["input_ids"].to(accelerator.device)
            control_image = None
            if batch.get("control_image") is not None:
                control_image = batch["control_image"].to(accelerator.device)
            # Encode images to latents
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor

            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
            # noise and noisy latents
            noise = torch.randn_like(latents)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # text encoder embeddings
            encoder_outputs = text_encoder(input_ids)
            encoder_hidden_states = encoder_outputs.last_hidden_state if hasattr(encoder_outputs, "last_hidden_state") else encoder_outputs[0]

            # ControlNet forward: produce additional residuals (no grad if frozen)
            cn_out = None
            if controlnet and control_image is not None:
                # controlnet call signature may vary: try common arg name 'conditioning_image' then fallback 'controlnet_cond'
                try:
                    cn_out = controlnet(noisy_latents, timesteps, encoder_hidden_states, conditioning_image=control_image)
                except TypeError:
                    try:
                        cn_out = controlnet(noisy_latents, timesteps, encoder_hidden_states, controlnet_cond=control_image)
                    except Exception as e:
                        raise RuntimeError(f"ControlNet forward failed: {e}")

            # UNet forward with optional additional residuals
            unet_kwargs = {"encoder_hidden_states": encoder_hidden_states}
            if cn_out is not None:
                # expected keys: down_block_additional_residuals, mid_block_additional_residual
                if hasattr(cn_out, "down_block_additional_residuals"):
                    unet_kwargs["down_block_additional_residuals"] = cn_out.down_block_additional_residuals
                if hasattr(cn_out, "mid_block_additional_residual"):
                    unet_kwargs["mid_block_additional_residual"] = cn_out.mid_block_additional_residual

            model_pred = unet(noisy_latents, timesteps, **unet_kwargs).sample

            loss = F.mse_loss(model_pred, noise)
            loss = loss / args.gradient_accumulation_steps

            accelerator.backward(loss)

            if (global_step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if global_step % args.logging_steps == 0:
                loop.set_postfix({"step": global_step, "loss": float(loss.detach().cpu())})

            # Save checkpoints
            if global_step % args.save_steps == 0 and global_step > 0:
                os.makedirs(args.output_dir, exist_ok=True)
                # collect LoRA states
                sd = {}
                sd.update(collect_lora_state_dict(text_encoder, "text_encoder."))
                sd.update(collect_lora_state_dict(unet, "unet."))
                if controlnet:
                    sd.update(collect_lora_state_dict(controlnet, "controlnet."))
                ckpt_path = os.path.join(args.output_dir, f"lora_step_{global_step}.safetensors")
                safetensors_save(sd, ckpt_path)
                print(f"[INFO] Saved LoRA checkpoint: {ckpt_path}")

            global_step += 1
            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    # Final save
    os.makedirs(args.output_dir, exist_ok=True)
    sd = {}
    sd.update(collect_lora_state_dict(text_encoder, "text_encoder."))
    sd.update(collect_lora_state_dict(unet, "unet."))
    if controlnet:
        sd.update(collect_lora_state_dict(controlnet, "controlnet."))
    final_path = os.path.join(args.output_dir, f"lora_final.safetensors")
    safetensors_save(sd, final_path)
    print(f"[DONE] Training finished. Saved final LoRA: {final_path}")

# ---------------------
# CLI
# ---------------------
def get_args():
    parser = argparse.ArgumentParser(description="Train LoRA for Animagine-XL + ControlNet")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True, help="Base Animagine-XL model dir or HF id")
    parser.add_argument("--controlnet_model", type=str, default=None, help="ControlNet model dir or HF id (optional)")
    parser.add_argument("--train_data_dir", type=str, required=True, help="path to processed photos directory (data/processed/photos)")
    parser.add_argument("--control_data_dir", type=str, default=None, help="path to processed control maps (data/processed/edges)")
    parser.add_argument("--captions_file", type=str, default=None, help="optional captions CSV/JSONL mapping filename->caption")
    parser.add_argument("--output_dir", type=str, default="./lora_out", help="output dir for safetensors")
    parser.add_argument("--lora_checkpoint", type=str, default=None, help="optional existing safetensors to load (resume)")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=1000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--rank", type=int, default=4, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=None, help="LoRA alpha scaling")
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--flip_prob", type=float, default=0.0)
    parser.add_argument("--prompt_template", type=str, default=None, help="optional prompt template for samples without captions")
    parser.add_argument("--train_controlnet_lora", action="store_true", help="also inject and train LoRA in ControlNet")
    parser.add_argument("--te_target_names", nargs="*", default=["q_proj", "k_proj", "v_proj", "out_proj", "proj"], help="text encoder replacement name substrings")
    parser.add_argument("--unet_target_names", nargs="*", default=["to_q", "to_k", "to_v", "to_out", "proj_out", "proj_in"], help="unet replacement name substrings")
    parser.add_argument("--cn_target_names", nargs="*", default=["to_q", "to_k", "to_v", "to_out", "proj"], help="controlnet replacement name substrings")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    train(args)