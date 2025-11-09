# -*- coding: utf-8 -*-

from datetime import datetime
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import random
from PIL import Image, ImageDraw, ImageOps

import csv
import re
import torch
from PIL import Image
from diffusers import QwenImageEditPipeline,QwenImagePipeline
from diffusers import QwenImageInpaintPipeline,QwenImageEditInpaintPipeline
from diffusers.utils import load_image
import os
import cv2
import torch
import inspect
import numpy as np
from torch import nn
from tqdm.auto import tqdm
import json
import argparse  
import math      
from torchmetrics.image.fid import FrechetInceptionDistance

from torchvision import transforms

from typing import List, Tuple, Union, Optional

from datetime import timedelta

import torch.nn.functional as F

# from diffusers import QwenImageEditPlusPipeline
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus_ours import QwenImageEditPlusPipeline,retrieve_timesteps,calculate_shift
CONDITION_IMAGE_AREA = 384 * 384
VAE_IMAGE_AREA = 1024 * 1024


def extract_masked_region(
    ref_image: Union[Image.Image, np.ndarray],
    ref_mask: Union[Image.Image, np.ndarray],
    align_to: int = 1,
    padding: int = 0
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """
    提取mask对应的区域，并将宽高对齐到指定整数倍

    Args:
        ref_image: 参考图像 (PIL Image 或 numpy array)
        ref_mask: 参考mask (PIL Image 或 numpy array), 白色区域为要提取的部分
        align_to: 对齐的整数倍，例如28表示宽高对齐到28的整数倍
        padding: 在mask边界外额外添加的padding像素数

    Returns:
        extracted_image: 提取并对齐后的图像区域
        bbox: 边界框 (x1, y1, x2, y2) 在原图中的坐标
    """
    # 转换为numpy array
    if isinstance(ref_image, Image.Image):
        image_array = np.array(ref_image)
    else:
        image_array = ref_image.copy()

    if isinstance(ref_mask, Image.Image):
        mask_array = np.array(ref_mask)
    else:
        mask_array = ref_mask.copy()

    # 确保mask是二值化的
    if len(mask_array.shape) == 3:
        mask_array = mask_array[:, :, 0]

    # 二值化mask (threshold at 128)
    binary_mask = mask_array > 128

    # 找到mask的边界框
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)

    if not np.any(rows) or not np.any(cols):
        raise ValueError("Mask is empty or contains no white pixels")

    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]

    # 添加padding
    if padding > 0:
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image_array.shape[1] - 1, x2 + padding)
        y2 = min(image_array.shape[0] - 1, y2 + padding)

    # 计算当前区域的宽高
    width = x2 - x1 + 1
    height = y2 - y1 + 1

    # 对齐到指定整数倍
    if align_to > 1:
        aligned_width = int(np.ceil(width / align_to)) * align_to
        aligned_height = int(np.ceil(height / align_to)) * align_to

        # 计算需要扩展的像素数
        width_diff = aligned_width - width
        height_diff = aligned_height - height

        # 尽量居中扩展
        left_expand = width_diff // 2
        right_expand = width_diff - left_expand
        top_expand = height_diff // 2
        bottom_expand = height_diff - top_expand

        # 调整边界框，确保不超出图像范围
        new_x1 = max(0, x1 - left_expand)
        new_y1 = max(0, y1 - top_expand)
        new_x2 = min(image_array.shape[1] - 1, x2 + right_expand)
        new_y2 = min(image_array.shape[0] - 1, y2 + bottom_expand)

        # 如果一侧到达边界，在另一侧补偿
        actual_width = new_x2 - new_x1 + 1
        actual_height = new_y2 - new_y1 + 1

        if actual_width < aligned_width:
            if new_x1 == 0:
                new_x2 = min(image_array.shape[1] - 1, new_x1 + aligned_width - 1)
            else:
                new_x1 = max(0, new_x2 - aligned_width + 1)

        if actual_height < aligned_height:
            if new_y1 == 0:
                new_y2 = min(image_array.shape[0] - 1, new_y1 + aligned_height - 1)
            else:
                new_y1 = max(0, new_y2 - aligned_height + 1)

        x1, y1, x2, y2 = new_x1, new_y1, new_x2, new_y2

    # 提取区域
    extracted_array = image_array[y1:y2+1, x1:x2+1]

    # 转换回PIL Image
    extracted_image = Image.fromarray(extracted_array)

    return extracted_image, (x1, y1, x2, y2)


def extract_masked_region_with_mask(
    ref_image: Union[Image.Image, np.ndarray],
    ref_mask: Union[Image.Image, np.ndarray],
    align_to: int = 1,
    padding: int = 0
) -> Tuple[Image.Image, Image.Image, Tuple[int, int, int, int]]:
    """
    提取mask对应的区域（同时提取图像和mask），并将宽高对齐到指定整数倍

    Args:
        ref_image: 参考图像 (PIL Image 或 numpy array)
        ref_mask: 参考mask (PIL Image 或 numpy array), 白色区域为要提取的部分
        align_to: 对齐的整数倍，例如28表示宽高对齐到28的整数倍
        padding: 在mask边界外额外添加的padding像素数

    Returns:
        extracted_image: 提取并对齐后的图像区域
        extracted_mask: 提取并对齐后的mask区域
        bbox: 边界框 (x1, y1, x2, y2) 在原图中的坐标
    """
    # 转换为numpy array
    if isinstance(ref_mask, Image.Image):
        mask_array = np.array(ref_mask)
    else:
        mask_array = ref_mask.copy()

    # 提取图像区域
    extracted_image, bbox = extract_masked_region(ref_image, ref_mask, align_to, padding)

    # 提取对应的mask区域
    x1, y1, x2, y2 = bbox
    extracted_mask_array = mask_array[y1:y2+1, x1:x2+1]
    extracted_mask = Image.fromarray(extracted_mask_array)

    return extracted_image, extracted_mask, bbox


def calculate_dimensions(target_area: int, ratio: float) -> Tuple[int, int]:
    w = math.sqrt(target_area * ratio)
    h = w / ratio
    w = round(w / 32) * 32
    h = round(h / 32) * 32
    return int(w), int(h)


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode_vae_and_pack(pipe: QwenImageEditPlusPipeline, img_tensor_5d: torch.Tensor, generator=None):
    """
    img_tensor_5d: [B, C, T=1, H, W]，值域 [0,1]（由 pipe.image_processor.preprocess 保证）
    返回：packed_tokens [B, N, C*4] 以及 原始 latent HxW 尺寸
    """
    with torch.no_grad():
        enc = pipe.vae.encode(img_tensor_5d.to(device=pipe._execution_device, dtype=pipe.vae.dtype))
        z = enc.latent_dist.mode()
        # 标准化
        z_mean = torch.tensor(pipe.vae.config.latents_mean).view(1, pipe.latent_channels, 1, 1, 1).to(z.device, z.dtype)
        z_std  = torch.tensor(pipe.vae.config.latents_std ).view(1, pipe.latent_channels, 1, 1, 1).to(z.device, z.dtype)
        z = (z - z_mean) / z_std
    b, c, t, h, w = z.shape
    packed = pipe._pack_latents(z, b, c, h, w)  # [B, N, C*4]
    return packed, (h, w)

def load_models(pretrained_model_name_or_path="/home/v-qinhyang/code/hero_blob/preweight/Qwen/Qwen-Image",
                lora_path = None,
                device=torch.device("cuda")
                ):
    """加载所有必要的模型组件"""
    print(f"Loading models from: {pretrained_model_name_or_path}")
    if lora_path is None:
        if "2509" in pretrained_model_name_or_path:
            pipe = QwenImageEditPlusPipeline.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.bfloat16)
            print("QwenImageEditPlusPipeline Models loaded successfully!")
        if "Edit" in pretrained_model_name_or_path:
            pipe = QwenImageEditPipeline.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.bfloat16)
            print("QwenImageEditPipeline Models loaded successfully!")
        else:
            pipe = QwenImagePipeline.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.bfloat16)
            print("QwenImagePipeline Models loaded successfully!")
        model_name = os.path.basename(pretrained_model_name_or_path).replace(".safetensors", "")
    else:
        pipe = QwenImageEditPlusPipeline.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.bfloat16)
        print("QwenImageEditPlusPipeline Models loaded successfully!")
        weight_name = os.path.basename(lora_path)
        model_name = "lora_" + weight_name.replace(".safetensors", "")
        pipe.load_lora_weights(lora_path, weight_name=weight_name)
    pipe = pipe.to(device)
    return pipe, model_name
    # args, pipe, filename, guidance_scale, output_dir, device


def calculate_dimensions(target_area: int, ratio: float) -> Tuple[int, int]:
    w = math.sqrt(target_area * ratio)
    h = w / ratio
    w = round(w / 32) * 32
    h = round(h / 32) * 32
    return int(w), int(h)

def _make_masked_image(tar_image: Image, tar_mask_l: Image) -> Image:
    """根据 tar_mask 把 tar_image 打洞（洞处置 0）。"""
    if tar_image.mode != "RGB":
        tar_image = tar_image.convert("RGB")
    img = np.asarray(tar_image).copy()
    m = np.asarray(tar_mask_l) > 0
    img[m] = 0
    return Image.fromarray(img, mode="RGB")


def _ensure_l_mode_binary(img: Image, threshold: int = 128) -> Image:
    """确保 L 单通道，并按阈值二值化为 {0,255}。"""
    if img.mode != "L":
        img = img.convert("L")
    arr = np.asarray(img)
    arr = (arr >= threshold).astype(np.uint8) * 255
    return Image.fromarray(arr, mode="L")

def crop_white_instance(mask_image, paste_image):
    """从输入图像中根据mask裁剪出白色背景的实例"""
    from PIL import ImageOps
    
    object_bbox = mask_image.getbbox()
    if object_bbox is None:
        # 如果mask为空，返回空白图像
        return Image.new('RGB', (1, 1), (0, 0, 0))
    
    inverted_mask = ImageOps.invert(mask_image)
    white_background = inverted_mask.point(lambda x: 255 if x == 255 else 0)
    paste_clip_image = Image.composite(white_background, paste_image, inverted_mask)
    paste_clip_image = paste_clip_image.crop(object_bbox)
    
    # 转换为正方形
    width, height = paste_clip_image.size
    square_size = max(width, height)
    square_image = Image.new('RGB', (square_size, square_size), (0, 0, 0))
    
    x_offset = (square_size - width) // 2
    y_offset = (square_size - height) // 2
    
    square_image.paste(paste_clip_image, (x_offset, y_offset))
    return square_image


def _process_single_image(args, pipe, filename, true_cfg_scale,num_inference_steps, output_dir, device):
    """处理单张图像"""
    
    # # 构建文件路径
    # if "EnsText" in args.input_dir:
    #     # 构建文件路径
    #     input_path = os.path.join(args.input_dir, "all_images", filename)
        
    #     clean_path = input_path.replace("all_images", "all_labels")
    #     # /home/v-qinhyang/code/hero_blob/DATASET/SynthText/SCUT-EnsText_test/enhanced_masks/1_final_mask.png
    #     mask_path = input_path.replace("all_images", "enhanced_masks").replace(".jpg", "_final_mask.png")
    # elif "syn_test" in args.input_dir:
    #     input_path = os.path.join(args.input_dir, "img", filename)
    #     clean_path = input_path.replace("/img/", "/label/")   
    #     # /home/v-qinhyang/code/hero_blob/DATASET/SynthText/Syn-Text/syn_test/img/img_0.png
    #     mask_path = input_path.replace("/img/", "/rect_mask/")
    #     # /home/v-qinhyang/code/hero_blob/DATASET/SynthText/Syn-Text/syn_test/rect_mask/img_0.png
    # elif "bench" in args.input_dir:
    #     input_path = os.path.join(args.input_dir, "input", filename)
    #     clean_path = input_path.replace("input", "clean")   
    #     mask_path = input_path.replace("input", "rect_mask")     
    if "EnsText" in args.input_dir:
        input_path = os.path.join(args.input_dir, "all_images", filename)
        clean_path = input_path.replace("all_images", "all_labels")
        mask_path = input_path.replace("all_images", "enhanced_masks").replace(".jpg", "_final_mask.png")
    elif "syn_test" in args.input_dir:
        input_path = os.path.join(args.input_dir, "img", filename)
        clean_path = input_path.replace("/img/", "/label/")   
        mask_path = input_path.replace("/img/", "/rect_mask/")
    elif "AnyTE" in args.input_dir or "bench" in args.input_dir:
        input_path = os.path.join(args.input_dir, "input", filename)
        clean_path = input_path.replace("input", "clean")   
        mask_path = input_path.replace("input", "mask")
    else:
        # 默认情况
        input_path = os.path.join(args.input_dir, "input", filename)
        clean_path = os.path.join(args.input_dir, "clean", filename)
        mask_path = os.path.join(args.input_dir, "mask", filename)
        print(f"Warning: Unknown dataset type for {filename}, using default paths")

    # 加载图像
    try:
        input_image = Image.open(input_path).convert('RGB')
        mask_image = Image.open(mask_path).convert('L')
        clean_image = Image.open(clean_path).convert('RGB')
    except Exception as e:
        print(f"Failed to load images for {filename}: {e}")
        return None
    mask_image = _ensure_l_mode_binary(mask_image)
    masked_image = _make_masked_image(input_image, mask_image)
    input_image, _, _ = extract_masked_region_with_mask(
                    ref_image=input_image,
                    ref_mask=mask_image,
                    align_to=32,
                    padding=0
                )    

    output_type = "pil"
    W, H = clean_image.size
    calculated_width, calculated_height = calculate_dimensions(1024 * 1024, W / H)
    height =  calculated_height
    width = calculated_width

    W, H = clean_image.size
    cond_w, cond_h = calculate_dimensions(CONDITION_IMAGE_AREA, W / H)
    vae_w, vae_h   = calculate_dimensions(VAE_IMAGE_AREA,       W / H)


    batch_size = 1
    # Load VL Image
    # ---- 文本编码（带图条件）：建议使用 masked + ref 两张作为 VL 条件 ----
    if "004" in args.model_name: 
        pasted_image = crop_white_instance(mask_image, input_image)
        cond_images = [
            pipe.image_processor.resize(input_image, cond_h, cond_w),
            pipe.image_processor.resize(pasted_image,    cond_h, cond_w),
        ]
        prompt_embeds, prompt_mask = pipe.encode_prompt(
                prompt=["remove the text in Picture 2"],
                image=cond_images,
                device=pipe._execution_device,
                num_images_per_prompt=1,
                max_sequence_length=512,
            )
    elif "005" in args.model_name:
        cond_images = [
            pipe.image_processor.resize(masked_image, cond_h, cond_w),
        ]
        prompt_embeds, prompt_mask = pipe.encode_prompt(
                prompt=["remove all the text"],
                image=cond_images,
                device=pipe._execution_device,
                num_images_per_prompt=1,
                max_sequence_length=512,
            )        
    else:
        cond_images = [
            pipe.image_processor.resize(input_image, cond_h, cond_w),
        ]
        # remove all the text
        prompt_embeds, prompt_mask = pipe.encode_prompt(
                prompt=["remove all the text"],
                image=cond_images,
                device=pipe._execution_device,
                num_images_per_prompt=1,
                max_sequence_length=512,
            )        
    do_true_cfg = true_cfg_scale > 1
    if do_true_cfg:
        negative_prompt_embeds, negative_prompt_embeds_mask = pipe.encode_prompt(
            image=cond_images,
            prompt=[" "],  # 空文本作为 negative prompt
            device=pipe._execution_device,
            num_images_per_prompt=1,
            max_sequence_length=512,
        )
    # ---- VAE 条件段（pack 成 token 后拼接在 token 维）----
    # masked, ref, tar_mask(转RGB)
    #     VAE             VL
    # 000 i d              remove all the text,i
    # 001 m_i,m  d         remove all the text,i
    # 002 m_i             remove all the text,i
    # 003 i,m             remove all the text,i
    # 004 m_i,m  d         remove the text in Picture 2,i,paste
    # 005 m_i,m  d         remove all the text,m_i
    # baseline i  d        remove all the text,i
    # vae_tensors_5d: List[torch.Tensor] = []
    # if "000" in args.model_name or "Qwen-Image-Edit-2509" in args.model_name:
    #     for pil in (input_image):
    #         vae_tensors_5d.append(
    #             pipe.image_processor.preprocess(pil, vae_h, vae_w).unsqueeze(2)  # [1,3,1,H,W]
    #         )
    vae_image_sizes = []
    vae_images = []
    if "001" in args.model_name or "004" in args.model_name or "005" in args.model_name:
        for pil in (masked_image, mask_image.convert("RGB")):
            image_width, image_height = pil.size
            vae_width, vae_height = calculate_dimensions(VAE_IMAGE_AREA, image_width / image_height)
            vae_image_sizes.append((vae_width, vae_height))
            vae_images.append(pipe.image_processor.preprocess(pil, vae_height, vae_width).unsqueeze(2))
        # for pil in (masked_image, mask_image.convert("RGB")):
        #     vae_tensors_5d.append(
        #         pipe.image_processor.preprocess(pil, vae_h, vae_w).unsqueeze(2)  # [1,3,1,H,W]
        #     )

    elif "002" in args.model_name:
        for pil in (masked_image):
            image_width, image_height = pil.size
            vae_width, vae_height = calculate_dimensions(VAE_IMAGE_AREA, image_width / image_height)
            vae_image_sizes.append((vae_width, vae_height))
            vae_images.append(pipe.image_processor.preprocess(pil, vae_height, vae_width).unsqueeze(2))
            # vae_tensors_5d.append(
            #     pipe.image_processor.preprocess(pil, vae_h, vae_w).unsqueeze(2)  # [1,3,1,H,W]
            # )

    elif "003" in args.model_name:
        for pil in (input_image, mask_image.convert("RGB")):
            image_width, image_height = pil.size
            vae_width, vae_height = calculate_dimensions(VAE_IMAGE_AREA, image_width / image_height)
            vae_image_sizes.append((vae_width, vae_height))
            vae_images.append(pipe.image_processor.preprocess(pil, vae_height, vae_width).unsqueeze(2))
            # vae_tensors_5d.append(
            #     pipe.image_processor.preprocess(pil, vae_h, vae_w).unsqueeze(2)  # [1,3,1,H,W]
            # )
    
    else:
        for pil in (input_image):
            image_width, image_height = pil.size
            vae_width, vae_height = calculate_dimensions(VAE_IMAGE_AREA, image_width / image_height)
            vae_image_sizes.append((vae_width, vae_height))
            vae_images.append(pipe.image_processor.preprocess(pil, vae_height, vae_width).unsqueeze(2))
            # vae_tensors_5d.append(
            #     pipe.image_processor.preprocess(pil, vae_h, vae_w).unsqueeze(2)  # [1,3,1,H,W]
            # )        

    # 4. Prepare latent variables
    num_channels_latents = pipe.transformer.config.in_channels // 4
    latents, image_latents = pipe.prepare_latents(
        vae_images,
        batch_size ,
        num_channels_latents,
        vae_h,
        vae_w, 
        prompt_embeds.dtype,
        device,
        generator=None,
        latents=None,
    )
    img_shapes = [
        [
            (1, height // pipe.vae_scale_factor // 2, width // pipe.vae_scale_factor // 2),
            *[
                (1, vae_height // pipe.vae_scale_factor // 2, vae_width // pipe.vae_scale_factor // 2)
                for vae_width, vae_height in vae_image_sizes
            ],
        ]
    ] * batch_size
    # 5. Prepare timesteps
    sigmas = None
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        pipe.scheduler.config.get("base_image_seq_len", 256),
        pipe.scheduler.config.get("max_image_seq_len", 4096),
        pipe.scheduler.config.get("base_shift", 0.5),
        pipe.scheduler.config.get("max_shift", 1.15),
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        pipe.scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        mu=mu,
    )
    num_warmup_steps = max(len(timesteps) - num_inference_steps * pipe.scheduler.order, 0)
    pipe._num_timesteps = len(timesteps)
    # handle guidance
    guidance_scale = None
    if pipe.transformer.config.guidance_embeds and guidance_scale is None:
        raise ValueError("guidance_scale is required for guidance-distilled model.")
    elif pipe.transformer.config.guidance_embeds:
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
        guidance = guidance.expand(latents.shape[0])
    elif not pipe.transformer.config.guidance_embeds and guidance_scale is not None:
        print("Warning: guidance_scale is ignored since the model is not guidance-distilled.")
        guidance = None
    elif not pipe.transformer.config.guidance_embeds and guidance_scale is None:
        guidance = None

    if pipe.attention_kwargs is None:
        pipe._attention_kwargs = {}

    txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist() if prompt_embeds_mask is not None else None
    negative_txt_seq_lens = (
        negative_prompt_embeds_mask.sum(dim=1).tolist() if negative_prompt_embeds_mask is not None else None
    )

    # 6. Denoising loop
    pipe.scheduler.set_begin_index(0)
    with pipe.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if pipe.interrupt:
                continue

            pipe._current_timestep = t

            latent_model_input = latents
            if image_latents is not None:
                latent_model_input = torch.cat([latents, image_latents], dim=1)
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latents.shape[0]).to(latents.dtype)
            with pipe.transformer.cache_context("cond"):
                noise_pred = pipe.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    encoder_hidden_states=prompt_embeds,
                    img_shapes=img_shapes,
                    txt_seq_lens=txt_seq_lens,
                    attention_kwargs=pipe.attention_kwargs,
                    return_dict=False,
                )[0]
                noise_pred = noise_pred[:, : latents.size(1)]

            if do_true_cfg:
                with pipe.transformer.cache_context("uncond"):
                    neg_noise_pred = pipe.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        encoder_hidden_states_mask=negative_prompt_embeds_mask,
                        encoder_hidden_states=negative_prompt_embeds,
                        img_shapes=img_shapes,
                        txt_seq_lens=negative_txt_seq_lens,
                        attention_kwargs=pipe.attention_kwargs,
                        return_dict=False,
                    )[0]
                neg_noise_pred = neg_noise_pred[:, : latents.size(1)]
                comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                noise_pred = comb_pred * (cond_norm / noise_norm)

            # compute the previous noisy sample x_t -> x_t-1
            latents_dtype = latents.dtype
            latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if latents.dtype != latents_dtype:
                if torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                    latents = latents.to(latents_dtype)

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipe.scheduler.order == 0):
                progress_bar.update()


    pipe._current_timestep = None

    latents = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
    latents = latents.to(pipe.vae.dtype)
    latents_mean = (
        torch.tensor(pipe.vae.config.latents_mean)
        .view(1, pipe.vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(
        latents.device, latents.dtype
    )
    latents = latents / latents_std + latents_mean
    decoded_image = pipe.vae.decode(latents, return_dict=False)[0][:, :, 0]
    image = pipe.image_processor.postprocess(decoded_image, output_type=output_type)

    # Offload all models
    pipe.maybe_free_model_hooks()

    # 生成结果
    result_image = image[0] if isinstance(image, list) else image
    
    # 保存结果
    result_image.save(os.path.join(output_dir, filename))
    
    # 创建拼接图像
    _create_concat_image(
        input_image, result_image, clean_image, mask_image,
        filename, output_dir
    )
    
    # 计算评估指标
    return _calculate_metrics(result_image, clean_image, filename)


def _create_concat_image(input_image, result_image, clean_image, 
                        mask_image, filename, output_dir):
    """创建拼接图像"""
    target_size = result_image.size
    
    # 调整所有图像大小
    input_resized = input_image.resize(target_size, Image.LANCZOS)
    clean_resized = clean_image.resize(target_size, Image.LANCZOS)
    mask_resized = mask_image.resize(target_size, Image.NEAREST)
    
    # 创建拼接图像
    concat_width = target_size[0] * 4
    concat_height = target_size[1]
    concat_image = Image.new('RGB', (concat_width, concat_height))
    
    # 粘贴图像
    concat_image.paste(input_resized, (0, 0))
    concat_image.paste(result_image, (target_size[0], 0))
    concat_image.paste(clean_resized, (target_size[0] * 2, 0))
    concat_image.paste(mask_resized, (target_size[0] * 3, 0))
    
    # 保存拼接图像
    concat_path = os.path.join(output_dir, "concat_result", f"concat_{filename}")
    concat_image.save(concat_path)

def _calculate_metrics(result_image, clean_image, filename):
    """计算评估指标"""
    import math
    from scipy import signal, ndimage
    from skimage.metrics import structural_similarity as ssim_skimage
    
    # 转换为numpy数组 [0, 1]
    result_array = np.array(result_image).astype(np.float32) / 255.0
    clean_array = np.array(clean_image).astype(np.float32) / 255.0
    
    # 确保尺寸一致
    if result_array.shape != clean_array.shape:
        height, width = result_array.shape[:2]
        clean_image_resized = clean_image.resize((width, height), Image.LANCZOS)
        clean_array = np.array(clean_image_resized).astype(np.float32) / 255.0
    
    # 1. MSE计算
    mse_score = np.mean((clean_array - result_array) ** 2)
    
    # 2. PSNR计算
    if mse_score == 0:
        psnr_score = float('inf')
    else:
        psnr_score = 10 * math.log10(1.0 / mse_score)
    
    # 3. 转换为灰度图用于SSIM计算
    def rgb_to_gray(img):
        return 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    
    clean_gray = rgb_to_gray(clean_array)
    result_gray = rgb_to_gray(result_array)
    
    # 4. 标准SSIM计算（使用scikit-image）
    ssim_score = ssim_skimage(clean_gray, result_gray, data_range=1.0)
    
    # 5. MSSIM计算
    def fspecial_gauss(size, sigma):
        x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
        g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
        return g/g.sum()

    def ssim_single(img1, img2, cs_map=False):
        img1 = img1.astype(float)
        img2 = img2.astype(float)

        size = min(img1.shape[0], 11)
        sigma = 1.5
        window = fspecial_gauss(size, sigma)
        K1 = 0.01
        K2 = 0.03
        L = 255  # bitdepth of image
        C1 = (K1 * L) ** 2
        C2 = (K2 * L) ** 2
        
        mu1 = signal.fftconvolve(img1, window, mode='valid')
        mu2 = signal.fftconvolve(img2, window, mode='valid')
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = signal.fftconvolve(img1 * img1, window, mode='valid') - mu1_sq
        sigma2_sq = signal.fftconvolve(img2 * img2, window, mode='valid') - mu2_sq
        sigma12 = signal.fftconvolve(img1 * img2, window, mode='valid') - mu1_mu2
        
        if cs_map:
            return (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)), 
                    (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
        else:
            return ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                        (sigma1_sq + sigma2_sq + C2))

    def msssim_single(img1, img2):
        level = 5
        weight = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])

        mssim = np.array([])
        mcs = np.array([])
        for l in range(level):
            if l > 0:
                # 下采样
                img1 = ndimage.zoom(img1, 0.5, order=1)
                img2 = ndimage.zoom(img2, 0.5, order=1)
            
            ssim_map, cs_map = ssim_single(img1, img2, cs_map=True)
            mssim = np.append(mssim, ssim_map.mean())
            mcs = np.append(mcs, cs_map.mean())

        # 处理可能的负值或零值
        sign_mcs = np.sign(mcs[0: level - 1])
        sign_mssim = np.sign(mssim[level - 1])
        
        # 避免对负数或零取幂
        abs_mcs = np.abs(mcs[0: level - 1])
        abs_mssim = np.abs(mssim[level - 1])
        
        # 处理零值
        abs_mcs = np.maximum(abs_mcs, 1e-10)
        abs_mssim = max(abs_mssim, 1e-10)
        
        mcs_power = np.power(abs_mcs, weight[0: level - 1])
        mssim_power = np.power(abs_mssim, weight[level - 1])
        
        result = np.prod(sign_mcs * mcs_power) * sign_mssim * mssim_power
        return max(result, 0.0)  # 确保结果非负
    
    # 计算MSSIM (转换为255范围)
    clean_gray_255 = (clean_gray * 255).astype(np.uint8)
    result_gray_255 = (result_gray * 255).astype(np.uint8)
    mssim_score = msssim_single(clean_gray_255, result_gray_255)
    
    # 6. AGE计算 (Average Gray Error)
    diff = np.abs(result_gray_255.astype(np.float32) - clean_gray_255.astype(np.float32))
    age_score = np.mean(diff)
    
    # 7. pEPs计算 (percentage of Error Pixels)
    threshold = 20
    errors = diff > threshold
    eps = np.sum(errors).astype(float)
    total_pixels = float(errors.shape[0] * errors.shape[1])
    peps_score = eps / total_pixels
    
    # 8. pCEPs计算 (percentage of Clustered Error Pixels)
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    eroded_errors = ndimage.binary_erosion(errors, structure).astype(errors.dtype)
    ceps = np.sum(eroded_errors)
    pceps_score = ceps / total_pixels
    
    print(f"Image {filename}: PSNR={psnr_score:.2f}, SSIM={ssim_score:.4f}, MSSIM={mssim_score:.4f}, MSE={mse_score:.6f}, "
        f"AGE={age_score:.2f}, pEPs={peps_score:.4f}, pCEPs={pceps_score:.4f}")
    
    return {
        "image": filename,
        "psnr": float(psnr_score) if psnr_score != float('inf') else 100.0,  # 设置上限避免inf
        "ssim": float(ssim_score),  # 添加标准SSIM
        "mssim": float(mssim_score),
        "mse": float(mse_score),
        "age": float(age_score),
        "peps": float(peps_score),
        "pceps": float(pceps_score)
    }

def _save_intermediate_results(output_dir, metrics, model_name, 
                            guidance_scale, current_count, total_count):
    """保存中间结果"""
    if not metrics.get("psnr"):
        return
    
    # 计算各指标平均值
    avg_psnr = np.mean(metrics["psnr"])
    avg_mssim = np.mean(metrics["mssim"])
    avg_mse = np.mean(metrics["mse"])
    avg_age = np.mean(metrics["age"])
    avg_peps = np.mean(metrics["peps"])
    avg_pceps = np.mean(metrics["pceps"])
    
    print(f"Intermediate results ({current_count}/{total_count}): "
        f"PSNR={avg_psnr:.2f}, MSSIM={avg_mssim:.4f}, MSE={avg_mse:.6f}, "
        f"AGE={avg_age:.2f}, pEPs={avg_peps:.4f}, pCEPs={avg_pceps:.4f}")
    
    intermediate_results = {
        "status": "intermediate",
        "processed_count": current_count,
        "total_count": total_count,
        "progress_percentage": (current_count / total_count) * 100,
        "current_average_psnr": float(avg_psnr),
        "current_average_mssim": float(avg_mssim),
        "current_average_mse": float(avg_mse),
        "current_average_age": float(avg_age),
        "current_average_peps": float(avg_peps),
        "current_average_pceps": float(avg_pceps),
        "individual_scores": metrics["individual"],
        "timestamp": str(datetime.now())
    }
    
    intermediate_file = os.path.join(
        output_dir, f"intermediate_results_{current_count}items.json"
    )
    with open(intermediate_file, 'w') as f:
        json.dump(intermediate_results, f, indent=2)

def _save_final_results(args , output_dir, metrics, model_name, guidance_scale):
    """保存最终结果"""
    if not metrics.get("psnr"):
        print("No valid results to save")
        return
        
    # 计算FID和LOCAL-FID
    # 根据数据集类型设置路径
    if "EnsText" in args.input_dir:
        clean_images_path = os.path.join(args.input_dir, "all_labels")
        mask_images_path = os.path.join(args.input_dir, "orimask")
    elif "syn_test" in args.input_dir:
        clean_images_path = os.path.join(args.input_dir, "label")
        mask_images_path = os.path.join(args.input_dir, "rect_mask")
    elif "AnyTE" in args.input_dir or "bench" in args.input_dir:
        clean_images_path = os.path.join(args.input_dir, "clean")
        mask_images_path = os.path.join(args.input_dir, "mask")
    else:
        clean_images_path = None
        mask_images_path = None
    generated_images_path = output_dir
    fid_score = None
    local_fid_score = None
    # 计算各指标平均值
    avg_psnr = np.mean(metrics["psnr"])
    avg_mssim = np.mean(metrics["mssim"])
    avg_mse = np.mean(metrics["mse"])
    avg_age = np.mean(metrics["age"])
    avg_peps = np.mean(metrics["peps"])
    avg_pceps = np.mean(metrics["pceps"])
    
    print(f"Final results: PSNR={avg_psnr:.2f}, MSSIM={avg_mssim:.4f}, MSE={avg_mse:.6f}, "
        f"AGE={avg_age:.2f}, pEPs={avg_peps:.4f}, pCEPs={avg_pceps:.4f}")
    
    # 修复：检查FID分数是否为None
    if fid_score is not None and local_fid_score is not None:
        print(f"FID Score: {fid_score:.4f}, LOCAL-FID Score: {local_fid_score:.4f}")
    elif fid_score is not None:
        print(f"FID Score: {fid_score:.4f}, LOCAL-FID Score: N/A")
    elif local_fid_score is not None:
        print(f"FID Score: N/A, LOCAL-FID Score: {local_fid_score:.4f}")
    else:
        print("FID Score: N/A, LOCAL-FID Score: N/A")        
    
    final_results = {
        "status": "completed",
        "total_processed": len(metrics["individual"]),
        "average_psnr": float(avg_psnr),
        "average_mssim": float(avg_mssim),
        "average_mse": float(avg_mse),
        "average_age": float(avg_age),
        "average_peps": float(avg_peps),
        "average_pceps": float(avg_pceps),
        "fid_score": fid_score if fid_score is not None else "N/A",
        "local_fid_score": local_fid_score if local_fid_score is not None else "N/A",
        "individual_scores": metrics["individual"],
        "model": model_name,
        "guidance_scale": guidance_scale,
        "timestamp": str(datetime.now())
    }
    
    final_file = os.path.join(
        output_dir, f"evaluation_results_{model_name}_{guidance_scale}_final.json"
    )
    with open(final_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # 保存CSV格式 - 包含所有指标
    _save_csv_results(output_dir, metrics["individual"], model_name, guidance_scale)
    
    print(f"Results saved to:")
    print(f"  JSON: {final_file}")
    print(f"  CSV: {os.path.join(output_dir, f'evaluation_results_{model_name}_{guidance_scale}.csv')}")

def _save_csv_results(output_dir, individual_scores, model_name, guidance_scale):
    """保存CSV格式结果"""
    csv_filename = os.path.join(
        output_dir, f"evaluation_results_{model_name}_{guidance_scale}.csv"
    )
    
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # 修改表头，添加MSSIM列
        writer.writerow(['IMGNAME', 'PSNR', 'SSIM', 'MSSIM', 'MSE', 'AGE', 'pEPs', 'pCEPs'])
        
        for score in individual_scores:
            writer.writerow([
                score['image'],  # IMGNAME
                f"{score['psnr']:.2f}",      # PSNR
                f"{score.get('ssim', 0.0):.4f}",  # SSIM (如果有的话)
                f"{score['mssim']:.4f}",     # MSSIM
                f"{score['mse']:.6f}",       # MSE
                f"{score['age']:.2f}",       # AGE
                f"{score['peps']:.4f}",      # pEPs
                f"{score['pceps']:.4f}"      # pCEPs
            ])
    
    print(f"Individual image metrics saved to: {csv_filename}")



def run_test(args, model_name, pipe: QwenImageEditPlusPipeline, device):
    # 创建输出目录
    guidance_scale = args.guidance_scale
    output_dir = args.output_dir
    num_inference_steps = args.num_inference_steps
    output_dir = os.path.join(
        args.output_dir, model_name,
        f"GS_{guidance_scale}"
    )
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "concat_result"), exist_ok=True)
    # 检查是否已经完成
    result_file = os.path.join(
        output_dir, f"evaluation_results_{model_name}_{guidance_scale}_final.json"
    )
    if os.path.exists(result_file):
        print(f"Results already exist for {output_dir}, skipping...")
        return
    # 获取测试图像
    if 'syn_test' in args.input_dir:
        images_dir = os.path.join(args.input_dir, "img")
    elif "EnsText" in args.input_dir:
        images_dir = os.path.join(args.input_dir, "all_images")
    elif "AnyTE-all" in args.input_dir:
        images_dir = os.path.join(args.input_dir, "input")
    else:
        # 默认使用 input_dir 本身
        images_dir = os.path.join(args.input_dir, "input")
        print(f"Warning: Unknown dataset type, using {images_dir} as images directory")
    filenames = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png'))]
    print(f"Output directory: {output_dir}")
    todolist = [file for file in filenames if not os.path.exists(os.path.join(output_dir, file))]
    print(f"Processing {len(todolist)} images with guidance_scale={guidance_scale}")

    # 评估指标
    metrics = {
        "psnr": [], "mssim": [], "mse": [], 
        "age": [], "peps": [], "pceps": [], 
        "individual": []
    }
        
    for i, filename in enumerate(todolist):
        try:
            result = _process_single_image(
                args, pipe, filename, guidance_scale, num_inference_steps, output_dir, device
            )
          
            if result is not None:
                metrics["psnr"].append(result["psnr"])
                metrics["mssim"].append(result["mssim"])
                metrics["mse"].append(result["mse"])
                metrics["age"].append(result["age"])
                metrics["peps"].append(result["peps"])
                metrics["pceps"].append(result["pceps"])
                metrics["individual"].append(result)

            # 每400张图像保存一次中间结果
            if (i + 1) % 400 == 0:
                _save_intermediate_results(
                    output_dir, metrics, model_name,
                    guidance_scale, i + 1, len(todolist)
                )
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    # 保存最终结果
    _save_final_results(
        args, output_dir, metrics, model_name, guidance_scale
    )




def main(args):
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    pipe, model_name = load_models(
            pretrained_model_name_or_path=args.model_path,
            lora_path=args.lora_path,
            device=device
        )
    args.model_name = model_name
    # 冻结 VAE & 文本编码器，只训 transformer 的 LoRA
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)


    run_test(args, model_name, pipe, device)


# -------------------- CLI --------------------
def parse_args():
    p = argparse.ArgumentParser("Qwen Image Text Erasing Benchmark")
    # 模型 & 数据
    p.add_argument("--blob_dir", type=str, default="/home/v-qinhyang/code/hero_blob",
                       help="Directory containing model checkpoints")
    p.add_argument("--model_path", type=str, default="/home/v-qinhyang/code/hero_blob/preweight/Qwen/Qwen/Qwen-Image-Edit-2509",
                   help="你的 Qwen-Image-Edit-2509 路径或HF模型名（需为你已修改过的 pipeline 版本）")
    p.add_argument("--lora_path", type=str, default=None,
                   help="LoRA 权重路径（如果有的话）")
    p.add_argument("--input_dir", type=str, default="/home/v-qinhyang/code/hero_blob/DATASET/SynthText/AnyTE_bench/AnyTE-all/complex_background",
                   help="Directory containing test images")
    p.add_argument("--output_dir", type=str, default="/home/v-qinhyang/code/hero_blob/TESTRESULTS/qwen_baselines/Qwen-Image-Edit2509_rectmask_remove_ours_complex_background",
                   help="Output directory for test results")
    p.add_argument("--guidance_scale", type=float, nargs="+", default=[5.0],
                   help="List of guidance scales to test")
    p.add_argument("--strength", type=float, default=0.9999,
                   help="Strength of the text erasing")
    p.add_argument("--num_inference_steps", type=int, default=50,
                   help="Number of inference steps")
    p.add_argument("--dilate_kernel", type=int, default=5,
                   help="Dilate kernel size")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
