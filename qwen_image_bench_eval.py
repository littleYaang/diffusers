
from datetime import datetime
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import random
from PIL import Image, ImageDraw, ImageOps
import csv
import re
import torch
from PIL import Image
from diffusers import QwenImageEditPipeline,QwenImagePipeline,QwenImageEditPlusPipeline
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
import torch
from torchvision import transforms
from PIL import Image
import os
# code/dm/qwenimage/diffusers/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit_plus_inpaint.py
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus_inpaint import QwenImageEditPlusInpaintPipeline

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

def preprocess_images_with_padding(input_image, clean_image, mask_image, target_long_side=1024):
    """
    预处理图像：缩放到长边为1024，短边用黑色填充
    
    Args:
        input_image: PIL图像
        clean_image: PIL图像  
        mask_image: PIL图像
        target_long_side: 目标长边尺寸
    
    Returns:
        processed_images: 处理后的图像字典
        original_info: 原始图像信息，用于后处理
    """
    # 获取原始尺寸
    original_size = input_image.size  # (width, height)
    original_width, original_height = original_size
    
    # 计算缩放比例
    scale = target_long_side / max(original_width, original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    # 缩放图像
    input_resized = input_image.resize((new_width, new_height), Image.LANCZOS)
    clean_resized = clean_image.resize((new_width, new_height), Image.LANCZOS)
    mask_resized = mask_image.resize((new_width, new_height), Image.NEAREST)
    
    # 计算填充量
    pad_width = target_long_side - new_width
    pad_height = target_long_side - new_height
    
    # 计算左右和上下填充
    left_pad = pad_width // 2
    right_pad = pad_width - left_pad
    top_pad = pad_height // 2
    bottom_pad = pad_height - top_pad
    
    # 创建填充后的图像
    # 对于input_image和clean_image使用黑色填充
    input_padded = Image.new('RGB', (target_long_side, target_long_side), (0, 0, 0))
    clean_padded = Image.new('RGB', (target_long_side, target_long_side), (0, 0, 0))
    # 对于mask_image使用黑色填充（0值）
    mask_padded = Image.new('L', (target_long_side, target_long_side), 0)
    
    # 粘贴缩放后的图像到中心位置
    input_padded.paste(input_resized, (left_pad, top_pad))
    clean_padded.paste(clean_resized, (left_pad, top_pad))
    mask_padded.paste(mask_resized, (left_pad, top_pad))
    
    # 保存处理信息
    processing_info = {
        'original_size': original_size,
        'scale': scale,
        'new_size': (new_width, new_height),
        'padding': (left_pad, top_pad, right_pad, bottom_pad),
        'target_size': (target_long_side, target_long_side)
    }
    
    processed_images = {
        'input_image': input_padded,
        'clean_image': clean_padded,
        'mask_image': mask_padded
    }
    
    return processed_images, processing_info

def postprocess_result_image(result_image, processing_info):
    """
    后处理结果图像：去除黑色填充并恢复到原始尺寸
    
    Args:
        result_image: 生成的结果图像
        processing_info: 预处理时保存的信息
    
    Returns:
        final_image: 恢复到原始尺寸的图像
    """
    # 获取处理信息
    original_size = processing_info['original_size']
    scale = processing_info['scale']
    new_size = processing_info['new_size']
    left_pad, top_pad, right_pad, bottom_pad = processing_info['padding']
    target_size = processing_info['target_size']
    
    # 裁剪掉填充部分
    crop_box = (
        left_pad,
        top_pad,
        target_size[0] - right_pad,
        target_size[1] - bottom_pad
    )
    
    cropped_image = result_image.crop(crop_box)
    
    # 缩放回原始尺寸
    final_image = cropped_image.resize(original_size, Image.LANCZOS)
    
    return final_image

def detect_and_crop_black_padding(image, threshold=10):
    """
    检测并裁剪图像中的黑色填充区域
    
    Args:
        image: PIL图像
        threshold: 黑色检测阈值
    
    Returns:
        cropped_image: 裁剪后的图像
        crop_info: 裁剪信息
    """
    # 转换为numpy数组
    img_array = np.array(image)
    
    # 如果是灰度图像，转换为3通道
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    
    # 检测非黑色像素
    non_black_mask = np.any(img_array > threshold, axis=-1)
    
    # 找到非黑色区域的边界
    rows = np.any(non_black_mask, axis=1)
    cols = np.any(non_black_mask, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        # 如果整个图像都是黑色，返回原图像
        return image, None
    
    # 找到边界
    top = np.argmax(rows)
    bottom = len(rows) - np.argmax(rows[::-1]) - 1
    left = np.argmax(cols)
    right = len(cols) - np.argmax(cols[::-1]) - 1
    
    # 裁剪图像
    crop_box = (left, top, right + 1, bottom + 1)
    cropped_image = image.crop(crop_box)
    
    crop_info = {
        'original_size': image.size,
        'crop_box': crop_box,
        'cropped_size': cropped_image.size
    }
    
    return cropped_image, crop_info

class QwenTextEraserPipeline:
    """Qwen文本擦除测试管道"""
    
    def __init__(self, 
                 strength=0.9999,
                 num_inference_steps=30,
                 target_size=(1024, 1024),
                 dilate_kernel=5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.strength = strength
        # strengths = [0.6, 0.7, 0.8, 0.9, 1.0]
        self.num_inference_steps = num_inference_steps
        self.target_size = target_size
        self.dilate_kernel = dilate_kernel
        self.pipe = None


    def load_models(self, 
                   pretrained_model_name_or_path="/home/v-qinhyang/code/hero_blob/preweight/Qwen/Qwen-Image",
                   ):
        """加载所有必要的模型组件"""
        print(f"Loading models from: {pretrained_model_name_or_path}")
        if "2509" in   pretrained_model_name_or_path:
            self.pipe = QwenImageEditPlusInpaintPipeline.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.bfloat16)
        elif "Edit" in pretrained_model_name_or_path:
            self.pipe = QwenImageEditPipeline.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.bfloat16)
            print("QwenImageEditPipeline Models loaded successfully!")
        else:
            self.pipe = QwenImagePipeline.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.bfloat16)
            print("QwenImagePipeline Models loaded successfully!")
        self.pipe = self.pipe.to(self.device)
        self.pipe.load_lora_weights("/home/v-qinhyang/code/hero_blob/qwenoutput/plain_lora/edit_plus_scut2749_lora_128d_5e-5_16ep.safetensors",weight_name="qwen-edit-remover.safetensors")
        

    def _dilate_mask(self, mask_image):
        """膨胀mask"""
        if self.dilate_kernel <= 0:
            return mask_image
            
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, 
            (int(self.dilate_kernel * 2 + 1), int(self.dilate_kernel * 2 + 1))
        )
        mask_array = cv2.dilate(np.array(mask_image), kernel)
        return Image.fromarray(mask_array)

    def generate(self, 
                input_image, 
                mask_image, 
                prompt="Remove the text of ",
                guidance_scale=5.0,
                truncation_trick="noise",):
        if self.dilate_kernel > 0:
            mask_image = self._dilate_mask(mask_image)
        ##
        paste_image = crop_white_instance(mask_image, input_image)
        

        result = self.pipe(
            prompt=prompt,
            negative_prompt=" ",
            height=input_image.size[1],
            width=input_image.size[0],
            image=input_image,
            control_image=paste_image,
            mask=mask_image,
            true_cfg_scale=guidance_scale,
            num_inference_steps=self.num_inference_steps,
            strength=1,
        )
        result = result.images[0]
        return result        

class QwenBenchmark:
    """Qwen基准测试类"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = QwenTextEraserPipeline(
            strength=args.strength,
            num_inference_steps=args.num_inference_steps,
            dilate_kernel=args.dilate_kernel
        )

        self.model_path  = args.model_path
        self.prompt = args.prompt
        # 调度器配置

    def run_benchmark(self):
        """运行基准测试"""
        # for model_name in self.args.model_names:
        #     model_path = self._find_latest_checkpoint(
        #         os.path.join(self.args.test_dirs, model_name)
        #     )
        # model_name = "qwen-image-inpainting-egtest"
        if "2509" in self.model_path :
            model_name = "qwen-image-edit-0925-lora-remover-egtest"
        elif "Edit" in self.model_path:
            model_name = "qwen-image-edit-lora-remover-egtest"
        else:
            model_name = "qwen-image-inpainting-egtest"
        model_path = self.model_path
        if model_path is None:
            print(f"Warning: No checkpoint found for {model_name}")
        
        print(f"Checkpoint: {model_path}")
        
        # 加载模型
        self.pipeline.load_models(
            model_path,
        )
        
       # 测试不同配置
        self._test_model_configurations(model_name, model_path)

    def _test_model_configurations(self, model_name, model_path):
        """测试模型的不同配置""" 
        for guidance_scale in self.args.guidance_scales:
            # for mask_type in self.args.mask_types:
            self._run_single_test(
                model_name, guidance_scale
            )


    def _run_single_test(self, model_name, guidance_scale):
        """运行单个测试配置"""
        
        # 创建输出目录
        output_dir = os.path.join(
            self.args.output_dir, model_name,
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
        # 获取测试图像
        if 'syn_test' in self.args.input_dir:
            images_dir = os.path.join(self.args.input_dir, "img")
        elif "EnsText" in self.args.input_dir:
            images_dir = os.path.join(self.args.input_dir, "all_images")
        elif "AnyTE-all" in self.args.input_dir:
            images_dir = os.path.join(self.args.input_dir, "input")
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
                result = self._process_single_image(
                    filename, guidance_scale, output_dir
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
                    self._save_intermediate_results(
                        output_dir, metrics, model_name,
                        guidance_scale, i + 1, len(todolist)
                    )
                    
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
        
        # 保存最终结果
        self._save_final_results(
            output_dir, metrics, model_name, guidance_scale
        )

    def _process_single_image(self, filename, guidance_scale, output_dir):
        """处理单张图像"""
        
        # 构建文件路径
        if "EnsText" in self.args.input_dir:
            # 构建文件路径
            input_path = os.path.join(self.args.input_dir, "all_images", filename)
            
            clean_path = input_path.replace("all_images", "all_labels")
            # /home/v-qinhyang/code/hero_blob/DATASET/SynthText/SCUT-EnsText_test/enhanced_masks/1_final_mask.png
            mask_path = input_path.replace("all_images", "enhanced_masks").replace(".jpg", "_final_mask.png")
        elif "syn_test" in self.args.input_dir:
            input_path = os.path.join(self.args.input_dir, "img", filename)
            clean_path = input_path.replace("/img/", "/label/")   
            # /home/v-qinhyang/code/hero_blob/DATASET/SynthText/Syn-Text/syn_test/img/img_0.png
            mask_path = input_path.replace("/img/", "/rect_mask/")
            # /home/v-qinhyang/code/hero_blob/DATASET/SynthText/Syn-Text/syn_test/rect_mask/img_0.png
        elif "AnyTE-all" in self.args.input_dir:
            input_path = os.path.join(self.args.input_dir, "input", filename)
            clean_path = input_path.replace("input", "clean")   
            mask_path = input_path.replace("input", "rect_mask")     

        # 加载图像
        try:
            input_image = Image.open(input_path).convert('RGB')
            mask_image = Image.open(mask_path).convert('L')
            clean_image = Image.open(clean_path).convert('RGB')
        except Exception as e:
            print(f"Failed to load images for {filename}: {e}")
            return None
            
        processed_images, processing_info = preprocess_images_with_padding(
            input_image, clean_image, mask_image, target_long_side=1024
        )
        input_image = processed_images['input_image']
        mask_image = processed_images['mask_image']
        clean_image = processed_images['clean_image']

        # 生成结果
        result_image = self.pipeline.generate(
            input_image=input_image,
            mask_image=mask_image,
            prompt=self.prompt,
            guidance_scale=guidance_scale,
        )
        result_image = postprocess_result_image(result_image, processing_info)
        
        # 保存结果
        result_image.save(os.path.join(output_dir, filename))
        
        # 创建拼接图像
        self._create_concat_image(
            input_image, result_image, clean_image, mask_image,
            filename, output_dir
        )
        
        # 计算评估指标
        return self._calculate_metrics(result_image, clean_image, filename)


    def _create_concat_image(self, input_image, result_image, clean_image, 
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

    def _calculate_metrics(self, result_image, clean_image, filename):
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

    def _save_intermediate_results(self, output_dir, metrics, model_name, 
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

    def _save_final_results(self, output_dir, metrics, model_name, guidance_scale):
        """保存最终结果"""
        if not metrics.get("psnr"):
            print("No valid results to save")
            return
            
        # 计算FID和LOCAL-FID
        clean_images_path = os.path.join(self.args.input_dir, "all_labels")
        generated_images_path = output_dir
        mask_images_path = os.path.join(self.args.input_dir, "orimask")   

        print("Calculating FID score...")
        fid_score = calculate_fid_score(clean_images_path, generated_images_path, self.device)
        
        print("Calculating LOCAL-FID score...")
        local_fid_score = calculate_local_fid_score(clean_images_path, generated_images_path, 
                                                mask_images_path, self.device)
        
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
        self._save_csv_results(output_dir, metrics["individual"], model_name, guidance_scale)
        
        print(f"Results saved to:")
        print(f"  JSON: {final_file}")
        print(f"  CSV: {os.path.join(output_dir, f'evaluation_results_{model_name}_{guidance_scale}.csv')}")

    def _save_csv_results(self, output_dir, individual_scores, model_name, guidance_scale):
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

    def _save_csv_results(self, output_dir, individual_scores, model_name, guidance_scale):
        """保存CSV格式结果"""
        csv_filename = os.path.join(
            output_dir, f"evaluation_results_{model_name}_{guidance_scale}.csv"
        )
        
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['FILENAME', 'PSNR', 'MSSIM', 'MSE', 'AGE', 'pEPs', 'pCEPs'])
            
            for score in individual_scores:
                writer.writerow([
                    score['image'],
                    f"{score['psnr']:.2f}",
                    f"{score['mssim']:.4f}",
                    f"{score['mse']:.6f}",
                    f"{score['age']:.2f}",
                    f"{score['peps']:.4f}",
                    f"{score['pceps']:.4f}"
                ])


# 添加FID计算的辅助函数
def calculate_fid_score(real_images_path, generated_images_path, device='cuda'):
    """
    使用torch-fidelity计算FID分数
    需要安装: pip install torch-fidelity
    """
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        import torch
        from torchvision import transforms
        from PIL import Image
        import os
        
        # 初始化FID计算器
        fid = FrechetInceptionDistance(feature=2048).to(device)
        
        # 图像预处理
        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 加载和处理真实图像
        real_images = []
        for img_name in os.listdir(real_images_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(real_images_path, img_name)
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0)
                real_images.append(img_tensor)
        
        # 加载和处理生成图像
        generated_images = []
        for img_name in os.listdir(generated_images_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(generated_images_path, img_name)
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0)
                generated_images.append(img_tensor)
        
        # 转换为批次
        real_batch = torch.cat(real_images).to(device)
        generated_batch = torch.cat(generated_images).to(device)
        
        # 更新FID计算器
        fid.update(real_batch, real=True)
        fid.update(generated_batch, real=False)
        
        # 计算FID分数
        fid_score = fid.compute()
        return float(fid_score)
        
    except ImportError:
        print("torch-fidelity not installed. Please install it with: pip install torch-fidelity")
        return None
    except Exception as e:
        print(f"Error calculating FID: {e}")
        return None

def calculate_local_fid_score(real_images_path, generated_images_path, mask_path, device='cuda'):
    """
    计算LOCAL-FID分数（仅在mask区域内计算）
    """
    try:

        
        # 初始化FID计算器
        fid = FrechetInceptionDistance(feature=2048).to(device)
        
        # 图像预处理
        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        def extract_masked_region(image_path, mask_path):
            """提取mask区域"""
            img = Image.open(image_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')
            
            # 确保尺寸一致
            img = img.resize(mask.size, Image.LANCZOS)
            
            # 找到mask的边界框
            bbox = mask.getbbox()
            if bbox is None:
                return None
            
            # 裁剪mask区域
            masked_img = img.crop(bbox)
            return masked_img
        
        # 处理真实图像
        real_images = []
        for img_name in os.listdir(real_images_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(real_images_path, img_name)
                mask_file = os.path.join(mask_path, img_name.replace('.jpg', '_mask.png'))
                
                if os.path.exists(mask_file):
                    masked_img = extract_masked_region(img_path, mask_file)
                    if masked_img is not None:
                        img_tensor = transform(masked_img).unsqueeze(0)
                        real_images.append(img_tensor)
        
        # 处理生成图像
        generated_images = []
        for img_name in os.listdir(generated_images_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(generated_images_path, img_name)
                mask_file = os.path.join(mask_path, img_name.replace('.jpg', '_mask.png'))
                
                if os.path.exists(mask_file):
                    masked_img = extract_masked_region(img_path, mask_file)
                    if masked_img is not None:
                        img_tensor = transform(masked_img).unsqueeze(0)
                        generated_images.append(img_tensor)
        
        if not real_images or not generated_images:
            print("No valid masked images found for LOCAL-FID calculation")
            return None
        
        # 转换为批次
        real_batch = torch.cat(real_images).to(device)
        generated_batch = torch.cat(generated_images).to(device)
        
        # 更新FID计算器
        fid.update(real_batch, real=True)
        fid.update(generated_batch, real=False)
        
        # 计算LOCAL-FID分数
        local_fid_score = fid.compute()
        return float(local_fid_score)
        
    except Exception as e:
        print(f"Error calculating LOCAL-FID: {e}")
        return None

# 保持原有的辅助函数
def crop_white_instance(mask_image, paste_image):
    """裁剪白色实例"""
    object_bbox = mask_image.getbbox()
    inverted_mask = ImageOps.invert(mask_image)
    white_background = inverted_mask.point(lambda x: 255 if x == 255 else 0)
    paste_clip_image = Image.composite(white_background, paste_image, inverted_mask)
    paste_clip_image = paste_clip_image.crop(object_bbox)
    return paste_clip_image

def resize_and_pad_white_bg(image, target_size):
    """调整大小并填充白色背景"""
    target_h, target_w = target_size

    if isinstance(image, np.ndarray):
        if len(image.shape) == 3:
            image = Image.fromarray(image)
        else:
            image = Image.fromarray(image).convert('RGB')

    w, h = image.size

    if h > w:
        scale = target_h / h
        new_h = target_h
        new_w = int(w * scale)
    elif w > h:
        scale = target_w / w
        new_w = target_w
        new_h = int(h * scale)
    else:
        scale = min(target_h / h, target_w / w)
        new_h = int(h * scale)
        new_w = int(w * scale)

    resized_image = image.resize((new_w, new_h), Image.LANCZOS)
    padded_image = Image.new('RGB', (target_w, target_h), (255, 255, 255))
    
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    padded_image.paste(resized_image, (paste_x, paste_y))
    
    return padded_image

def encode_prompt(prompt_batch, text_encoders, tokenizers, is_train=True):
    """编码提示词"""
    prompt_embeds_list = []
    captions = []
    
    for caption in prompt_batch:
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                captions, padding="max_length", max_length=tokenizer.model_max_length,
                truncation=True, return_tensors="pt"
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(
                text_input_ids.to(text_encoder.device), output_hidden_states=True
            )

            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds

def retrieve_timesteps(scheduler, num_inference_steps, device, timesteps=None, sigmas=None):
    """获取时间步"""
    assert timesteps is None and sigmas is None
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

def get_timesteps(scheduler, num_inference_steps, strength, device, denoising_start=None):
    """获取时间步"""
    if denoising_start is None:
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = scheduler.timesteps[t_start * scheduler.order :]
        if hasattr(scheduler, "set_begin_index"):
            scheduler.set_begin_index(t_start * scheduler.order)
        return timesteps, num_inference_steps - t_start
    else:
        discrete_timestep_cutoff = int(
            round(scheduler.config.num_train_timesteps - (denoising_start * scheduler.config.num_train_timesteps))
        )
        num_inference_steps = (scheduler.timesteps < discrete_timestep_cutoff).sum().item()
        if scheduler.order == 2 and num_inference_steps % 2 == 0:
            num_inference_steps = num_inference_steps + 1
        t_start = len(scheduler.timesteps) - num_inference_steps
        timesteps = scheduler.timesteps[t_start:]
        if hasattr(scheduler, "set_begin_index"):
            scheduler.set_begin_index(t_start)
        return timesteps, num_inference_steps

def retrieve_latents(encoder_output, generator=None, sample_mode="sample"):
    """获取latents"""
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Qwen Image Text Erasing Benchmark")
    parser.add_argument("--blob_dir", type=str, default="/home/v-qinhyang/code/hero_blob",
                       help="Directory containing model checkpoints")
    parser.add_argument("--prompt", type=str, default="remove the text of",
                       help="Text prompt for the inpainting task")
    parser.add_argument("--model_path", type=str, default="/home/v-qinhyang/code/hero_blob/preweight/Qwen/Qwen/Qwen-Image-Edit-2509",
                       help="model checkpoint path")
    parser.add_argument("--input_dir", type=str, default="/home/v-qinhyang/code/hero_blob/DATASET/SynthText/SCUT-EnsText_test",
                       help="Directory containing test images")
    parser.add_argument("--output_dir", type=str, default="/home/v-qinhyang/code/hero_blob/TESTRESULTS/qwen_baselines/Qwen-Image-Edit2509_scut2749lora_128_rectmask_remove",
                       help="Output directory for test results")
    parser.add_argument("--guidance_scales", type=float, nargs="+", default=[5.0],
                       help="List of guidance scales to test")
    # parser.add_argument("--mask_types", type=str, nargs="+", default=["orig", "bbox"],
    #                    help="List of mask types to test")
    parser.add_argument("--strength", type=float, default=0.9999,
                       help="Strength of the text erasing")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                       help="Number of inference steps")
    parser.add_argument("--dilate_kernel", type=int, default=5,
                       help="Dilate kernel size")
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    print("Starting Qwen Image Text Erasing Benchmark")
    print(f"Models to test: Qwen Image")
    print(f"Guidance scales: {args.guidance_scales}")
    # print(f"Mask types: {args.mask_types}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 运行基准测试
    benchmark = QwenBenchmark(args)
    benchmark.run_benchmark()
    
    print("Benchmark completed!")

if __name__ == "__main__":
    main()