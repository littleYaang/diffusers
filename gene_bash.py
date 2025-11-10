#!/usr/bin/env python3
# generate_eval_scripts.py

import os

# 配置
datasets = {
    "scut": {
        "input_dir": "/mnt/hero_blob/DATASET/SynthText/SCUT-EnsText_test",
        "gpu": 0,
        "name": "scut"
    },
    "syn": {
        "input_dir": "/mnt/hero_blob/DATASET/SynthText/Syn-Text/syn_test",
        "gpu": 1,
        "name": "syn"
    },
    "laion": {
        "input_dir": "/mnt/hero_blob/DATASET/SynthText/AnyTE_bench/laion_bench/all",
        "gpu": 2,
        "name": "laion"
    },
    "prim": {
        "input_dir": "/mnt/hero_blob/DATASET/SynthText/PrismLayersPro/paired_data/all_text",
        "gpu": 3,
        "name": "PrismLayersPro"
    }
}

model_path = "/mnt/hero_blob/preweight/Qwen/Qwen/Qwen-Image-Edit-2509"
lora_template = "/home/v-qinhyang/code/hero_blob/qwenoutput/plain_lora/edit_plus_ablation_ens_syn_{:03d}_lora_128d_5e-5_16ep-000004.safetensors"
output_base = "/mnt/hero_blob/TESTRESULTS/qwen_baselines"

# 创建输出目录
os.makedirs("generated_scripts", exist_ok=True)

# 生成20个脚本
# for model_id in range(1, 6):  # 001-005
for model_id in range(1):
    # model_num = f"{model_id:03d}"
    model_num = f"baseline"
    lora_path = lora_template.format(model_id)
    
    for dataset_key, dataset_info in datasets.items():
        input_dir = dataset_info["input_dir"]
        gpu_id = dataset_info["gpu"]
        dataset_name = dataset_info["name"]
        
        # 构建输出目录
        output_dir = f"{output_base}/Qwen-Image-Edit2509_ablation_{dataset_name}_lora128_{model_num}_ens_syn"
        
        # 脚本文件名
        script_name = f"generated_scripts/eval_{model_num}_{dataset_key}.sh"
        
        # 生成脚本内容
        script_content = f"""#!/bin/bash
# Generated automatically - Model {model_num} on {dataset_key}
export CUDA_VISIBLE_DEVICES={gpu_id}
echo "Starting evaluation for model {model_num} on {dataset_key} with guidance_scale=5.0"
echo "Model: {model_path}"
echo "LoRA: {lora_path}"
echo "Input: {input_dir}"
echo "Output: {output_dir}"
echo "Started at: $(date)"

python inference_qwen_image_edit_our.py \\
    --model_path {model_path} \\
    --input_dir {input_dir} \\
    --output_dir {output_dir} \\
    --num_inference_steps 50 \\
    --guidance_scale 5.0

echo "Completed at: $(date)"
echo "Results saved to: {output_dir}"
"""
        
        # 写入文件
        with open(script_name, 'w') as f:
            f.write(script_content)
        
        # 添加执行权限
        os.chmod(script_name, 0o755)
        
        print(f"Generated: {script_name}")

print("\n✓ All 20 scripts generated in 'generated_scripts' directory!")
print("\nTo run all scripts for model 001:")
print("  cd generated_scripts && ./eval_001_scut.sh & ./eval_001_syn.sh & ./eval_001_laion.sh & ./eval_001_prim.sh & wait")
print("\nTo run all models sequentially:")
print("  cd generated_scripts && for i in {001..005}; do ./eval_${i}_scut.sh & ./eval_${i}_syn.sh & ./eval_${i}_laion.sh & ./eval_${i}_prim.sh & wait; done")