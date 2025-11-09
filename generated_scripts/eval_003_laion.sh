#!/bin/bash
# Generated automatically - Model 003 on laion
export CUDA_VISIBLE_DEVICES=2
echo "Starting evaluation for model 003 on laion with guidance_scale=5.0"
echo "Model: /mnt/hero_blob/preweight/Qwen/Qwen/Qwen-Image-Edit-2509"
echo "LoRA: /home/v-qinhyang/code/hero_blob/qwenoutput/plain_lora/edit_plus_ablation_ens_syn_003_lora_128d_5e-5_16ep-000004.safetensors"
echo "Input: /mnt/hero_blob/DATASET/SynthText/AnyTE_bench/laion_bench/all"
echo "Output: /mnt/hero_blob/TESTRESULTS/qwen_baselines/Qwen-Image-Edit2509_ablation_laion_lora128_003_ens_syn"
echo "Started at: $(date)"

python inference_qwen_image_edit_our.py \
    --model_path /mnt/hero_blob/preweight/Qwen/Qwen/Qwen-Image-Edit-2509 \
    --input_dir /mnt/hero_blob/DATASET/SynthText/AnyTE_bench/laion_bench/all \
    --output_dir /mnt/hero_blob/TESTRESULTS/qwen_baselines/Qwen-Image-Edit2509_ablation_laion_lora128_003_ens_syn \
    --lora_path /home/v-qinhyang/code/hero_blob/qwenoutput/plain_lora/edit_plus_ablation_ens_syn_003_lora_128d_5e-5_16ep-000004.safetensors \
    --num_inference_steps 50 \
    --guidance_scale 5.0

echo "Completed at: $(date)"
echo "Results saved to: /mnt/hero_blob/TESTRESULTS/qwen_baselines/Qwen-Image-Edit2509_ablation_laion_lora128_003_ens_syn"
