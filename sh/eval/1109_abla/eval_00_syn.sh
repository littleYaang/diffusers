#!/bin/bash
# Generated automatically
export CUDA_VISIBLE_DEVICES=1
echo "Starting evaluation for base with guidance_scale=5.0"
echo "Model: /mnt/hero_blob/preweight/Qwen/Qwen/Qwen-Image-Edit-2509"
echo "Output: /mnt/hero_blob/TESTRESULTS/qwen_baselines/Qwen-Image-Edit2509_iprompt_syn_lora128_000008_scut_enstest"
echo "Started at: $(date)"

python inference_qwen_image_edit_our.py \
    --model_path /mnt/hero_blob/preweight/Qwen/Qwen/Qwen-Image-Edit-2509 \
    --input_dir /mnt/hero_blob/DATASET/SynthText/Syn-Text/syn_test \
    --output_dir /mnt/hero_blob/TESTRESULTS/qwen_baselines/Qwen-Image-Edit2509_iprompt_syn_lora128_000008_scut_enstest \
    --lora_path /home/v-qinhyang/code/hero_blob/qwenoutput/plain_lora/edit_plus_scut2749_lora_128d_5e-5_16ep-000008.safetensors \
    --num_inference_steps 50 \
    --guidance_scale 5.0 \

echo "Completed at: $(date)"
echo "Results saved to: /mnt/hero_blob/TESTRESULTS/qwen_baselines/Qwen-Image-Edit2509_iprompt_syn_000008_lora128_scut_enstest"
