#!/bin/bash
# Evaluation script for edit2509 with guidance_scale=5.0
# Generated automatically
export CUDA_VISIBLE_DEVICES=2
echo "Starting evaluation for edit2509 with guidance_scale=5.0"
echo "Model: /mnt/hero_blob/preweight/Qwen/Qwen/Qwen-Image-Edit-2509"
echo "Output: /mnt/hero_blob/TESTRESULTS/SCUT-EnsText_test/a100_test"
echo "Started at: $(date)"

python qwen_image_removetext_eval.py \
    --model_path /mnt/hero_blob/preweight/Qwen/Qwen/Qwen-Image-Edit-2509 \
    --input_dir /mnt/hero_blob/DATASET/SynthText/SCUT-EnsText_test \
    --output_dir /mnt/hero_blob/TESTRESULTS/SCUT-EnsText_test/a100_test \
    --prompt "remove the text" \
    --num_inference_steps 50 \
    --guidance_scale 5.0 \

echo "Completed at: $(date)"
echo "Results saved to: /mnt/hero_blob/TESTRESULTS/SCUT-EnsText_test/a100_test"
